import time
from typing import Dict, List
from players.planner.flow_planner.flow_trajectory_generator import flow_based_generator

import numpy as np

# 是否启用可行性选择（Feasibility Selection，即剔除撞墙和撞车的轨迹）
# 如果改成 False，就相当于论文里的消融实验 "S-GTP w/o Feasibility Selection"
MPPI_POST_SELECT = True  # Whether to enable feasibility selection on ego trajectories.
# MPPI_POST_SELECT = False  # Whether to enable feasibility selection on ego trajectories.


# 是否把对手的预测轨迹传给规划器。
# 如果改成 False，车辆就不会考虑对手，只会傻傻地沿着赛道中心线跑
USE_OPP_PRED = True  # Whether to pass predicted opponent trajectories to the planner.
# USE_OPP_PRED = False  # Disable opponent-trajectory predictions; _other_trajs returns None.

# ---------------------------------------------------------------------------
# Inlined generator utilities内联的生成器工具函数
# ---------------------------------------------------------------------------


#解析方法名后缀，判断当前是什么交互模式（ibr, opp-reactive, nonreactive）
def resolve_current_method_mode(method: str) -> str:
    """Resolve method suffix into the interaction mode used by the generator."""
    method = str(method)
    if method.endswith("_ibr"):
        return "ibr"
    if method.endswith("_opp-reactive"):
        return "opp-reactive"
    if method.endswith("_nonreactive"):
        return "nonreactive"
    raise ValueError(
        f"Could not resolve interaction mode from method name '{method}'. "
        "Please specify --interaction_mode explicitly or use method names with "
        "suffixes like '_ibr' or '_opp-reactive'."
    )



# 获取当前交互模式的入口
def get_interaction_mode(args) -> str:
    """Return nonreactive, opp-reactive, or ibr."""
    mode = args.interaction_mode
    return resolve_current_method_mode(args.method) if mode is None else str(mode).lower()


# 从环境观测 obs 里，把除了自己（agent_idx）之外的其他所有车的坐标 [x, y, theta] 提取出来
def collect_other_poses_from_obs(obs: Dict, agent_idx: int) -> np.ndarray:
    """Return current poses of all non-controlled agents as [M, 3]."""

    def num_agents_from_obs(obs: Dict) -> int:
        return len(obs["poses_x"])

    return np.asarray(
        [
            [
                float(obs["poses_x"][j]),
                float(obs["poses_y"][j]),
                float(obs["poses_theta"][j]),
            ]
            for j in range(num_agents_from_obs(obs))
            if j != agent_idx
        ],
        dtype=float,
    )

# 把除了自己之外的其他车的预测轨迹提取出来
def collect_other_trajs(trajs: List[np.ndarray], agent_idx: int) -> List[np.ndarray]:
    """Return predicted trajectories of all agents except agent_idx."""
    return [
        trajs[j]
        for j in range(len(trajs))
        if j != agent_idx and trajs[j] is not None
    ]


# Short local helpers used below; these keep the IBR loop visually compact.
_other_poses = collect_other_poses_from_obs

# 带全局开关的轨迹提取：如果 USE_OPP_PRED = False，直接返回 None"
def _other_trajs(trajs: List[np.ndarray], agent_idx: int):
    """Return opponent predictions, or None when opponent prediction is disabled."""
    if not USE_OPP_PRED:
        return None
    return collect_other_trajs(trajs, agent_idx)



#核心函数循环
# args	全局配置，包括交互模式、MPPI 偏向模式等
# agent_planners	所有车辆对应的规划器对象   players/planner/competitive_planner.py:202 初始化，:241 逐个 append
# obs	当前仿真环境的观测信息 env.step() 返回，f1tenth_gym/gym/f110_gym/envs/f110_env.py:333 / :389 → return obs, reward, done, info	仿真器输出的 dict，含 poses_x / poses_y / poses_theta / linear_vels_x
# params_dict	主程序传入的规划参数，包含 IBR 迭代次数等  players/planner/competitive_planner.py:407-415
def ocp_based_multi_generator(args, agent_planners, obs, params_dict):
    """Multi-agent MPPI generator with compact synchronous IBR.

    The loop is intentionally explicit:
        selfish prediction -> L synchronous best-response updates.
        “带有紧凑型同步 IBR（迭代最佳响应）的多智能体 MPPI 生成器。
        这个循环被刻意写得非常显式（直白）：
        自私预测 -> L 次同步最佳响应更新。”
    """
    mode = get_interaction_mode(args)# 获取交互模式
    ego_idx = 0# 规定第 0 辆车是自车


    # 内部辅助函数：获取某辆车的偏向模式（ancillary / none）
    def get_agent_bias_mode(agent_idx):
        if agent_idx == ego_idx:
            return str(args.mppi_bias_mode_ego).lower()
        return str(args.mppi_bias_mode_opp).lower()


        # 内部辅助函数：让某辆车真正去算一次轨迹
    def plan_once(agent_idx, opp_pred_traj, post_select, round_name, bias_override):
        del round_name  # Kept in the signature for readable call sites.

        if bias_override is not None:
            bias_mode = str(bias_override).lower()
        else:
            bias_mode = get_agent_bias_mode(agent_idx)

        if bias_mode not in {"none", "ancillary"}:
            raise ValueError(
                "Unsupported MPPI bias_mode after cleanup: "
                f"{bias_mode!r}. Use 'none' or 'ancillary'."
            )

        planner = agent_planners[agent_idx]        
        # 调用了底层 MPPI 规划器的 plan 方法
        # 注意这里的输入：如果 opp_pred_traj 不是 None，说明它拿到了对手的轨迹，开始博弈
        return planner.plan(
            # ... 传入当前状态 ..
            float(obs["poses_x"][agent_idx]),
            float(obs["poses_y"][agent_idx]),
            float(obs["poses_theta"][agent_idx]),
            _other_poses(obs, agent_idx),
            float(obs["linear_vels_x"][agent_idx]),
            opp_pred_traj=opp_pred_traj, # 对手预测轨迹
            post_select=post_select,# 是否开启安全性过滤
            bias_mode=bias_mode,
        )

    initial_plan_result_by_agent_idx = [None for _ in range(len(agent_planners))]

    # First build unbiased/selfish predictions for every agent. These trajectories
    # are then used by the interaction modes below as opponent predictions.
 
    # 第一轮：所有人都不考虑别人，自私地跑一次


    for agent_idx in range(len(agent_planners)):
        initial_plan_result_by_agent_idx[agent_idx] = plan_once(
            agent_idx=agent_idx,
            opp_pred_traj=None, # No opponent predictions for the initial round, to get truly selfish trajectories regardless of the MPPI bias settings for each agent.
            # 重点：此时还没有对手轨迹，传 None
            post_select=MPPI_POST_SELECT,
            round_name="initial_none",
            bias_override="none", # Explicitly override bias_mode to "none" for the initial prediction round, to get truly unbiased/selfish trajectories regardless of the MPPI bias settings for each agent.# 强制纯 MPPI 采样
        )

    # 把所有人第一轮算出来的轨迹存到 curr 里
    curr = [
        initial_plan_result_by_agent_idx[i][0]
        for i in range(len(agent_planners))
    ]

    t0 = time.time()
    # nonreactive: compute selfish trajectories, then run one
    # safety-aware replanning pass against those fixed predictions.
    """模式分支:Nonreactive(无反应)"""
    if mode == "nonreactive":
        out = list(curr)
        for i in range(len(agent_planners)):
            out[i] = plan_once(
                agent_idx=i,
                opp_pred_traj=_other_trajs(curr, i), # Use the initial selfish predictions as opponent predictions.
                # 把上一轮的轨迹当做对手位置
                post_select=MPPI_POST_SELECT,
                round_name="nonreactive",
                bias_override=None, # Use the default bias mode, which can be configured separately for ego and opponents.
            )[0]
        infer_time = time.time() - t0
        return out, infer_time


    """ 模式分支Opp-reactive对手反应"""
    if mode == "opp-reactive":
        out = list(curr)
        for i in range(len(agent_planners)):
            if i != ego_idx:# 重点：只有对手更新了轨迹，自车不更新！
                out[i] = plan_once(
                    agent_idx=i,
                    opp_pred_traj=_other_trajs(curr, i),
                    post_select=MPPI_POST_SELECT,
                    round_name="opp-reactive",
                    bias_override=None, # Use the default bias mode for the opponent agents, which can be configured separately for ego and opponents.
                )[0]
        infer_time = time.time() - t0
        return out, infer_time


    """模式分支:IBR迭代最佳响应,SGTP 的灵魂！）"""
    if mode != "ibr":
        raise ValueError(f"Unsupported interaction mode: {mode!r}")

    # IBR: every agent responds to the same clean trajectory bank
    # from the previous round.    # 开始 IBR 迭代（对应论文里的 L_IBR 轮）
    for ibr_iter in range(int(params_dict["num_ibr_iters"])):
        prev = curr# 保存上一轮的轨迹
        curr = [
            plan_once(
                agent_idx=i,
                opp_pred_traj=_other_trajs(prev, i),# 基于对手上一轮的轨迹，计算本轮的最佳响应
                post_select=MPPI_POST_SELECT,
                round_name=f"ibr_{ibr_iter}",
                bias_override=None, # Use the default bias mode for all agents in the IBR rounds, which can be configured separately for ego and opponents.
            )[0]
            for i in range(len(agent_planners))
        ]


   # 最后一步：自车的额外响应（Extra Ego Response）
    curr[ego_idx] = plan_once(
        agent_idx=ego_idx,
        opp_pred_traj=_other_trajs(curr, ego_idx),
        post_select=MPPI_POST_SELECT,
        round_name="final_ego_response",
        bias_override=None, # Use the default bias mode for the ego agent in the final response, which can be configured separately for ego and opponents.
    )[0]

    infer_time = time.time() - t0
    # print(f"IBR inference time: {infer_time:.3f}s for {len(agent_planners)} agents, {params_dict['num_ibr_iters']} IBR iterations.")
    return curr, infer_time
