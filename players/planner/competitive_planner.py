#导入库与全局变量
import contextlib
import os
import time
with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stderr(devnull):
        import gym
import f110_gym
import numpy as np
from typing import List
from tqdm import tqdm
from pathlib import Path
import yaml

import warnings
# 引入各种规划器生成器
from players.planner.spliner.spliner_generator import spliner_based_multi_generator
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained since 2022.*")

from players.planner.mpc_planner.mpc_generator import mpc_based_multi_generator
from players.utils.offline_save_video import OfflineRaceVideoRenderer, VehicleState
from players.planner.End2Race.utils import create_planner_render_callback
from players.utils.planner_registry import planners
from players.planner.planner_generators import ocp_based_multi_generator

from players.planner.flow_planner.flow_trajectory_generator import build_flow_context, flow_based_generator
from players.planner.lattice_planner.lattice_generator import lattice_multi_generator
from players.planner.End2Race.end2race_generator import end2race_mppi_generator, compute_direct_ego_mppi_opp_actions

from players.utils.common import (
    _compute_raw_progresses,
    _nearest_agents_by_ref_projection,
    _update_multi_progresses,
    compute_multi_control_actions,
    compute_control_actions,
    initialize_multi_agents_positions,
)
from players.utils.gym_helper import (
    save_data,
    initialize_refline,
    generate_output_filename,
)

# ============================================================================
# Render State渲染阶段
# ============================================================================
# 定义渲染时的全局状态字典（方向盘角度、速度等）
RENDER_INFO_TEMPLATE = {
    "ego_steer": 0.0,
    "ego_speed": 0.0,
    "opp_steer": 0.0,
    "opp_speed": 0.0,
    "track_name": None,
}

render_info = RENDER_INFO_TEMPLATE.copy()
draw_grid_pts = []
draw_traj_pts = []
draw_sample_traj_pts = []
draw_select_sample_traj_pts = []

# Kept as globals for render/debug compatibility with create_planner_render_callback.
# 为了兼容 create_planner_render_callback 的渲染与调试功能，将其保留为全局变量。
ego_planner = None
opp_planner = None


    # 处理赛道首尾相连的“环绕”问题。
    # 如果赛道长100m，自车在0m，对手在99m。对手其实就在自车后面1m，而不是领先99m。
    # 这个函数把相对的进度差转换为 [-0.5*L, 0.5*L] 的带符号距离。

def _align_initial_progresses_to_ego(raw_progresses, refline_total_length):
    """Align initial multi-agent progress values to ego with signed circular offsets."""
    raw = np.asarray(raw_progresses, dtype=float)
    current = raw.copy()
    lap_counts = np.zeros(raw.shape[0], dtype=int)
    if raw.shape[0] <= 1:
        return current, lap_counts

    track_len = float(refline_total_length)
    ego_s = float(raw[0])

    for i in range(1, raw.shape[0]):
        signed_delta = (
            (float(raw[i]) - ego_s + 0.5 * track_len) % track_len
        ) - 0.5 * track_len
        current[i] = ego_s + signed_delta
        lap_counts[i] = int(round((current[i] - float(raw[i])) / track_len))

    return current, lap_counts


# ============================================================================
# End2Race Data Collection Helpers
# ============================================================================
# End2Race open-source demonstration.py records ego-only demonstrations at 10 Hz:
#   time, steer, desired_speed, lidar_0, ..., lidar_359
# The final file writing is intentionally left to players.utils.gym_helper.save_data.
END2RACE_EGO_AGENT_IDX = 0
END2RACE_DEFAULT_SAMPLE_INTERVAL = 0.1
END2RACE_DEFAULT_NUM_LIDAR_BEAMS = 360
    #把高分辨率的激光雷达（1440线）均匀降采样到360线，为了兼容End2Race数据集
def _downsample_lidar_for_end2race(scan, target_points=END2RACE_DEFAULT_NUM_LIDAR_BEAMS):
    
    scan = np.asarray(scan, dtype=np.float32).reshape(-1)
    target_points = int(target_points)

    if target_points <= 0 or scan.size == 0:
        raise ValueError("target_points must be positive and scan must not be empty")
    if scan.size == target_points:
        return scan

    # For 1440 -> 360 this is equivalent to evenly selecting one ray per degree.
    indices = np.linspace(0, scan.size - 1, target_points).round().astype(np.int64)
    return scan[indices].astype(np.float32, copy=False)

    #收集自车的状态、动作和雷达数据，用于后续训练End2Race模型
def _append_end2race_sample( collected_data: List,
    *,
    record_time: float,
    obs,
    action,
    ego_agent_idx: int = END2RACE_EGO_AGENT_IDX,
    target_lidar_points: int = END2RACE_DEFAULT_NUM_LIDAR_BEAMS,
) -> None:
    
    if "scans" not in obs:
        raise KeyError("obs does not contain 'scans'; End2Race data collection needs ego LiDAR")

    lidar_ego = _downsample_lidar_for_end2race(
        obs["scans"][ego_agent_idx],
        target_points=target_lidar_points,
    )

    ego_action = np.asarray(action[ego_agent_idx], dtype=float).reshape(-1)
    steer = float(ego_action[0])
    desired_speed = float(ego_action[1])

    collected_data.append(
        [round(float(record_time), 4), steer, desired_speed]
        + lidar_ego.astype(float).tolist()
    )

# ============================================================================
# Main Simulation Function主函数
# ============================================================================
def run_competitive_players(args, return_metrics=False) -> None:
    global ego_planner, opp_planner # 声明全局变量，方便渲染回调函数读取


    if "flow" in str(args.method).lower():
        with open("players/config/flow_config.yaml", "r") as f:
            flow_cfg = yaml.safe_load(f)

        for k, v in flow_cfg.items():
            if not hasattr(args, k):
                setattr(args, k, v)

    rng = np.random.default_rng(args.rand_seed) # 初始化随机数种子
    steer_limits = (-args.delta_max, args.delta_max)# 从配置读取最大转向角

    # ========================================================================
    # Setup Phase算法装配区
    # ========================================================================
    
        # 1. 从注册表（planner_registry）中提取算法选择自车规划器
    try:
        planner_func = planners[args.planner_family]
    except KeyError:
        raise ValueError(f"Unsupported method: {args.planner_family}")
        
        
        # 2. 为对手选择基础规划器（对手通常不用复杂的博弈论，用简单的MPPI即可）
    # Only used to initialize opponent MPPI solvers; it is not the IBR process itself.
    if "lattice" in args.method:
        opp_planner_func = planners["lattice_planner"]
    elif "mpc" in args.method:
        opp_planner_func = planners["mpc"]
    elif "spliner" in args.method:
        opp_planner_func = planners["spliner"]
    else:
        opp_planner_func = planners["mppi"]

    render_info.update({"track_name": args.map_name})

    rollout_trace = []
    planned_traj_records = []


    # --------------------- planner config selection ---------------------
    # Normal run: read planner defaults from config files.
    ego_ocp_conf = None
    opp_ocp_conf = None
    ego_game_conf = None
    opp_game_conf = None
    # --------------------- planner config selection ---------------------

    #创建每辆车的规划器
    num_agents = int(getattr(args, "num_agents"))

    # 3. 为每一辆车（自车和对手）实例化规划器对象
    agent_planners = []
    config_directory = None

    for agent_idx in range(num_agents):
        if agent_idx == 0:# 自车（Ego）# 读取自车的赛道线、速度缩放等...
            raceline_i = args.raceline
            v_scale_i = args.v_global_limit
            ocp_conf_i = ego_ocp_conf
            game_conf_i = ego_game_conf

            # Ego follows args.method.
            setup_fn_i = planner_func[0]
            biased_type_i = args.mppi_bias_mode_ego
        else:# 对手（Opponent）
            raceline_i = args.opp_raceline
            v_scale_i = args.opp_speed_scale * args.v_global_limit
            ocp_conf_i = opp_ocp_conf
            game_conf_i = opp_game_conf

            # All opponents are initialized as BiasedMPPIPlanner.
            setup_fn_i = opp_planner_func[0]
            biased_type_i = args.mppi_bias_mode_opp
        
        
         # 实例化！真正创建规划器
        planner_i, config_directory_i = setup_fn_i(
            args,
            args.map_name,
            raceline_i,
            config_path=args.config,
            v_scale=v_scale_i,
            ocp_conf=ocp_conf_i,
            game_block_conf=game_conf_i,
            biased_type=biased_type_i,
        )

        if config_directory is None:
            config_directory = config_directory_i

        agent_planners.append(planner_i)

    # Keep old globals for render/debug compatibility.“为了保持渲染（render）和调试（debug）功能的兼容性，保留旧的全局变量。”
    ego_planner = agent_planners[0]
    opp_planner = agent_planners[1] if num_agents > 1 else None

    # ========================================================================
    # Environment and Rendering Setup
    # ========================================================================
       # 1. 创建 F1TENTH Gym 仿真环境
    env = gym.make(
        "f110-v0",
        map=ego_planner.map_path,
        map_ext=".png",
        timestep=args.dt,
        num_agents=num_agents,
    )
    #初始化渲染回调
    if args.render:
        def get_ego_planner():
            return ego_planner

        render_callback = create_planner_render_callback(
            render_info,
            get_ego_planner,
            draw_grid_pts,
            draw_traj_pts,
            draw_sample_traj_pts,
            draw_select_sample_traj_pts,
            margin=args.render_margin,
        )
        env.add_render_callback(render_callback)

    # Initialize refline.
    # 2. 初始化赛道中心线 (refline) 和车辆初始位置
    refline, refline_total_length = initialize_refline(
        config_directory,
        args.map_name,
        refline=args.raceline,
    )

    # Initialize agent positions.
    agent_poses, opp_idx = initialize_multi_agents_positions(
        agent_planners,
        args.ego_idx,
        args.interval_idx,
        rng,
    )
    if num_agents == 1:
        opp_idx = 0

    # Reset environment.# 3. 重置环境（发车）
    obs, _, done, _ = env.reset(poses=agent_poses)

    # Render initial state. This only works for gym-based rendering,
    # not for OfflineRaceVideoRenderer.
    # “渲染初始状态。此功能仅适用于基于 Gym 的渲染方式，不适用于 OfflineRaceVideoRenderer（离线竞速视频渲染器）。”
    if args.render:
        env.render()
    # 4. 如果配置了保存视频，初始化视频渲染器
    video_renderer = None
    map_yaml_path = f"{ego_planner.map_path}.yaml"
    if args.save_video:
        raceline_xy = np.stack(
            [ego_planner.waypoints[:, 0], ego_planner.waypoints[:, 1]],
            axis=1,
        )
        video_renderer = OfflineRaceVideoRenderer(
            map_yaml_path=map_yaml_path,
            refline_xy=raceline_xy,
            boundary_offsets=ego_planner.d_right_left,
            output_fps=args.out_fps,
            vehicle_length=args.length,
            vehicle_width=args.width,
            vehicle_ref_offset=0.0,
            hud_speed_max=args.v_max*args.v_global_limit,
            steer_limit=args.delta_max,
        )

    # ========================================================================
    # State Tracking Initialization
    # ========================================================================
    # 5. 计算发车时每辆车的初始赛道进度 (s)
    initial_raw_progresses = _compute_raw_progresses(obs, refline)  # shape: [N_agents]
    current_progresses, lap_counts = _align_initial_progresses_to_ego(
        initial_raw_progresses,#车辆在参考线上投影得到的原始进度。
        refline_total_length,#经过圈数对齐后的实际累计进度
    )
    prev_raw_progresses = initial_raw_progresses.copy()

    current_ego_progress = float(current_progresses[0])

    #根据车辆数量设置初始状态  
    initial_state = "multi_agent_tracking" if num_agents > 1 else "single_agent_tracking"
    current_state = initial_state
    final_state = initial_state

    num_ibr_iters = args.ibr_time

    # ========================================================================
    # Simulation Loop Setup
    # ========================================================================
    laptime = 0.0               #当前仿真时间
    collected_data: List = []   #End2Race采集数据

    # End2Race-compatible demonstration collection.
    # Default is off so normal evaluation remains unchanged unless explicitly enabled.
    collect_end2race_data = args.end2race
    end2race_sample_interval = END2RACE_DEFAULT_SAMPLE_INTERVAL
    end2race_num_lidar_beams = END2RACE_DEFAULT_NUM_LIDAR_BEAMS
    next_end2race_record_time = end2race_sample_interval

    video_frames: List = []     #视频帧
    collision_occurred = False  #是否发生碰撞
    # 获取底层控制器的控制频率（一次规划跑几步物理仿真）
    tracker_steps = (   
        #每次规划完成后，车辆连续执行多少个跟踪控制步骤，才重新规划一次。
        ego_planner.conf.tracker_steps
        if hasattr(ego_planner, "conf") and hasattr(ego_planner.conf, "tracker_steps")
        else ego_planner.tracker_steps
    )
    video_step_count = 0
      # 初始化进度条
    pbar = tqdm(
        total=args.sim_duration,
        desc="Simulation",
        unit="s",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| "
            "{n:.2f}/{total:.1f} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        ),
        mininterval=5.0,
        miniters=1,
        disable=args.end2race or args.collect_pairwise_rows, 
    ) 

    flow_context = None
    if "flow" in str(args.method):
        flow_context = build_flow_context(args)




    #满足以下两个条件，仿真就继续：
    #- 没有发生终止事件。
    #- 当前仿真时间没有超过设定时长。
    while not done and laptime < args.sim_duration:
        # ====================================================================
        # Planning Phase (executed once per tracker cycle) A. 规划阶段 (Planning Phase) - 慢速循环
        # ====================================================================
        raw_progresses = _compute_raw_progresses(obs, refline)# 计算当前所有车的进度
        if num_agents > 1:
            nearest_agent_indices = _nearest_agents_by_ref_projection(# 找到离自车最近的车
                raw_progresses,
                refline_total_length,
            )
        else:
            nearest_agent_indices = np.zeros(1, dtype=int)




        # 把当前环境状态打包成一个字典
        params_dict = {
            "num_ibr_iters": int(num_ibr_iters),#IBR迭代次数
            "agent_progresses": current_progresses.#车辆修正进度
            copy(), # with lap counting and wraparound handled, shape: [N_agents]
            "agent_raw_progresses": raw_progresses.copy(),#车辆原始进度用于计算最近车辆
            "nearest_agent_indices": nearest_agent_indices.copy(),#最近的车辆
            "agent_planners": agent_planners,#车辆策略
            "ego_agent_idx": 0,#自车编号
        }

        if flow_context is not None:
            params_dict.update(flow_context)

        generator = planner_func[1] # 调用轨迹生成器（比如 mppi_based_multi_generator）

        # 调用生成器！这里会执行昂贵的 GPU 采样、博弈代价计算和 IBR 迭代
        #第一类多车轨迹生成器
        if generator is lattice_multi_generator \
            or generator is ocp_based_multi_generator \
            or generator is mpc_based_multi_generator \
            or generator is spliner_based_multi_generator \
            or generator is end2race_mppi_generator:
            # 返回的 best_trajs 包含了自车和对手未来 H 步的轨迹坐标
            best_trajs, infer_time = generator(
                args=args,
                agent_planners=agent_planners,
                obs=obs,
                params_dict=params_dict,
            )
        #Flow轨迹生成器
        elif generator is flow_based_generator:
            # print(f"[INFO] Running flow-based generator...")
            ego_best_traj, opponent_trajs_by_agent_idx, infer_time = generator(
                args=args,
                ego_planner=ego_planner,
                opp_planner=opp_planner,
                obs=obs,
                params_dict=params_dict,
            )

            best_trajs = [None for _ in agent_planners]
            best_trajs[0] = ego_best_traj

            for agent_idx, traj in opponent_trajs_by_agent_idx.items():
                best_trajs[int(agent_idx)] = traj

        else:
            raise ValueError(f"Unsupported planner generator: {generator}")

        del infer_time


        #保存规划轨迹,转化为numpy数组
        ego_record_traj = None if best_trajs[0] is None else np.asarray(best_trajs[0], dtype=np.float32)
        #如果轨迹维度大于2
        if ego_record_traj is not None and ego_record_traj.ndim == 2 and ego_record_traj.shape[1] > 2:
            ego_record_traj = ego_record_traj[:, :2]#只保留前两列

        #录的是规划轨迹
        planned_traj_records.append({
            "trace_start_index": int(len(rollout_trace)),
            "trajs": [None if ego_record_traj is None else ego_record_traj.copy()],
        })

        ego_best_traj = best_trajs[0]
        opp_best_traj = best_trajs[1] if num_agents > 1 else None

        # ====================================================================
        # Tracking Phase (multiple steps following the planned trajectory)B. 跟踪阶段 (Tracking Phase) - 快速循环
        # ====================================================================
        
        # 规划出来的轨迹很长，我们不需要每一步都重新规划，只需要跟着这条轨迹走几步
        for tracker_count in range(tracker_steps):
            if done or laptime >= args.sim_duration:
                break

            #直接动作控制
            if getattr(ego_planner, "direct_action", False):
                action = compute_direct_ego_mppi_opp_actions(obs, agent_planners, best_trajs, steer_limits)
            # for only 1 agent只有一辆车
            elif int(num_agents) == 1: 
                steer, speed = ego_planner.tracker.plan(
                    obs["poses_x"][0],
                    obs["poses_y"][0],
                    obs["poses_theta"][0],
                    obs["linear_vels_x"][0],
                    ego_best_traj,
                )

                steer = np.clip(steer, steer_limits[0], steer_limits[1])#转向角限制
                action = np.asarray([[steer, speed]], dtype=float)#最终形成动作


            elif int(num_agents) > 2: # for > 2 agents# 多车控制！计算每辆车为了跟踪各自轨迹所需的底层指令
                action = compute_multi_control_actions(
                    obs=obs,
                    agent_planners=agent_planners,
                    best_trajs=best_trajs,
                    nearest_agent_indices=nearest_agent_indices,
                    steer_limits=steer_limits,
                )


            else: # for 2 agents# 两车控制
                action = compute_control_actions(
                    obs,
                    ego_planner,
                    opp_planner,
                    ego_best_traj,
                    opp_best_traj,
                    steer_limits=steer_limits,
                )

            # Update render information.# 记录视频帧
            if args.render or args.save_video:
                render_info.update({
                    "ego_steer": action[0, 0],
                    "ego_speed": action[0, 1],
                    "opp_steer": action[1, 0] if num_agents > 1 else 0.0,
                    "opp_speed": action[1, 1] if num_agents > 1 else 0.0,
                })

            # Step environment.更新环境
            obs, timestep, done, _ = env.step(action)

            video_step_count += 1
            if (
                args.save_video
                and video_renderer is not None
                and video_step_count % args.video_capture_every == 0
            ):
                num_render_agents = len(obs["poses_x"])

                vehicle_states = [
                    VehicleState(
                        x=float(obs["poses_x"][i]),
                        y=float(obs["poses_y"][i]),
                        heading=float(obs["poses_theta"][i]),
                    )
                    for i in range(num_render_agents)
                ]
                # 把位置信息传给渲染器
                video_renderer.capture_multi(
                    vehicle_states=vehicle_states,
                    sim_time=laptime,
                    title="Closed-loop Racing",
                    vehicle_trajs=best_trajs,
                    extra_text=None,
                    render_info=render_info.copy(),
                    follow_vehicle_idx=0,
                )

            # Update simulation time.# 更新时间
            prev_laptime = laptime
            laptime += timestep
            if laptime > args.sim_duration:
                laptime = args.sim_duration

            pbar.update(laptime - prev_laptime)

            # End2Race demonstration collection.“End2Race 专家演示数据（示教数据）的采集。”
            if collect_end2race_data:
                # here while is used to strictly make sure the time stamps of recorded samples align with the specified interval, even if there are occasional longer delays in the loop due to environment stepping or planning.“这里使用 while 循环是为了严格确保记录样本的时间戳与指定的时间间隔对齐，即使由于环境步进或规划导致循环中偶尔出现较长的延迟。使用 while 而不是 if，是为了避免某次仿真更新耗时过长时漏掉应该采集的时间点。
                while laptime >= next_end2race_record_time:
                    _append_end2race_sample(
                        collected_data,
                        record_time=next_end2race_record_time,
                        obs=obs,
                        action=action,
                        ego_agent_idx=END2RACE_EGO_AGENT_IDX,
                        target_lidar_points=end2race_num_lidar_beams,
                    )
                    next_end2race_record_time += end2race_sample_interval

            # Handle wraparound. _update_multi_progresses updates lap counting
            # and wraparound internally, so it is the single source of progress. # 更新赛道进度和圈数
            raw_progresses, current_progresses, lap_counts = _update_multi_progresses(
                obs=obs,
                refline=refline,
                refline_total_length=refline_total_length,
                prev_raw_progresses=prev_raw_progresses,
                lap_counts=lap_counts,
            )
            prev_raw_progresses = raw_progresses.copy()

            # Compatibility variable for existing debug/render output.
            current_ego_progress = float(current_progresses[0])
            # 记录轨迹数据
            rollout_trace.append({
                "t": float(laptime),
                "poses_x": np.asarray(obs["poses_x"], dtype=float).tolist(),
                "poses_y": np.asarray(obs["poses_y"], dtype=float).tolist(),
                "poses_theta": np.asarray(obs["poses_theta"], dtype=float).tolist(),
                "action": np.asarray(action, dtype=float).tolist(),
            })
            
            # Keep final_state independent from any selected opponent.：“确保最终状态数据与所选对手解耦，互不干扰。”
            current_state = "multi_agent_tracking" if num_agents > 1 else "single_agent_tracking"
            final_state = current_state

            # Collision detection. # 检查碰撞（如果自车撞了，直接终止循环）
            collisions = np.asarray(obs["collisions"], dtype=bool).ravel()
            collision_detected = bool(collisions[0])
            collision_msg = "[WARNING] Collision detected for ego!"

            if collision_detected:
                done = True
                collision_occurred = True
                print(collision_msg)

            # Update progress bar postfix info.更新进度条补充信息
            pbar.set_postfix({
                "state": current_state,
                "ego_s": f"{current_ego_progress:.2f}",
                "opp_s": f"{current_progresses[nearest_agent_indices[0]]:.2f}" if num_agents > 1 else "N/A",
                "ego_v": f"{obs['linear_vels_x'][0]:.2f}",
            })

            # Rendering.生成当前仿真画面的 RGB 图像。
            if args.render:
                frame = env.render(mode="rgb_array")
                if collect_end2race_data and args.save_gym:
                    video_frames.append(frame)

    pbar.close()

    # ========================================================================
    # Post-Simulation Processing
    # ========================================================================
        # 打印最终的所有配置参数
    print("=" * 80, flush=True)
    print("[ARGS]", flush=True)
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}", flush=True)
    print("=" * 80, flush=True)
    #整理参考线和赛道边界
    refline_xy = np.asarray(ego_planner.waypoints, dtype=np.float32)#赛道参考线坐标
    boundary_offsets = np.asarray(ego_planner.d_right_left, dtype=np.float32)#赛道左右边界偏移量
   
   # 打包本次仿真的所有数据（用于批量评估或画图）
    metrics = {
        "rollout_trace": rollout_trace,
        "planned_trajs": planned_traj_records,
        "refline_xy": refline_xy,
        "boundary_offsets": boundary_offsets,
    }

    output_filename = "" # 保存视频，生成输出文件名
    if args.end2race or args.save_video or not return_metrics:
        output_filename = generate_output_filename(
            args,
            final_state,
            args.opp_raceline,
            args.ego_idx,
            opp_idx,
            args.opp_speed_scale,
        )
    #保存端到端数据
    if args.end2race:
        save_data(
            args,
            collected_data,
            video_frames,
            collision_occurred,
            final_state,
            output_filename,
            laptime,
            opp_idx,
        )
    #保存视频
    if args.save_video and video_renderer is not None:
        video_path = os.path.join("results", f"{output_filename}.mp4")
        video_renderer.save(video_path)
        video_renderer.close()
    
    
    # 如果开启了批量测试返回指标，就把 metrics 返回去
    if return_metrics:
        return metrics

    print(f"[INFO] Results saved with filename: {output_filename}")


if __name__ == "__main__":
    # This would be called from trainer/main script with args.
    pass
