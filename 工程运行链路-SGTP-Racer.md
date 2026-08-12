# SGTP-Racer 工程运行链路：从启动文件到论文功能落地

> 本文回答一个问题：**工程跑起来之后，程序从哪个文件开始、按什么顺序穿过哪些文件，把论文（arXiv 2607.25388）所述的功能逐条实现出来。**
> 视角是**动态的运行时调用链**——静态的仓库结构见同目录 `工程脉络-SGTP-Racer.md`，公式的逐行定位见 `论文公式与代码对应-任务二.md`，本文只讲"程序怎么跑"。
> 参数互证来源：论文 §IV-B（Implementation and Racing Scenarios）逐条与本仓库 `players/config/*.yaml` 对照。

---

## 0. 一句话总览

```
python players/run_multi_method.py          ← 唯一的主入口
   └─ competitive_planner.run_competitive_players()      装配 + 主循环
        └─ planner_generators.ocp_based_multi_generator()   论文 Algorithm 1
             └─ biased_mppi_post_selection.plan()            一次最佳响应
                  ├─ biased_mppi_planner.plan()                 采样准备
                  │    └─ biased_mppi_core.forward()               GPU 批量 rollout + 打分
                  │         └─ game_block_cost(…)                    博弈四项代价
                  │    └─ post_selection._post_select_mppi_output()  可行性筛选（论文核心创新）
                  └─ （重复 N×(1+L+1) 次 MPPI 求解）
        └─ pure_pursuit.plan()                            轨迹 → (转角, 目标速度)
        └─ f110_env.step(action)                          仿真推进一步（0.01 s）
```

每个 0.01 s 的仿真拍都完整走一遍这条链。论文的"Real-Time IBR at 10 Hz"指的是**单次规划求解**的耗时量级（0.095 s），而这份工程把重规划频率开到了 100 Hz（`tracker_steps=1`），比论文口径更密。

---

## 1. 启动阶段：从命令行到内存里的规划器

### 1.1 入口文件（三个，只有第一个是主线）

| 入口 | 命令 | 用途 | 论文对应 |
|---|---|---|---|
| **`players/run_multi_method.py`** | `python players/run_multi_method.py --map_name Nuerburgring` | 单次实验，出一个 mp4 | 论文 quickstart / 项目页复现入口 |
| `players/run_batched_competitive.py` | 读 `eval_config.yaml` | 批量评测（多地图 × 多起点，进程池） | 论文 §IV 的实验规模（7 test 图 × 多 trial → 95.24% 胜率） |
| `players/batch_render_episode_videos.py` | 读批量产物 | 离线回放视频 | — |

批量入口内部**复用同一个** `run_competitive_players(return_metrics=True)`，所以主线只有一条，下面只讲 `run_multi_method.py`。

### 1.2 启动链逐文件走（装配阶段，只发生一次）

| 步 | 文件 : 行 | 发生什么 | 论文对应 |
|---|---|---|---|
| 1 | `run_multi_method.py:11-24` | argparse 解析 CLI：`map_name / num_agents / ego_idx / sim_duration / interval_idx / mppi_bias_mode_*` | 实验的自由变量 |
| 2 | `:27-70` | 读 `players/config/run_config.yaml`，**yaml 覆盖 CLI 同名项**；拼出 `method = "mppi_ibr"` | 方法选择（SGTP = mppi 家族 + ibr 模式） |
| 3 | `competitive_planner.py:159` | `planners[args.planner_family]` 查注册表 `planner_registry.py:130-141` → `[setup_biased_mppi_planner, ocp_based_multi_generator]` | 论文的开源基准插座：8 种方法都从这里插 |
| 4 | `:164-171` | 按 method 子串决定**对手**用什么规划器（`mppi_ibr` → 对手也是 mppi） | 论文实验：SGTP 场次里所有车都用 SGTP |
| 5 | `:193-227` | 逐个 agent 调 `setup_biased_mppi_planner`（`planner_registry.py:24-58`）：ego 速度比例 0.7（`v_global_limit`），对手再乘 `opp_speed_scale=0.9` | **论文 §IV-B 原文**："The reference speeds of the opponent vehicles are scaled by a factor of 0.9 to promote close-range interactions" ✓ |
| 6 | `biased_mppi_planner.py:41-52` | 规划器构造函数里 `np.loadtxt` 读 `MapZoo/<map>/<map>_racing_line.csv` → `waypoints[N,5]=(x,y,v_ref,ψ,s)` + `d_right_left[2,N]` | **论文的 G（raceline 几何）**，式(8)(10) 的数据地基 |
| 7 | `:100-122` | 构造 `GameBlockCost`（读 `game_cost_config.yaml` 的四项权重）+ 读 `ocp_config.yaml` 全部 MPPI 超参 + 建 `PurePursuitPlanner` | 论文 §IV-B 的全部数值（见 §4 对照表） |
| 8 | `:336-382` | `_build_mppi_core()` 构造 `BiasedMPPI`（`biased_mppi_core.py:39-210`）：预分配 `_state_seq_batch[128,13,4]`、高斯噪声分布、warm start 置零 | 论文 Algorithm 1 line 2 的初始化（"initialized to zero upon reset"） |
| 9 | `competitive_planner.py:236-242` | `gym.make("f110-v0", map, timestep=0.01, num_agents=3)` | **论文 §IV-A 原文**："we use the F1TENTH Gym environment described in [9]" ✓ |
| 10 | `common.py:435-476` | `initialize_multi_agents_positions`：`idx = (ego_idx + i*interval_idx) % N_wp`，ego 取 `ego_idx`，第 i 个对手往前加 `i*interval_idx` | **论文 §IV-B**："vehicles are initialized in a longitudinal queue with a uniform 2m inter-vehicle spacing in ascending Frenet progress, **with the ego vehicle always placed rearmost**" ✓（`interval_idx=20` × 路点间距 ≈0.1 m = 2 m） |
| 11 | `competitive_planner.py:277` | `env.reset(poses)` → 返回 obs | — |
| 12 | `:306-313` + `common.py:713` | `_compute_raw_progresses`（各车投影到参考线取 s）→ `_align_initial_progresses_to_ego`（环形对齐） | 论文胜率指标的地基：`∆s_j = s_{j,T} − s_{j,0}`（§IV-C） |
| 13 | `:284-301` | 若 `save_video=True`，建 `OfflineRaceVideoRenderer`（`offline_save_video.py`） | 项目页视频 |

装配阶段结束时的内存状态：**3 个 `BiasedMPPIPostSelectionPlanner` 实例**（每个自带一套 waypoints/PurePursuit/128 样本缓冲）+ 1 个 gym env + 进度追踪状态。

---

## 2. 运行时主循环：一个 0.01 s 拍子里的完整调用链

主循环在 `competitive_planner.py:363` `while not done and laptime < sim_duration:`，每一拍分四个阶段：

### 2.1 阶段 A｜规划（论文 Algorithm 1 的完整执行）

```
competitive_planner.py:376-401
  params_dict = {num_ibr_iters: 2, agent_progresses, nearest_agent_indices, ...}
  best_trajs, infer_time = ocp_based_multi_generator(args, agent_planners, obs, params_dict)
```

进入 `planner_generators.py:78`，这里就是**论文 Algorithm 1 的逐行翻译**：

| Algorithm 1 行 | 代码 | 说明 |
|---|---|---|
| line 3 初始化（距离式避障） | `:122-129` 每 agent `plan_once(opp_pred_traj=None, bias_override="none")` | 无预测 → 椭圆斥力（`planner:296-322`），论文"Initialize without game-aware costs" |
| line 4-5 for ℓ = 1..L_IBR | `:171` `for ibr_iter in range(2)` | L_IBR = `ibr_time` = 2 ✓（§IV-B） |
| line 6 Ξ̂^ℓ₋ᵢ ← 上一轮全体解 | `:172` `prev = curr`；`:176` `_other_trajs(prev, i)` | **同步（Jacobi）更新**：一轮内所有车对着同一份快照 |
| line 7 生成候选集 𝒞^ℓ | 详见 2.1.1 | GPU 批量采样 + rollout |
| line 8 打分 c^(k) | 详见 2.1.2 | 式(2)-(9) |
| line 9 可行性 h^(k) | 详见 2.1.3 | 式(10)(11) |
| line 10 选 k* | `_select_index_numba` | 式(1) |
| line 11 warm start | `_commit_selected_action_sequence` | 写回 `_previous_action_seq` |
| line 14 ego 额外响应 | `:184-190` `curr[ego_idx] = plan_once(...)` | 论文"Extra ego response"，ego 信息永远比对手新一步 |

**一次拍的求解次数**：3 车 × (1 自私 + 2 IBR + 1 终轮) = **12 次 MPPI 求解**（ego 是 1+2+1=4，对手各 1+2=3）。

#### 2.1.1 一次 `plan_once` 内部（`post_selection.py:406-443` → `planner.py:477-522`）

```
plan_once
 └─ post_selection.plan(post_select=True)          # MPPI_POST_SELECT=True（planner_generators.py:8）
     ├─ planner.plan()                             # 先跑标准 MPPI 全流程
     │   ├─ build_reference()                      # planner.py:405-418 → 式(8) 的参考轨迹 x^r
     │   └─ mppi.forward(x0, info)                 # core.py:440-529
     │       ├─ :455  mean = _previous_action_seq  # warm start（论文 U^ℓ）
     │       ├─ :422-432  ε~N(0,Σu)，mean+ε，clamp # 论文的 Π_𝓤 投影
     │       ├─ :474-499 逐 τ 推演 12 步动力学     # f(·) 运动学自行车（planner.py:384-403）
     │       ├─ :501 _compute_batch_costs          # 详见 2.1.2
     │       └─ :511-517 softmax 加权平均          # 标准 MPPI 输出——**开了后选择后被丢弃**
     └─ _post_select_mppi_output(opp_pred_traj)    # post_selection.py:359-400
         ├─ :365 边界检查 → 式(10)
         ├─ :370 碰撞检查 → 式(11)
         ├─ :376 选索引    → 式(1)/(12) + 回退
         └─ :395 写回 warm start
```

`plan_once` 返回 `(traj[T,3]=(x,y,v), heading)`——这是全仓库统一的规划器输出契约，对手预测也用这个形状。

#### 2.1.2 打分（式(2)-(9) 的三处接力）

```
core.py:501 _compute_batch_costs
 ├─ :295-305  对 τ=0..11 逐步调 planner._cost_func_torch     → 式(8) 单步跟踪/控制/平滑
 ├─ :316-322  末步（零动作）+ Σ_τ                            → 式(9) 前半
 └─ :324-338  + trajectory_cost_func
      └─ planner.py:420-475 _trajectory_game_cost_torch
          ├─ :451-472 对每个对手 j 调 game_block_cost          → 式(2) 的 Σⱼ
          └─ :474 乘 w_game=60                                → 式(9) 后半
               └─ game_block_cost.py:111-151（njit）
                   ├─ :128-131 纵向优势                        → 式(4)
                   ├─ :133-137 缠斗维持                        → 式(3)
                   ├─ :139-142 阻挡对齐                        → 式(5)（⚠️ 取了 |Δd|）
                   └─ :144-148 安全惩罚                        → 式(7)
```

#### 2.1.3 可行性筛选（论文 Fig.1 的"Feasibility Selection"模块）

```
post_selection.py:359-400
 ├─ _boundary_validity_numba(:11-82)     → 式(10)：ρ_bd = min_τ min(w_R+d, w_L−d) − 0.155 − 0.30 ≥ 0.06
 ├─ _collision_validity_one_opp_numba(:86-118) → 式(11)：ρ_obs = min_j min_τ ‖p_i−p̂_j‖ ≥ 0.9
 └─ _select_index_numba(:121-166)
     ├─ :135-142 可行集内取 cost 最小                      → 式(1)/(12)
     └─ :144-166 全不可行 → 取 ν_bd+ν_obs 最小              → 论文 §III-B 回退
```

这一步是 SGTP 与标准 MPPI 的**本质分野**：不在候选间做加权平均（会把"超车"与"防守"糊成一条四不像），而是在可行候选里**选一条**——这就是论文 Fig.1 里"Strategy Diversity"三种模态能平滑切换的机制保证。

### 2.2 阶段 B｜跟踪（轨迹 → 方向盘）

```
competitive_planner.py:438-470
  tracker_steps = 1（run_config.yaml）→ 规划一次只执行一步
  num_agents=3 → compute_multi_control_actions (common.py:508-542)
      └─ 对每辆车切"自己 + 最近邻"的二元组 → 复用 compute_control_actions (:478-506)
          └─ tracker.plan(...)  → pure_pursuit.py:68-83
              ├─ 前视距离 L = v·(maxL−minL)/Lscale + minL   # 速度自适应
              ├─ get_wp_xyv_with_interp (:92-114)           # 路点段内插值找前视点
              └─ get_actuation_PD (common.py:596)            # PD 转向 + 目标速度
  输出 action[3,2] = (转角, 目标速度)
```

**论文 §IV-A 原文互证**："In all cases, the planned trajectories are tracked with the same pure-pursuit controller [9]"——所有 8 种方法共用这一个跟踪器，保证对比公平 ✓。

**一处必须知道的模型失配**：规划器内部推演用运动学自行车（`a, δ`），仿真器执行用 ST 动力学 + PID 速度跟踪（`f1tenth_gym/base_classes.py`），输入是"目标速度"而非加速度。两个模型不共享，中间靠纯跟踪吸收误差。这是论文没写但工程上决定成败的一环。

### 2.3 阶段 C｜仿真推进一步

```
competitive_planner.py:482  obs, timestep, done, _ = env.step(action)
  └─ f1tenth_gym/f110_env.py → Simulator.step
       ├─ 每车 update_pose(steer, vel)   # ST 单轨 + PID
       ├─ LiDAR 1440 束更新（obs["scans"]，本工程主线未用）
       └─ 碰撞/完赛判定（ego 撞车即 done）
```

### 2.4 阶段 D｜记录与终止判定

```
:485-509  每 10 步抓一帧 → OfflineRaceVideoRenderer（10 fps 出 mp4）
:535-545  _update_multi_progresses（common.py:757）→ 圈数/进度唯一真源
:560-567  obs["collisions"][0] 为真 → done（"Collision detected for ego!"）
:363      laptime ≥ sim_duration → 退出
:585-634  打印全部 args、打包 metrics、存 results/<method>_....mp4
```

---

## 3. 论文功能 → 文件落地的总对照

论文 Fig.1 画了四个 Main Modules，Fig.2 画了执行流，Algorithm 1 写了伪代码。全部映射如下：

| 论文模块/内容 | 落地文件 | 关键函数 |
|---|---|---|
| **Real-Time IBR**（10 Hz） | `planner_generators.py` | `ocp_based_multi_generator`（118-194） |
| **Batched Sampling and Ranking**（GPU） | `biased_mppi_core.py` | `_sample_action_sequences` + `forward`（410-529） |
| 采样动力学 rollout（f(·)，式含 β） | `biased_mppi_planner.py` | `_dynamics_torch`（384-403） |
| **Game-Aware Cost**（式(2)-(7)） | `game_block_cost.py` + `biased_mppi_planner.py` | `_compute_game_block_cost`（111-151）+ `_trajectory_game_cost_torch`（420-475） |
| 跟踪/控制代价（式(8)） | `biased_mppi_planner.py` + `biased_mppi_core.py` | `_cost_func_torch`（262-323）+ `_compute_batch_costs`（277-340） |
| 总评分（式(9)） | `biased_mppi_core.py` | `:322 + :324-338` |
| **Feasibility Selection**（式(10)(11)(12)+回退） | `biased_mppi_post_selection.py` | `_boundary_validity_numba` / `_collision_validity_one_opp_numba` / `_select_index_numba` |
| warm start（Algorithm 1 line 11） | `post_selection.py:306-333` | `_commit_selected_action_sequence` |
| 纯跟踪执行 | `controller/pure_pursuit.py` | `plan`（68-83） |
| 仿真环境（F1TENTH Gym） | `f1tenth_gym/`（子模块） | `f110_env.step` → `Simulator.step` |
| 赛道与 raceline（G、MapZoo） | `MapZoo/`（子模块）+ `planner.py:41-52` | CSV 加载 |
| 8 种基线（§IV-A） | `lattice/spliner/End2Race/flow_planner/mpc_planner/` + 注册表 | `PLANNER_SPECS`（`planner_registry.py:130-141`） |
| 两种消融（§IV-A） | `planner_generators.py:8-12` 两个布尔 | 见 §5 |
| 评测指标（§IV-C：Wins/CFW/FPR） | **未实现**——批量链路只落盘 trace，指标要自己从 `trace.npy` 算 | 缺口，见 §6 |

---

## 4. 论文 §IV-B 参数 ↔ 仓库配置逐条互证

论文 Implementation 小节的每个数字都能在仓库里找到，这可以作为"复现配置正确性"的核对清单：

| 论文 §IV-B | 仓库 | 一致？ |
|---|---|---|
| K=128 | `ocp_config.yaml: num_samples=128` | ✅ |
| 1.2 s 时域，H=12 | `horizon=1.2, N=12`（→Δt=0.1） | ✅ |
| σ_a=0.335, σ_δ=0.025 | `sigma: [0.33544, 0.02489]` | ✅ |
| Q=[60, 60, 47.75, 39.48] | `running: track=4.0, heading=47.75, speed=39.48` | ⚠️ **位置权重不一致：论文 60，代码 4.0**（heading/speed 完全一致） |
| R=[8.43, 20.0], S=[1.0, 19.26] | `u_accel=8.43, u_steer=20.0, du_accel=1.0, du_steer=19.26` | ✅ |
| w_game=60 | `game_cost_config.yaml: game_cost_weight=60` | ✅ |
| s_contest=8.0, s_role=1.0, µ_tail=0.3, g_safe=1.0 | `contest_s_gap / role_s_margin / tail_ratio / safe_dist` | ✅ |
| [w_contest,w_long,w_block,w_safety]=[1,2,10,50] | 同名四项 | ✅ |
| L_IBR=2 | `run_config.yaml: ibr_time=2` | ✅ |
| **m_bd = 0.515 m** | 代码三个常数之和：`half_width 0.155 + buffer 0.30 + margin 0.06 = 0.515` | ✅ **精确互证**——论文直接把"含几何余量的等效阈值"写成了 m_bd，复现时别把 0.06 当绝对阈值 |
| d_col=0.9 | `post_selection.py:170 COLLISION_DIST=0.9` | ✅ |
| 7 条 test 赛道 | `config/data_split_final.yaml: test`（warehouse_v1, berlin, Brands_Hatch, f, Oschersleben, MoscowRaceway, Nuerburgring） | ✅ |
| 对手速度 ×0.9 | `opp_speed_scale: 0.9` | ✅ |
| ego 摆最后、车间距 2 m | `interval_idx=20`（×≈0.1 m 路点距）+ 摆车公式 ego 取 `ego_idx`、对手加 `i·interval_idx` | ✅ |
| 纯跟踪统一跟踪器 | 所有方法共用 `pure_pursuit.py` | ✅ |
| RTX 4060 GPU | 运行环境（代码侧 `torch.cuda.is_available()` 自适应） | — |

唯一不一致的是 Q 的位置分量（60 vs 4.0）：heading/speed/R/S 全对上而唯独 track 差 15 倍，更像是**论文定稿后仓库又被调过参**（位置权重调小 = 进一步让位给方向与速度）。复现实验时建议按论文的 60 试一版。

---

## 5. 论文的两种消融在工程里怎么跑

论文 §IV-A 两个消融变体，工程里都靠 `planner_generators.py:8-12` 的两个模块级布尔（**不在任何 yaml**）：

| 消融 | 论文描述 | 工程开关 |
|---|---|---|
| **SGTP w/o GC**（去掉博弈代价，保留可行性选择） | `USE_OPP_PRED = False`（`:11`）→ `_other_trajs` 返回 None → 博弈四项全部失效，只剩椭圆斥力 | 还需把 `game_cost_weight` 的影响隔离开 |
| **GA-IBR-MPPI**（博弈代价 + 标准 MPPI 加权平均） | `MPPI_POST_SELECT = False`（`:8`）→ `plan(post_select=False)` 直接用 `core:511-517` 的 softmax 加权平均，不做可行性筛选 | 完整 SGTP = 两个都 True |
| 标准 MPPI 基线 | 两个都 False + `game_cost_weight=0` | — |

---

## 6. 论文声称 ↔ 工程现状的三个缺口

| 论文说法 | 工程现状 |
|---|---|
| "mean computation time of 0.095 s" | **规划耗时被主动丢弃**：`competitive_planner.py:422` `del infer_time`。要复现这个数必须自己接出来计时 |
| 胜率 95.24% / 完赛率 99.35% / CFW / FPR（§IV-C 有明确定义） | 批量链路（`run_batched_competitive.py` → `competitive_metrics.py`）**只落盘 trace 和 manifest，不算任何指标**。`∆s_j = s_{j,T} − s_{j,0}` 的定义在代码里没有现成实现，要自己从 `traces/<case>.npy` 算 |
| 最多 10 车 | 支持：`--num_agents 10`。但注意 `common.py` 里 1 / 2 / >2 车是三条不同控制路径，加车数时留意 |

---

## 7. 阅读这份工程的最短路径（按运行顺序）

1. `run_multi_method.py`（82 行，10 分钟）——看参数怎么进来
2. `competitive_planner.py:140-360`——看对象怎么装配、env 怎么建、车怎么摆
3. `planner_generators.py`（194 行，全文）——**Algorithm 1 就在这，先读它**
4. `biased_mppi_post_selection.py:406-443`——一次 `plan()` 的外壳
5. `biased_mppi_core.py:440-529`——采样与 rollout
6. `biased_mppi_planner.py:262-323, 384-418, 420-475`——代价、动力学、参考轨迹、博弈代价加权
7. `game_block_cost.py`（188 行，全文）——博弈四项
8. `pure_pursuit.py:68-114`——轨迹怎么变成方向盘
9. `competitive_planner.py:363-583`——主循环四阶段
10. 基线与评测链路（`planner_registry.py` / `run_batched_competitive.py`）按需再看

> 相邻文档：静态结构 `工程脉络-SGTP-Racer.md`；公式逐行定位 `论文公式与代码对应-任务二.md`；变量释义 `SGTP-变量总表-任务一.md`。
