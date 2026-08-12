# SGTP-Racer 工程脉络

> 阅读范围：86 个 `.py`（17,609 行）、7 个 yaml、32 张赛道（`MapZoo/` 子模块）。
> 一句话结论：**这是一座「多车竞技场 + 7 种规划器即插即用 + 离线评测流水线」的工程；SGTP 只是插在里面的一个算法。**

---

## 0. 一页速览

| 维度 | 内容 |
|---|---|
| 仿真器 | F1TENTH Gym（`f1tenth_gym/`，git 子模块，本地 pip -e 安装） |
| 任务 | N 辆车（默认 3）同场竞速，ego 索引 0，其余为对手 |
| 算法主体 | SGTP = GPU 批量采样 MPPI + Iterative Best Response + 可行性后选择 + 博弈代价 |
| 基线 | Lattice / Spliner / End2Race / Flow-CFM / 标准 MPPI / Biased-MPPI / EVO-MPCC(+IBR) |
| 地图 | `MapZoo/` 32 张，`config/data_split_final.yaml` 划 25 train / 2 test |
| 两个入口 | `players/run_multi_method.py`（单次出视频）、`players/run_batched_competitive.py`（批量存轨迹） |
| 唯一共用内核 | `players/planner/competitive_planner.py` |
| 控制频率 | 环境 100 Hz（dt=0.01），每次规划预测未来 1.2 s（12 步 × 0.1 s） |

---

## 1. 仓库的三重身份

读这个仓库最容易犯的错，是把它当成"一篇论文的代码"。它其实是三件事叠在一起：

1. **一个竞技场**（占代码量约 60%）：把任意规划器塞进同一套多车闭环里跑，统一摆车、统一推进、统一记录。
2. **一个算法**（约 20%）：SGTP 的四个部件，全部集中在 `biased_mppi/` + `planner_generators.py` + `game_block_cost.py`。
3. **一套评测**（约 20%）：批量采样起点、落盘 raw artifact、离线回放成视频。

因此"读懂工程"的正确目标不是逐行看算法，而是先看清**竞技场的接口契约**——只要知道规划器必须提供什么样的 `plan()`、必须返回什么形状的轨迹，7 种方法就变成了同一张图上的 7 个格子。

---

## 2. 目录 = 职责

```
SGTP-Racer/
├── players/                      ← 全部业务代码
│   ├── run_multi_method.py       入口 A：单次实验（控制台参数 → 一次 run）
│   ├── run_batched_competitive.py入口 B：批量评测（配置 → 多 case → raw 产物）
│   ├── batch_render_episode_videos.py  入口 C：从 raw 产物回放成视频
│   ├── config/                   7 个 yaml（方法选择、MPPI 超参、博弈代价、跟踪器…）
│   ├── planner/
│   │   ├── competitive_planner.py        ★ 编排层：建规划器 / 建 env / 主循环
│   │   ├── planner_generators.py         ★ 博弈层：IBR 三种模式的唯一实现
│   │   ├── biased_mppi/                  ★ SGTP 本体（core / planner / post_selection / ancillary）
│   │   ├── lattice_planner/  spliner/  mpc_planner/  End2Race/  flow_planner/   基线
│   │   └── controller/pure_pursuit.py    轨迹 → (转角, 目标速度)
│   ├── utils/                    common(数据/进度/控制聚合) · planner_registry(注册表) ·
│   │                             game_block_cost(博弈代价) · offline_save_video(渲染器) ·
│   │                             gym_helper · trace_dashboard(回放)
│   └── metrics/competitive_metrics.py   raw manifest → 可回放 episode 表
├── f1tenth_gym/                  ← 仿真器（子模块）：ST 动力学 / LiDAR / 碰撞 / 渲染
├── MapZoo/                       ← 赛道数据（子模块）：png + yaml + racing_line.csv
├── checkpoint/                   ← Flow-CFM 权重（latest.pth 98 MB + config.yaml）
├── config/data_split_final.yaml  ← train / test 地图划分
├── results/                      ← 视频与评测输出
└── 学习路线-SGTP-Racer.md         ← 另一份"怎么学"的文档（阶段任务 + 实验清单）
```

---

## 3. 主链路：从命令行到方向盘

以默认 SGTP 配置（`planner_family: mppi` + `interaction_mode: ibr`）走一遍：

| # | 发生什么 | 位置 |
|---|---|---|
| 1 | 解析 CLI，只保留 `map_name / ego_idx / num_agents / sim_duration / interval_idx / bias_mode` | `run_multi_method.py:11-24` |
| 2 | 读 `run_config.yaml`，**yaml 覆盖 CLI 同名项**；`method = f"{family}_{mode}"` | `:32-70` |
| 3 | 查注册表拿到 `[setup_fn, generator_fn]` | `competitive_planner.py:159` |
| 4 | 决定"对手用哪套规划器"（按 method 子串，见 §6） | `:164-171` |
| 5 | 逐个 agent 建规划器：ego 用 `planner_family`，对手用 `opp_planner_func`；ego 速度比例 0.7、对手 0.9×0.7 | `:193-227` |
| 6 | 规划器构造时读赛车线 CSV → `waypoints[N,5]`、`d_right_left[2,N]`，并建 `PurePursuitPlanner` + `BiasedMPPI` | `biased_mppi_planner.py:41-122` |
| 7 | `gym.make("f110-v0", map, timestep=0.01, num_agents)` | `:236-242` |
| 8 | 沿赛车线摆车：`idx = (ego_idx + i*interval_idx) % N` | `common.py:435-476` |
| 9 | `env.reset(poses)`，算初始进度并对齐到 ego（环形有符号偏移） | `:277` / `:65-83` |
| 10 | **主循环**：规划一次 → 跟踪执行 `tracker_steps` 次 → `env.step` → 记录 | `:363-583` |
| 11 | 规划 = 组 `params_dict` → 调 generator → 得 `best_trajs[N]` | `:376-420` |
| 12 | 跟踪 = 每条轨迹经纯跟踪转成 `action[N,2]=(steer, target_speed)` | `:438-470` |
| 13 | 每 10 步抓一帧（10 fps），结束时 `OfflineRaceVideoRenderer.save()` 出 mp4 | `:485-509` / `:626-629` |

**一个必须记住的事实**：`tracker_steps = 1`，所以"规划一次"就等于"推进 0.01 s"，即 **100 Hz 重规划**。这也是 README 的 quickstart 只跑 1 秒的原因（见 §11 算力账）。

---

## 4. 五层分解

```
① 入口层   run_multi_method.py        run_batched_competitive.py
                  └──────────┬──────────────┘
② 编排层        competitive_planner.py（建对象 → 主循环 → 记录/出图）
③ 博弈层        planner_generators.py（selfish → nonreactive/opp-reactive/ibr → final ego）
④ 规划器层      biased_mppi(SGTP) / lattice / spliner / mpc / end2race / flow  + pure_pursuit
⑤ 仿真层        f1tenth_gym：ST 动力学 + LiDAR + 碰撞 + 地图
```

### 4.1 入口层
- `run_multi_method.py`（82 行）：极薄。**只做参数合并**，真正逻辑全在编排层。默认 `save_video=True`，跑完必出一个 mp4 到 `results/`。
- `run_batched_competitive.py`（471 行）：把 `eval_config.yaml` 展开成 case 列表（地图 × 均匀起点），用进程池并发，每个 case 调同一个 `run_competitive_players(return_metrics=True)`，只落盘张量。

### 4.2 编排层（`competitive_planner.py`，639 行）
四段结构，边界很清：
1. **装配**（L155-231）：注册表查规划器 → 逐 agent 构造 → 对手类型分派。
2. **环境**（L233-301）：gym env、refline、摆车、离线渲染器。
3. **主循环**（L363-583）：规划 → 跟踪 → step → 记录进度/轨迹/视频帧 → 撞车即终止。
4. **收尾**（L585-634）：打印全部 args、打包 `metrics`、存视频或 End2Race 数据集。

值得注意的两个函数：
- `_align_initial_progresses_to_ego`（L65-83）：把各车弧长投影到 ego 的"环形邻域"，避免起跑线附近出现 ±整圈跳变。这是所有"谁在前谁在后"判断的地基。
- `_update_multi_progresses`（`common.py:743-765`）：用 `prev_raw - raw` 的负跳变判断越线并累加圈数。**进度更新的唯一真源**。

### 4.3 博弈层（`planner_generators.py`，194 行）
整个"博弈"只有 194 行，逻辑是刻意写平的：

```
for 每个 agent:  用 opp_pred=None + bias='none' 算一条"纯自私"轨迹      → curr
if nonreactive:  每个 agent 对固定 curr 再算一次                        → return   （2N 次求解）
if opp-reactive: 只有对手重算，ego 直接用第 0 轮                        → return   （1N + (N-1) 次）
if ibr:          重复 L 轮：所有 agent 同时用 prev 全体轨迹当预测重算     → curr
                 最后 ego 再对最终 curr 响应一次（保证 ego 信息最新）    → return   （N×(1+L+1) 次）
```

配套的两个模块级开关（**不在任何 yaml 里**，L8-12）：

```python
MPPI_POST_SELECT = True   # True=可行性后选择(SGTP) / False=标准MPPI加权平均
USE_OPP_PRED     = True   # True=把对手预测送进博弈代价 / False=只看当前位姿的椭圆斥力
```

### 4.4 规划器层
统一契约只有一句：`plan(x, y, θ, opp_poses, v, ...) -> (traj, heading)`，`traj` 形状 `[T,3] = (x, y, v)`。
- **SGTP**：`BiasedMPPIPostSelectionPlanner`（默认类）→ 见 §7。
- **Biased-MPPI**：`AncillaryBiasedMPPIPlanner`，为部分采样槽位注入人工先验策略（跟踪/加速跟车/对齐/刹车/规避 5 种），`bias_mode='ancillary'` 时生效。
- **Lattice**：`pyclothoids` 在"前视距离 × 横向偏移 × 速度"网格上生成 30×11 条 clothoid，5 项代价选最优。
- **Spliner**：Frenet 下造超车样条，按左右空间决定从哪侧过。
- **EVO-MPCC**：CasADi + IPOPT 解 MPCC（contouring/lag + 硬边界/硬碰撞约束）。
- **End2Race**：360 维 LiDAR + 速度 → CNN → 直接出 `(steer, speed)`，`direct_action=True` 会**跳过纯跟踪**。
- **Flow-CFM**：条件流匹配生成轨迹，用 `checkpoint/latest.pth`；对手轨迹作为条件输入。

### 4.5 仿真层（`f1tenth_gym/`）
- `f110_env.step(action[N,2])` → `Simulator.step` → 每车 `update_pose(steer, vel)`。
- 动力学是 **ST 单轨模型 + PID 速度跟踪**（`base_classes.py:254-320`），不是运动学自行车；输入是"目标速度"而非加速度。
- 感知：`Simulator` 默认 1440 束、全向 FOV 6.28（`base_classes.py:459`），观测里 `scans` 是 `[N, 1440]`；End2Race 采集时会降采样到 360 束。
- 终止条件：ego 碰撞，或任一车跑满 2 圈（`_check_done`）。

---

## 5. 数据契约

### 5.1 赛车线 CSV → `waypoints[N,5]`
`MapZoo/<map>/<map>_racing_line.csv`，分号分隔、跳过 1 行表头，代码只取 5 列（0 基索引）：

| 源码列 | 含义 | 落到 waypoints 第几列 |
|---|---|---|
| `1` | 累计弧长 s | 4 |
| `3` / `4` | x / y | 0 / 1 |
| `5` / `6` | 右 / 左边界宽 | 另存 `d_right_left` |
| `7` | 航向角 | 3 |
| `9` | 参考速度 | 2（乘 `v_scale` 缩放） |

### 5.2 各层的数据形态

| 层 | 形态 | 备注 |
|---|---|---|
| 规划器内部 | `state_seq[T+1,4] = (x,y,ψ,v)`，`action_seq[T,2] = (a,δ)` | MPPI 里控制是"加速度 + 转角" |
| 规划器输出 | `traj[T,3] = (x,y,v)` | 对手预测也是这个形状 |
| 跟踪器输入 | 同上 | 纯跟踪按 `L = v·(maxL-minL)/Lscale + minL` 挑前视点 |
| env 动作 | `action[N,2] = (前轮转角, 目标速度)` | **不是加速度** |
| 落盘 trace | `[steps, agents, 6]` | t, x, y, θ, steer, speed |

### 5.3 一处模型失配（重要）
规划器内部推演用**运动学自行车模型**（`biased_mppi_planner.py:384-403`，`lf=0.162, lr=0.145`，含质心侧偏角 β）；仿真器用**ST 动力学 + PID**。二者不共享模型，所以规划出来的轨迹在执行时会有偏差，纯跟踪的作用正是把这个偏差吸收掉。任何"提高规划精度"的改动都要记得这一点。

---

## 6. 规划器注册：一张表 + 一处不对称

`players/utils/planner_registry.py:130-141` 是全仓库的"插座"：

| key | setup 函数 | generator 函数 |
|---|---|---|
| `mppi` | `setup_biased_mppi_planner` | `ocp_based_multi_generator` |
| `flow_planner` | `setup_runtime` | `flow_based_generator` |
| `lattice_planner` | `setup_lattice_planner` | `lattice_multi_generator` |
| `spliner` | `setup_spliner_planner` | `spliner_based_multi_generator` |
| `mpc` | `setup_mpc_planner` | `mpc_based_multi_generator` |
| `end2race` | `setup_end2race_planner` | `end2race_mppi_generator` |

**不对称设计**（`competitive_planner.py:164-171`、`:193-227`）：
- **ego 用 `args.planner_family`**，想评什么就换什么。
- **对手永远按 method 字符串子串决定**：含 `lattice`→lattice，含 `mpc`→mpc，含 `spliner`→spliner，其余→**MPPI**。
- 结果：跑 SGTP 时对手也是 MPPI；跑 End2Race / Flow 时对手同样是 MPPI；跑 EVO-MPCC 时对手是 EVO-MPCC。

这意味着**"对手实力"不是独立可调的变量**，它绑在你选的 family 上。想做"强 ego vs 弱对手"的对照实验，必须改这段分派逻辑。

---

## 7. SGTP 的四个部件

### 7.1 GPU 批量采样（`biased_mppi_core.py:410-529`）
```
mean = 上一拍解出的动作序列（warm start，第 0 条样本强制保留为均值）
noises = 高斯采样(128, T, 2)
samples = clip(mean + noises)            # 可为部分槽位注入先验动作（biased-mppi）
for t in 0..T-1:  state_seq[:, t+1] = dynamics(state_seq[:, t], u_t)     # 纯 GPU 张量推演
costs = 逐步代价 + 末步代价 + （可选）轨迹级博弈代价
λ 自适应（MPO / LBPS / ESSPS）→ w = softmax(-costs/λ)
最优动作序列 = Σ w_i · u_i                # 标准 MPPI：加权平均
Savitzky-Golay 平滑 → 写回 warm start
```
一次求解 = 128 条 × 12 步的并行 rollout。

### 7.2 逐步代价（`biased_mppi_planner.py:262-323`）
`track·(ex²+ey²) + heading·eψ² + speed·ev² + u_accel·a² + u_steer·δ² + du·(Δa²+Δδ²)`
权重全在 `ocp_config.yaml`：航向 47.8、速度 39.5、位置 4.0 —— **方向与速度远比位置重要**，这是竞速调参的一个隐含偏好。
无对手预测时，额外加"椭圆斥力"（长短轴 2.0 / 1.6 m，权重 4.0）。

### 7.3 博弈代价（`utils/game_block_cost.py:70-151`）
一次调用评估一整批候选 × 一条对手预测，四项（都是**奖励**，即负代价）：

| 项 | 直觉 | 权重 |
|---|---|---|
| 纵向占优 | 未来弧长差 × 接近度因子（越近权重越大） | 2.0 |
| 缠斗状态 | 未来仍在 `contest_s_gap=8 m` 内则加分 | 1.0 |
| 封堵对齐 | 自车在前 1~8 m 时，横向偏差越小奖励越高 `1/(1+Δd)` | 10.0 |
| 安全惩罚 | 最短对齐距离 < 1 m 时二次罚 | 50.0 |

整体再乘 `game_cost_weight = 60.0`，用来和量纲巨大的跟踪代价对齐。**这个 60 是"对抗强度"的总旋钮**。

### 7.4 可行性后选择（`biased_mppi_post_selection.py`）
```
先照常算一遍标准 MPPI（加权平均）——结果只用来填 warm start 之外的内部状态
再对同一批 128 条候选做两次硬筛：
   ① 边界可行性：全轨迹点离左右边界留足半车宽 + 0.30 buffer，允许 0.06 容差
   ② 碰撞可行性：与每条对手预测轨迹的逐点最小距离 ≥ 0.9 m
可行集内取 cost 最小者；若全不可行，取"违规量最小"者
把选中的 (状态序列, 动作序列) 写回 warm start
```
三个魔法数：`COLLISION_DIST=0.9`、`BOUNDARY_MARGIN=0.06`、`BOUNDARY_SAFETY_BUFFER=0.30`。

**这一步解决的是"意图糊化"问题**：加权平均会把"防守"和"超车"两类样本混成一条既不像防守也不像超车的轨迹；在候选里"选一条"而不是"平均一批"，才能在多样的竞速行为之间平滑安全地切换。这是 SGTP 与标准 MPPI 最本质的差异，其余差异（博弈代价、IBR）都是为它服务的。

### 7.5 与三个基线的一行对照

| 方法 | `MPPI_POST_SELECT` | `USE_OPP_PRED` | `game_cost_weight` | `interaction_mode` |
|---|---|---|---|---|
| SGTP | True | True | 60 | ibr |
| 标准 MPPI | False | False | 0 | nonreactive |
| Biased-MPPI | False | False | 0 | nonreactive（+`--mppi_bias_mode_*=ancillary`） |
| EVO-MPCC(+IBR) | — | — | — | nonreactive / ibr |

---

## 8. 八个方法的接入方式

| 方法 | family | 支持的模式 | 实现要点 |
|---|---|---|---|
| SGTP | `mppi` | nonreactive / opp-reactive / **ibr** | 唯一实现完整 IBR 的采样式方法 |
| 标准 MPPI | `mppi` | 同上 | 靠两个布尔开关降级 |
| Biased-MPPI | `mppi` | 同上 | 采样槽位注入 5 种人工先验策略 |
| Lattice | `lattice_planner` | **仅 nonreactive**（否则抛错） | 30×11 clothoid 网格 + 5 项代价 |
| Spliner | `spliner` | **仅 nonreactive**（否则抛错） | Frenet 超车样条，左右空间选边 |
| End2Race | `end2race` | **仅 nonreactive** | ego 直出动作，对手仍走 MPPI（借用其地图几何） |
| Flow-CFM | `flow_planner` | **仅 nonreactive** | 强制 `ego_idx=0` 且至少 1 个对手；对手轨迹作为条件 |
| EVO-MPCC | `mpc` | nonreactive / **ibr** | CasADi/IPOPT，唯一另一个支持 IBR 的求解器 |

注意 `lattice / spliner / flow / end2race` 的 generator 都会**显式拒绝非 nonreactive 模式**；只有 `mppi` 和 `mpc` 两套 generator 实现了 IBR。

---

## 9. 配置体系与优先级

| 文件 | 管什么 | 谁读 |
|---|---|---|
| `run_config.yaml` | 方法选择、车体参数、dt、仿真/录制参数、`ibr_time`、对手速度比例 | 单次实验 |
| `eval_config.yaml` | `algorithm`（同 run_config 的方法段）+ `rollout` + 采样/并发/输出 | 批量评测 |
| `ocp_config.yaml` | MPPI 全部超参：`num_samples/N/horizon/lambda_/sigma/cost_weights` | MPPI、MPC |
| `game_cost_config.yaml` | 博弈代价四项权重 + `game_cost_weight` | MPPI |
| `pp_config.yaml` | 纯跟踪：`minL/maxL/Lscale/minP/maxP/D/interpScale` | 所有方法 |
| `lattice_config.yaml` | 网格范围、速度分层、代价开关 | lattice |
| `flow_config.yaml` | CFM 权重路径与条件配置 | flow |

**优先级**：`run_config.yaml` > CLI（同名项）。CLI 真正独占的只有 `map_name / ego_idx / num_agents / sim_duration / save_video / interval_idx / mppi_bias_mode_*`。
**不在任何 yaml 里的**：`MPPI_POST_SELECT`、`USE_OPP_PRED`（`planner_generators.py:8-12`）。

---

## 10. 评测链路与产物

```
run_batched_competitive.py
  ├─ build_cases: test 图 × uniform_indices(每个地图沿赛道均匀取 N 个起点)
  ├─ 进程池(parallel_backend=process, num_workers=3)  ← 不能用 thread
  └─ 每个 case: run_competitive_players(return_metrics=True)
        └─ results/eval/<method>_raw_eval_<ts>/
             ├─ configs/            eval/algorithm/rollout/split 四个快照（可复现）
             ├─ cases/resolved_cases.yaml
             ├─ traces/<case>.npy + <case>_meta.json        ← 完整 rollout
             ├─ artifacts/<map>/<case>/planned_trajs.npz    ← 仅 ego 的规划轨迹
             │                          boundary_offsets.npy  refline_xy.npy
             │                          render_artifact.json
             └─ metrics/raw_episode_manifest.csv            ← 一个 episode 一行
                    ↓
competitive_metrics.py      → post_metrics/episode_metrics.csv（仅重排成可回放表）
                    ↓
batch_render_episode_videos.py → results/all_episode_videos/<case>.mp4
```

**两个必须知道的缺口**：
1. 批量模式 **不算任何指标**（`build_summary: metrics_computed=False`）。`competitive_metrics.py` 只把 manifest 重排成渲染器能读的表，**完赛率、碰撞率、圈速都要自己从 `trace.npy` 算**。
2. 批量模式 **只保存 ego 的规划轨迹**（`competitive_planner.py:427-430` 只写入 `best_trajs[0]`），所以回放视频里能看到 ego 的候选/参考线，看不到对手的规划。

---

## 11. 时间尺度与算力账

| 量 | 值 | 出处 |
|---|---|---|
| 环境步长 `dt` | 0.01 s | `run_config.yaml` |
| 每次规划执行步数 | 1 | `tracker_steps` |
| → 重规划频率 | **100 Hz** | 推论 |
| MPPI 预测步长 | 1.2 / 12 = 0.1 s | `ocp_config.yaml` |
| MPPI 预测时域 | 1.2 s | 同上 |
| 视频帧率 | 10 fps（每 10 步抓 1 帧） | `video_capture_every` |

算力账（3 车、`ibr_time=2`）：每拍 3×(1 自私 + 2 IBR + 1 终轮) = **12 次 MPPI 求解**；每拍 0.01 s → **每秒仿真 1200 次求解**；每次 128 条 × 12 步 rollout → 约 **每秒 18 万条轨迹推演**。这就是"必须有 GPU"和"quickstart 只跑 1 秒"的全部原因。

---

## 12. 设计取舍与坑（读代码时最容易踩）

1. **两个总开关在源码里**。改 yaml 无法把 SGTP 变成标准 MPPI，必须改 `planner_generators.py:8-12` 的布尔值——这是全仓库最容易踩的坑。
2. **`--save_video` 是 argparse `type=bool`**，`--save_video False` 仍被解析为 `True`。要关掉请改 yaml 或代码。
3. **规划耗时被主动丢弃**：`competitive_planner.py:422` 有 `del infer_time`。想做实时性统计，必须自己在这里接出来。
4. **模型失配是设计的一部分**：规划用运动学、执行用 ST+PID，靠纯跟踪兜底。改规划器动力学要同步评估跟踪误差。
5. **对手类型绑在 method 字符串上**（§6），改 family 名字会静默换掉对手的规划器。
6. **`num_agents=2` 与 `>2` 是两条不同的代码路径**：前者 `compute_control_actions`，后者 `compute_multi_control_actions`（对每个车切"自己 + 最近邻"的二元组）；`=1` 又是第三条。加车数时留意。
7. **End2Race 的 ego 绕过跟踪器**：`direct_action=True` → `compute_direct_ego_mppi_opp_actions` 直接用网络输出的 `(steer, speed)`。
8. **路径依赖强**：所有配置路径都是相对仓库根目录的字符串，必须在根目录用 `python players/xxx.py` 运行；`batch_render_episode_videos.py` 里用的是 `from utils.trace_dashboard import ...`，依赖脚本自身目录在 `sys.path`。
9. **Flow 方法有硬约束**：`ego_idx` 必须是 0，且至少 1 个对手，否则直接抛错。
10. **`results/` 里的 mp4 命名编码了实验条件**：`<method>_<f|o>_ol<raceline>_e<ego>_o<opp>_s<scale>.mp4`（`gym_helper.py:9-22`），批量跑完靠文件名就能分辨。

---

## 13. 任务导向索引：想改 X，去动 Y

| 目标 | 动手位置 | 备注 |
|---|---|---|
| 换方法 | `run_config.yaml` 的 `planner_family` + `interaction_mode` | 两者要配对，组合见注释 |
| SGTP ↔ 标准 MPPI | `planner_generators.py:8-12` 两个布尔 | 不在 yaml |
| 调对抗强度 | `game_cost_config.yaml: game_cost_weight` | 从 60 起，>200 易自撞 |
| 调博弈轮数 | `run_config.yaml: ibr_time` | 每 +1 轮，计算量 +N 次求解 |
| 调采样分辨率 | `ocp_config.yaml: num_samples` | 一次只改一个 |
| 调预测时域 | `ocp_config.yaml: N / horizon` | 改后 `dt=horizon/N` 自动变 |
| 调跟踪性格 | `pp_config.yaml: minL/maxL/Pscale/D` | 影响横向激进程度 |
| 加新规划器 | 写 `setup_fn` + `generator_fn`，在 `PLANNER_SPECS` 加一行 | 建议抄 `mpc_generator.py`（88 行，最短） |
| 加诊断/指标 | `generator` 返回值 → `competitive_planner` 收集 → `run_batched` 落盘 | 最短闭环改动 |
| 换地图 | `--map_name`，并确认在 `data_split_final.yaml` 里 | 否则批量评测取不到 |
| 加车 | `--num_agents` + `interval_idx` | 注意 §12.6 的三条分支 |

---

## 14. 附：体量与阅读优先级

| 文件 | 行数 | 建议 |
|---|---|---|
| `run_multi_method.py` | 82 | 通读（10 分钟） |
| `planner_generators.py` | 194 | 通读，博弈全在这里 |
| `planner_registry.py` | 141 | 通读，插座表 |
| `mpc_generator.py` | 88 | 写新 generator 的模板 |
| `pure_pursuit.py` | 114 | 读 `plan` + 前视点插值 |
| `game_block_cost.py` | 188 | 精读，四项公式 |
| `biased_mppi_post_selection.py` | 445 | 精读（numba 三段 + `plan`） |
| `biased_mppi_planner.py` | 524 | 精读，代价与动力学 |
| `biased_mppi_core.py` | 589 | 只精读 `forward` 及采样/自适应 λ |
| `competitive_planner.py` | 639 | 读装配段 + 主循环 |
| `common.py` | 765 | 按需查（进度/摆车/控制聚合） |
| `offline_save_video.py` | 1238 | 当黑盒，不必读 |
| `flow_planner/` 整棵 | ~3000 | 与主线解耦，最后看 |

> 学习节奏与实验清单见同目录 `学习路线-SGTP-Racer.md`；本文只负责"结构是什么、东西在哪、边界在哪"。
