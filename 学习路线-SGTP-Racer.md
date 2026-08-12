# SGTP-Racer 学习路线图

> 面向"完全没头绪"的起点。按本文顺序走，两周内可以从「能跑」到「能改」。
> 全部结论来自对仓库的实际通读（86 个 .py 文件，共 17,609 行）。

---

## 一、先建立正确的问题定位

不要一上来读代码。先用三句话想清楚：

1. **环境是什么**：F1TENTH Gym —— 一个 1/10 缩比无人车的 2D 仿真器。赛道是一张黑白 PNG 占位栅格图，车是运动学自行车模型，感知是 2D LiDAR 射线。
2. **任务是什么**：N 辆车（默认 3 辆）在同一条赛道上同时跑。自车（ego，索引 0）要**又快又不撞**，对手也在为自己最优地跑，于是产生"防守挡位、并排缠斗、伺机超车"这类行为。
3. **这个仓库的贡献是什么**：把"采样式规划（MPPI）"改造成"会博弈的采样式规划"，即 **SGTP**，由三块拼成：

   | 组件 | 作用 | 代码位置 |
   |---|---|---|
   | GPU 批量采样 MPPI | 一次生成 128 条候选轨迹并并行评估 | `biased_mppi_core.py` |
   | Iterative Best Response（IBR） | 每辆车反复对他车最新预测做最优响应，逼近博弈均衡 | `planner_generators.py` |
   | 可行性后选择（Post-Selection） | **不**对候选加权平均，而是从"不撞墙且不撞车"的可行集里挑代价最小的那条 | `biased_mppi_post_selection.py` |
   | 对抗性博弈代价 | 纵向占优 + 进入缠斗 + 封堵对齐 + 安全惩罚 | `players/utils/game_block_cost.py` |

   同时它还是一个 **benchmark**：把 Lattice / Spliner / End2Race / Flow-Matching / MPPI / EVO-MPCC 统一接进同一套多车竞争框架，配 32 张赛道（25 train + 7 test）做横向评测。

**记住这句话**：这个仓库 90% 的代码不是"算法"，而是"把 7 种算法装进同一个多车竞技场并评测"。所以你要学的是**两层**：竞技场框架（工程）+ 一两个算法（理论）。

---

## 二、五层架构

```
① 入口层    run_multi_method.py（单次，出视频）      run_batched_competitive.py（批量，存轨迹）
                ↓
② 编排层    competitive_planner.py
            建规划器 → 建 env → 摆车 → while循环{ 规划阶段 → 跟踪阶段 → env.step → 记轨迹 }
                ↓
③ 博弈层    planner_generators.py
            第0轮自私预测 → (nonreactive | opp-reactive | ibr) → 最终自车响应
                ↓
④ 规划器层  SGTP(MPPI+后选择) / Biased-MPPI / Lattice / Spliner / EVO-MPCC / End2Race / Flow-CFM
            + pure_pursuit.py（把轨迹转成 [转角, 目标速度]）
                ↓
⑤ 仿真层    f1tenth_gym：ST 车辆动力学 / LiDAR 射线投射 / 碰撞判定 / 渲染
```

**数据从哪来**：`MapZoo/<地图名>/<地图名>_racing_line.csv`，分号分隔、跳过 1 行表头。
代码实际取用的列（0 基索引）：`1=s(累计弧长)`、`3=x`、`4=y`、`5=右边界宽`、`6=左边界宽`、`7=航向角`、`9=参考速度`。
规划器内部统一存成 5 列 `[x, y, v_ref, heading, s]`。

---

## 三、三张必须建立的"心智模型"

不懂这三点，读代码会一直绕。

### 模型 1：时间尺度（最容易晕）

| 量 | 值 | 在哪 | 含义 |
|---|---|---|---|
| 仿真步长 `dt` | 0.01 s | `run_config.yaml` | env 每步推进 10 ms |
| MPPI 预测步长 | 0.1 s | `horizon 1.2 / N 12` | 预测内每步 100 ms |
| 跟踪步数 `tracker_steps` | 1 | `run_config.yaml` | 每次规划后只执行 1 个 env 步 |
| 抓帧间隔 | 每 10 步 1 帧 | `video_capture_every` | 视频 10 fps |

**推论**：系统以 **100 Hz 重规划**，每次预测未来 **1.2 s**。
单秒仿真里，3 辆车 + `ibr_time=2` 需要 **约 1000 次 MPPI 求解**（每次 3 车 × (初始 1 + IBR 2 + 终轮 1) 次调用）。
→ 这就是为什么 README 里 quickstart 只用 `--sim_duration=1`，也是为什么必须有 GPU。

### 模型 2：一条轨迹在各层的形态

| 层 | 形态 | 说明 |
|---|---|---|
| 规划器输出 | `traj [T, 3]` = (x, y, 航向 或 速度) | MPPI 内部是 `state_seq [T+1, 4]` = (x, y, psi, v) |
| 跟踪器输入 | `[N, 3]` = (x, y, v) | 纯跟踪按前视距离挑点 |
| env 动作 | `[num_agents, 2]` = (前轮转角, 目标速度) | 不是加速度！ST 模型直接吃速度 |
| 评测产物 | `trace [steps, agents, 6]` | t, x, y, theta, steer, speed |

### 模型 3：两个"硬编码总开关"

在 `players/planner/planner_generators.py` 第 8～12 行：

```python
MPPI_POST_SELECT = True   # True = 可行性后选择（SGTP）；False = 标准 MPPI 加权平均
USE_OPP_PRED     = True   # True = 用对手预测触发博弈代价；False = 只看当前位姿的椭圆斥力
```

**这两个开关不在任何 yaml 里，改 yaml 改不动它们。** 论文里 SGTP 与"标准 MPPI"的区别就落在这两行 + `game_cost_config.yaml` 的 `game_cost_weight`。这是全仓库最容易踩的坑。

---

## 四、阅读顺序：三遍法

### 第一遍（约 2 小时）：只读 5 个文件，画出骨架

| 顺序 | 文件 | 读什么 | 为什么 |
|---|---|---|---|
| 1 | `README.md` L99-L145 | Overview + 各方法配置表 | 知道有哪些方法、怎么切换 |
| 2 | `players/run_multi_method.py`（82 行） | 全部 | 入口极短，看清参数从哪来 |
| 3 | `players/planner/competitive_planner.py` L140-L260 | 建规划器、建 env、摆车 | 看清"谁被创建了" |
| 4 | 同上 L363-L583 | 主循环 | 看清"每步发生了什么" |
| 5 | `players/utils/planner_registry.py`（141 行） | `PLANNER_SPECS` 表 | 一张表看懂 6 种规划器怎么注册 |

读完你应该能回答：`args.planner_family` 是怎么变成一个能 `plan()` 的对象的？

### 第二遍（约 1 天）：顺着一条路径读透

**只读 SGTP 这一条链**，别扩散：

```
planner_generators.py:78-194   (ocp_based_multi_generator，IBR 主循环)
  → biased_mppi_planner.py:477-522  (plan，一次求解入口)
    → biased_mppi_planner.py:405-418 (build_reference，参考轨迹怎么造)
    → biased_mppi_planner.py:262-323 (逐点代价：跟踪+控制+障碍)
    → biased_mppi_planner.py:384-403 (四状态运动学)
    → biased_mppi_planner.py:420-475 (博弈代价，仅当有对手预测时生效)
      → game_block_cost.py:70-151   (四项博弈代价公式)
    → biased_mppi_core.py:440-529   (采样 → 并行推演 → softmax 加权 → 最优控制序列)
  → biased_mppi_post_selection.py:10-166 (边界可行性 + 碰撞可行性，numba)
  → biased_mppi_post_selection.py:359-443 (覆盖加权平均结果)
  → pure_pursuit.py:88-114        (轨迹 → [转角, 速度])
```

### 第三遍（按需）：公式 ↔ 代码对照

打开 arXiv 论文，逐节找对应实现。建议对照顺序：
问题定义 → MPPI 采样与代价 → 博弈代价设计 → IBR 收敛性讨论 → 可行性选择 → 实验设置（对应 `eval_config.yaml`）。

---

## 五、五个阶段任务

> 每阶段都有「验收标准」。**做不到验收标准就别进下一阶段**，否则会积累假懂。

### 阶段 0：把轮子转起来（0.5～1 天）

**目标**：亲眼看到一个 mp4。

**任务**

- [ ] 初始化地图子模块（当前 `MapZoo/` 是空目录，必须先做）
  ```bash
  git config submodule.MapZoo.url https://github.com/zhouhengli/MapZoo.git
  git submodule update --init --recursive
  ls MapZoo/Nuerburgring   # 应能看到 .png .yaml _racing_line.csv
  ```
- [ ] 建环境（README 指定 Python 3.9）
  ```bash
  conda create -n sgtp-racer python=3.9 -y
  conda activate sgtp-racer
  pip install -r requirements.txt
  pip install -e .
  pip install -e f1tenth_gym
  ```
- [ ] 冒烟测试
  ```bash
  python players/run_multi_method.py --sim_duration=1 --num_agents=3 --interval_idx=20 --ego_idx=1100
  # 成功：进度条 + results/mppi_ibr_*.mp4
  ```
- [ ] 换一次参数：`--num_agents=2`、`--map_name=berlin`，确认换地图/车数都能跑

**验收**：`results/` 下有 mp4，能用播放器看到 3 辆车在跑；能说出为什么用 `--sim_duration=1` 而不是 15。

**踩坑预警**
- 子模块默认是 SSH 地址，没配 key 会失败 → 用上面改成 HTTPS。
- `pyglet==1.5.0` + 无显示环境：`args.render` 打开会崩，但 `save_video` 走的是 matplotlib 离线渲染器（`offline_save_video.py`），**不需要显示器**，默认路径是安全的。
- 无 CUDA 也能跑，`biased_mppi_core.py:79-83` 会自动回退 CPU，但会慢一个数量级。

---

### 阶段 1：骨架与数据流（1～2 天）

**目标**：能脱稿讲清"一条数据从 CSV 到方向盘"。

**必读**

| 文件 | 范围 | 重点 |
|---|---|---|
| `run_multi_method.py` | 全部 | `merge_args_with_config`：yaml 怎么覆盖命令行 |
| `competitive_planner.py` | L193-L232 | ego 用 `planner_family`，对手用 `opp_planner_func`（一个容易被忽略的不对称） |
| `competitive_planner.py` | L363-L583 | 双阶段循环、进度/圈数维护、碰撞即终止 |
| `utils/common.py` | L435-L476 | N 车按 `ego_idx + i*interval_idx` 沿赛车线摆放 |
| `utils/common.py` | L478-L542 | 2 车 / >2 车两套控制聚合（>2 车走"最近邻切片"） |
| `utils/common.py` | L701-L765 | Frenet 进度投影 + 圈数展开（负 delta 判圈） |

**动手**

1. 在 `competitive_planner.py` 主循环开头打印一次：
   `obs` 的所有 key 与每个字段的 shape/类型。
2. 打印 `agent_planners[0].waypoints.shape` 与 `waypoints[:3]`，确认 5 列含义。
3. 打印第一步的 `params_dict`，看清 `agent_progresses` / `nearest_agent_indices` 是什么。
4. 画一张你自己的调用图（纸质即可），标注 4 个时间量（0.01 / 0.1 / 1.2 / 100Hz）。

**验收**
- [ ] 能解释 `_align_initial_progresses_to_ego` 为什么要做"环形有符号偏移"
- [ ] 能解释为什么 `num_agents=2` 和 `=3` 走不同控制分支
- [ ] 能说出 `interval_idx=20` 对应的实际车距大概多少米

---

### 阶段 2：吃透 MPPI 单智能体（2～3 天）

**目标**：理解"128 条候选轨迹 → 一个动作"的全过程。

**必读**

- `biased_mppi_core.py` L440-L529 `forward`：**这是全仓库最核心的 90 行**
- `biased_mppi_core.py` L410-L438 采样（warm-start + 高斯扰动 + 保留均值样本）
- `biased_mppi_core.py` L343-L370 `lambda` 自动调节（MPO / LBPS / ESSPS 三种）
- `biased_mppi_core.py` L372-L408 Savitzky-Golay 平滑
- `biased_mppi_planner.py` L262-L323 逐点代价；L384-L403 运动学；L405-L418 参考轨迹
- `biased_mppi_planner.py` L336-L382 求解器组装（超参全从 `ocp_config.yaml` 来）
- `pure_pursuit.py` L88-L114 + L96-L114 前视点插值

**关键参数（`players/config/ocp_config.yaml`）**

| 参数 | 值 | 直觉 |
|---|---|---|
| `num_samples` | 128 | 候选轨迹条数。越大越稳、越慢 |
| `N` / `horizon` | 12 / 1.2 s | 预测 1.2 秒，内部步长 0.1 s |
| `lambda_` | 2.99 | softmax 温度。小→只看最优，大→平均化 |
| `sigma` | [0.335, 0.0249] | 加速度/转角扰动幅度 |
| `cost_weights.running` | track 4 / heading 47.8 / speed 39.5 / u 8.4,20 / du 1,19.3 | 航向与速度权重远大于位置 |
| `opp_cost_weight` | 4.0 | 无对手预测时的椭圆斥力强度 |

**动手实验（每个都要看视频对比）**

1. `num_samples: 128 → 8`：观察抖动加剧、行为变差。**体会"采样数就是分辨率"**。
2. `lambda_: 2.99 → 0.1`：权重几乎全压在最优样本上，行为变激进/抖。
3. `cost_running_heading` 调成 1.0：车开始切弯、偏离赛车线。
4. `num_agents=1` 单跑一圈，确认没有对手时整条链依然自洽。

**验收**
- [ ] 能手算：给定 128 个 cost 和 λ，权重怎么来、最终控制序列怎么来
- [ ] 能解释 `keep_mean_sample` 与"warm start"的关系
- [ ] 能解释 `build_reference` 为什么用"弧长等速外推"而不是插值时间序列
- [ ] 能说出 `_dynamics_torch` 里 `beta` 是什么、为什么 `psi` 更新要除 `lr`

---

### 阶段 3：博弈层与 SGTP 的两个卖点（2～3 天）

这是**最值得投入**的阶段，也是读懂论文的关键。

**必读**

| 主题 | 文件 | 范围 |
|---|---|---|
| IBR 主循环 | `planner_generators.py` | L78-L194 |
| 三种交互模式 | 同上 | L139-L194 |
| 后选择（可行性） | `biased_mppi_post_selection.py` | L10-L166（numba 实现） |
| 后选择（覆盖输出） | 同上 | L339-L443 |
| 博弈代价四项 | `utils/game_block_cost.py` | L70-L151 |
| 博弈代价接入 | `biased_mppi_planner.py` | L420-L475 |

**读法提示：IBR 循环其实只有 4 步**（`planner_generators.py`）

1. **第 0 轮**：所有车用 `opp_pred=None`、`bias_mode='none'` 各算一条"纯自私"轨迹 → `curr`
2. **按模式分支**：
   - `nonreactive`：所有人对固定 `curr` 再算一次 → 返回（这其实是"一次安全察觉"，不是真不反应）
   - `opp-reactive`：只有对手重算，自车直接用第 0 轮结果
   - `ibr`：`L = ibr_time` 轮**同步**最优响应，每轮用上一轮全体轨迹当预测
3. **终轮**：自车再对最终 `curr` 响应一次（保证自车最后拿到的是最新信息）
4. 返回 `best_trajs`（每车一条），供跟踪器执行

**博弈代价的四个分项**（`game_block_cost.py`，全部是"奖励"即负代价）

| 项 | 公式直觉 | 权重 |
|---|---|---|
| 纵向占优 | 未来弧长差 × 接近度因子，越大越好 | 2.0 |
| 缠斗状态 | 若未来仍处于 `contest_s_gap=8 m` 内，加分 | 1.0 |
| 封堵对齐 | 当自车在前方 1~8 m 时，横向差越小越好（`1/(1+Δd)`） | 10.0 |
| 安全惩罚 | 最短对齐距离 < 1 m 时二次惩罚 | 50.0 |

再乘总权重 `game_cost_weight = 60.0` 与量纲巨大的跟踪代价对齐。

**后选择做了什么**（`biased_mppi_post_selection.py`）

```
softmax 加权平均（标准 MPPI）  →  可能得到一条"谁都像、但谁都不像"的轨迹
        ↓ 替换为
在 128 条候选里筛掉：① 出赛道边界 ② 距离对手预测轨迹 < 0.9 m
        ↓
可行集内取 cost 最小者；若全不可行，取"违规量最小"者
        ↓ 把选中的动作序列写回 warm start
```

三个魔法数：`COLLISION_DIST=0.9`、`BOUNDARY_MARGIN=0.06`、`BOUNDARY_SAFETY_BUFFER=0.30`。
**作者原话（代码注释）**：这一步是为了"在多样的竞速行为之间实现平滑安全的切换"——即解决"加权平均会把防守和超车两种意图糊在一起"的问题。

**动手：做一张 4 组对照实验表**

| 组 | `MPPI_POST_SELECT` | `USE_OPP_PRED` | `game_cost_weight` | 预期现象 |
|---|---|---|---|---|
| A 标准 MPPI | False | False | 0.0 | 只会绕开"当前位姿"，不会预判 |
| B +博弈代价 | False | True | 60.0 | 会预判、会走位，但仍加权平均 |
| C +后选择 | True | False | 0.0 | 更贴边、更平滑，但不主动对抗 |
| D SGTP | True | True | 60.0 | 出现封堵/并排/超车 |

每组用同一张图、同一 `ego_idx`、同一 `rand_seed`，跑 `--sim_duration=15`，记录：
碰撞有无 / 是否完成超车 / 单次规划耗时（在 `planner_generators.py` 的 `infer_time` 处加打印，目前代码用 `del infer_time` 把它丢弃了）。

**验收**
- [ ] 能解释"为什么竞速场景下加权平均不够"（用 A vs D 的视频说明）
- [ ] 能解释 IBR 与 `nonreactive` 在计算量和行为上的差别
- [ ] 能说清后选择"全不可行时的降级策略"
- [ ] 能动 `game_cost_weight` 并预测行为变化方向（调大到 200 会怎样？）

---

### 阶段 4：基线与评测体系（2～3 天）

**目标**：能自己跑出一张横向对比表。

**每个基线只需读"入口 + 代价"两处**

| 方法 | 入口 | 一句话原理 |
|---|---|---|
| Lattice | `lattice_planner/lattice_planner.py:147-260` | 在"前视距离 × 横向偏移 × 速度"网格上撒 clothoid，5 项代价取最小 |
| Spliner | `spliner/spliner_planner.py:264-295` | Frenet 下造超车样条，按障碍左右空间选超车侧 |
| EVO-MPCC | `mpc_planner/evo_mpcc.py:470-...` | CasADi + IPOPT 解 MPCC，contouring/lag 误差 + 硬边界/硬碰撞约束 |
| End2Race | `End2Race/end2race_policy.py:72-94` | 360 维 LiDAR + 速度 → CNN → (转角, 期望速度)，端到端 |
| Flow-CFM | `flow_planner/flow_trajectory_generator.py` | 条件流匹配生成轨迹，用 `checkpoint/latest.pth` |

**评测体系三件套**

```
run_batched_competitive.py   → results/eval/<method>_raw_eval_<时间戳>/
                                  ├─ metrics/raw_episode_manifest.csv   （一 episode 一行）
                                  └─ <case>/ art/  trace.npy, planned_trajs.npz, refline.npy, boundary.npy
batch_render_episode_videos.py → results/all_episode_videos/  （从产物回放成视频）
players/metrics/competitive_metrics.py → 由 manifest 生成 renderer 可读的 episode 表
```

注意：**批量评测默认不算指标、不存视频**，只存轨迹张量与清单（`metrics_computed: false`）。这是为大规模跑设计的，指标需要自己从 `trace.npy` 算——这本身就是一个很好的练手任务。

**动手**

- [ ] 只改 `eval_config.yaml` 的 `split_key: test` 与小参数，跑一个小批量：
  ```bash
  python players/run_batched_competitive.py --sim-duration=15 --num-start-points-per-map=2
  ```
- [ ] 打开 `raw_episode_manifest.csv`，用 pandas 读懂每一列
- [ ] 用 `trace.npy` 自己算三个指标：完赛率、碰撞率、平均速度 → 写个 30 行脚本
- [ ] 从某个 case 的产物渲染一段视频，确认回放链路通

**验收**
- [ ] 能解释 `config/data_split_final.yaml` 里 train/test 划分的意义（学习型方法不能见 test 图）
- [ ] 能说清 `EGO_START` 采样逻辑（`uniform_indices` 沿赛道均匀取起点）
- [ ] 能出一个 SGTP vs 3 个基线的小表

---

### 阶段 5：改造与科研级（1～2 周）

**目标**：从"读得懂"到"改得动、说得清"。

**建议挑一个做（由易到难）**

1. **加诊断**：把被 `del infer_time` 丢掉的推理时间接进 metrics，输出"单步规划耗时分布"。
   涉及：`planner_generators.py` 返回 → `competitive_planner.py` 收集 → `run_batched` 落盘。**改动最小、收益最大。**
2. **加一个规划器**：写 `my_planner.py` + `my_generator.py`，在 `planner_registry.py` 的 `PLANNER_SPECS` 注册一行。
   建议先复制 `mpc_generator.py`（最短的 generator，88 行）当模板。
3. **做参数敏感性实验**：扫 `game_cost_weight ∈ {0, 20, 60, 120, 200}` × `ibr_time ∈ {0, 1, 2, 4}`，画"激进程度—碰撞率"权衡曲线。
4. **复现论文图表**：对照论文的实验节，用 `eval_config.yaml` + 批量评测重现。

**验收**
- [ ] 你的改动能用一条命令复现
- [ ] 能写出一页"这个方法的假设在哪、失效条件是什么"
- [ ] 能指出代码与论文不一致或未实现的地方（这种地方通常存在，找到就是贡献）

---

## 六、关键旋钮速查表

| 我想改… | 改哪 | 注意 |
|---|---|---|
| 用哪个规划器 | `run_config.yaml` 的 `planner_family` + `interaction_mode` | 两者要配对，注释里有组合 |
| SGTP ↔ 标准 MPPI | `planner_generators.py` 两个布尔 | 不在 yaml 里！ |
| 对抗强度 | `game_cost_config.yaml: game_cost_weight` | 从 60 起调，>200 容易自撞 |
| 博弈迭代轮数 | `run_config.yaml: ibr_time` | 每 +1 轮，计算量 +N 车次 |
| 对手实力 | `opp_speed_scale`（默认 0.9）、`v_global_limit`（0.7） | 对手速度上限 = 两者相乘 |
| 车距 | `--interval_idx` | 20 个路点，视地图而定 |
| MPPI 精度/速度 | `ocp_config.yaml: num_samples / N / lambda_` | 一次改一个 |
| 跟踪器性格 | `pp_config.yaml: minL/maxL/Pscale/D` | 影响横向跟踪激进程度 |
| 要不要存视频 | `--save_video` / `eval_config: save_video` | 批量评测默认关 |

---

## 七、六个调试手法（比读代码更有用）

1. **断点打在 generator 返回值上**，看 `best_trajs[i].shape` 和首尾点，比读 10 遍代码快。
2. **把 `num_samples` 压到 8**，用肉眼盯 128 条候选太累，8 条足够看清机制。
3. **`_last_costs` 是金矿**：`mppi._last_costs` 保存了这一轮全部候选的代价，直接打印分布（min/median/max）就能判断代价项是否失衡。
4. **单步冻结**：把 `sim_duration` 设为 0.05（5 步），逐帧打印 obs 与 action，最容易发现坐标系/单位错误。
5. **用 `trace_dashboard.py`** 从已有产物回放，不必重跑仿真。
6. **关掉对手**：`num_agents=1` 是隔离"规划器自身问题"与"交互问题"的最快开关。

---

## 八、当前工作区已发现的两个阻塞项

在你开始阶段 0 之前必须先解决：

1. **`MapZoo/` 是空目录**（子模块未检出）。没有地图 PNG/YAML/赛车线 CSV，任何脚本都会在 `get_map_paths` 后失败。见阶段 0 的命令。
2. **Python 依赖一个都没装**（torch / numpy / gym / numba / casadi / scipy 全部 import 失败）。需要按 README 建 conda 环境。注意 README 指定 Python 3.9，而 `gym` 与 `pyglet==1.5.0` 对高版本 Python 兼容性差，**不要用 3.11+ 硬上**。

---

## 九、14 天日程建议（每天 2～3 小时）

| 天 | 内容 | 产出 |
|---|---|---|
| D1 | 阶段 0 全部 | 一个 mp4 + 环境跑通笔记 |
| D2-D3 | 阶段 1：骨架 + 数据流 | 一张手绘调用图 |
| D4-D6 | 阶段 2：MPPI core + 4 个参数实验 | 4 段对比视频 + 一页公式笔记 |
| D7-D8 | 阶段 3 上半：IBR 三种模式 | 能讲清 4 步循环 |
| D9-D10 | 阶段 3 下半：后选择 + 博弈代价 + 4 组对照 | 对照表 + SGTP 与 MPPI 行为差异解释 |
| D11 | 复盘：不看代码，白纸重画整条链 | 一页纸架构图 |
| D12-D13 | 阶段 4：基线 + 小批量评测 | 对比表 + 指标脚本 |
| D14 | 阶段 5 选题 | 一个明确的改造计划 |

---

## 十、配套阅读

- 论文：*SGTP: Sampling-based Game-Theoretic Planning for Real-Time Multi-Vehicle Autonomous Racing*（arXiv 2607.25388）——**先读第 3、4 节**，带着"这三块在代码里对应哪三个文件"去读。
- MPPI 原始工作：Williams et al., *Model Predictive Path Integral Control*（2017）。只看采样-加权那一节。
- Biased-MPPI：TUD-AMR 的 `biased-mppi` 仓库，对照 `biased_mppi_ancillary.py`。
- 前置概念清单（缺哪个补哪个）：
  - 运动学自行车模型（含质心侧偏角 β）
  - Frenet / 弧长坐标，为什么赛道规划都在 (s, d) 下做
  - 纯跟踪控制 + 前视距离随速度变化
  - MPPI：重要性采样 + 软最优
  - 博弈论：最优响应、迭代最优响应、Nash 均衡的直觉（不必推证明）
  - CasADi / 直接法 OCP（只为看懂 EVO-MPCC，可以最后学）

---

## 附：文件体量速查（先读小的，别一头扎进 1238 行）

| 文件 | 行数 | 建议 |
|---|---|---|
| `run_multi_method.py` | 82 | 第一遍必读全文 |
| `planner_generators.py` | 194 | 第二遍核心 |
| `planner_registry.py` | 141 | 第一遍必读 |
| `mpc_generator.py` | 88 | 写新 generator 的模板 |
| `pure_pursuit.py` | 114 | 第二遍读 plan 部分 |
| `game_block_cost.py` | 188 | 第三遍精读 |
| `biased_mppi_post_selection.py` | 445 | 第三遍精读（含 numba） |
| `biased_mppi_planner.py` | 524 | 第二遍核心 |
| `biased_mppi_core.py` | 589 | 只精读 forward 及前后 90 行 |
| `competitive_planner.py` | 639 | 第一遍读主循环即可 |
| `common.py` | 765 | 按需查，别通读 |
| `offline_save_video.py` | 1238 | **完全不用读**，当黑盒 |
| `flow_planner/`（整棵子树） | ~3000 | 最后再看，与主线解耦 |
