# 论文公式 ↔ 代码对照（SGTP-Racer）

> 对象：`SGTP: Sampling-based Game-Theoretic Planning for Real-Time Multi-Vehicle Autonomous Racing`（arXiv 2607.25388）第 3–4 节公式 (1)–(12)。
> 方法：逐式在仓库中定位实现，行号以当前工作区代码为准。
> 结论先行：**12 个公式全部有实现，无遗漏；但有 4 处实现与论文写法不等价，其中「阻挡项取绝对值」会改变代价的语义方向（见 §4.1），是最需要留意的一处。**

---

## 0. 阅读约定

| 约定 | 说明 |
|---|---|
| 时间索引 | `t` 当前时刻；`τ` 预测步（0…H）；`t+τ` 即"第 τ 步之后" |
| 横向符号 | 代码 `d = -dx·sinψ + dy·cosψ`（`game_block_cost.py:27`、`biased_mppi_post_selection.py:60`），**d > 0 表示参考线左侧** |
| 进度符号 | `Δs = s_i − s_j`，**Δs > 0 表示本车更靠前** |
| 代价符号 | 博弈四项都是**负代价（奖励）**，跟踪/控制项是正代价，二者相加 |
| 采样数 | 论文 `K` = 代码 `num_samples` = **128** |
| 时域 | 论文 `H` = 代码 `N` = **12**，`Δt` = `horizon/N` = **0.1 s** |

---

## 1. 变量总表

### 1.1 集合与索引

| 符号 | 数学定义 | 含义 | 代码对应 |
|---|---|---|---|
| `𝓘 = {1,…,n_v}` | 车辆集合 | 同场竞速的所有车 | `range(num_agents)`，`num_agents` 默认 3 |
| `n_v` | 车辆总数 | 车数 | `args.num_agents`（yaml `num_agents`） |
| `𝓘₋ᵢ` | `𝓘 \ {i}` | 车辆 i 的对手集合 | `collect_other_poses_from_obs(obs, i)` / `_other_trajs(trajs, i)` |
| `i` | — | 本车索引；ego 固定 0 | `ego_agent_idx = 0`（`planner_generators.py:85`） |
| `j` | — | 对手索引 | 同上对手集合的遍历变量 |
| `k` | `1…K` | 候选轨迹编号 | MPPI 采样批的第 k 行（`_state_seq_batch[k]`） |
| `K` | 候选数 | 采样分辨率 | `num_samples = 128`（`ocp_config.yaml`） |
| `ℓ` | IBR 轮次 | 第 ℓ 轮最佳响应 | `ibr_iter`，`range(num_ibr_iters)`，`ibr_time = 2` |
| `τ` | `0…H` | 预测步 | `for t in range(self._horizon)`（`biased_mppi_core.py:474`） |
| `t` | 当前时刻 | 本次规划的起始时刻 | 由 `obs` 给出，代码不显式存 `t` |
| `H` | 预测时域步数 | 预测 12 步 | `N = 12` → `self._horizon` |
| `T_tail` | 末段步集合 | 计算横向均值的尾段 | `tail_ratio = 0.3` → 索引 9…12，见 `game_block_cost.py:31-51` |
| `|T_tail|` | 尾段步数 | 均值分母 | `count`（同上） |
| `N_trk` | 参考线路点数 | 赛道离散化粒度 | `len(self.waypoints)`，逐地图不同 |
| `G` | `{(xⱼʳ, wⱼᴸ, wⱼᴿ)}` | 赛道几何：参考点 + 左右边界宽 | `self.waypoints[N,5]` + `self.d_right_left[2,N]` |

### 1.2 状态与控制

| 符号 | 数学定义 | 含义 | 代码对应 / 取值 |
|---|---|---|---|
| `x_{i,t}` | `[x, y, ψ, v]ᵀ ∈ ℝ⁴` | 状态向量 | `x0 = torch.tensor([pose_x, pose_y, pose_theta, velocity])`（`biased_mppi_planner.py:492-496`） |
| `x, y` | 平面坐标 | 后轴中心位置 | `pose_x`, `pose_y`；`state[:,0]`, `state[:,1]` |
| `ψ` | 航向角 | 车体朝向（rad） | `pose_theta`；`state[:,2]` |
| `v` | 纵向速度 | m/s | `obs["linear_vels_x"]`；`state[:,3]` |
| `u_{i,t}` | `[a, δ]ᵀ ∈ ℝ²` | 控制输入 | `action_seq[:, :, [0,1]]` |
| `a` | 纵向加速度 | m/s²（**规划器内部量**） | `action[:,0]` |
| `δ` | 前轮转角 | rad | `action[:,1]`；限幅 ±`delta_max`=0.4 |
| `f(·)` | 运动学自行车模型 | 状态转移 | `_dynamics_torch`（`biased_mppi_planner.py:384-403`） |
| `U_{i,t}` | `[u_t,…,u_{t+H−1}]` | 控制序列 `H×2` | `_perturbed_action_seqs` |
| `ξ_{i,t}` | `[x_t,…,x_{t+H}]` | 状态轨迹 `(H+1)×4` | `_state_seq_batch` |
| `p_{i,t}` | `(x, y)` | 位置的简写 | `state[:, :2]` |
| `ß`（代码独有） | `arctan(lr/(lf+lr)·tanδ)` | 质心侧偏角 | `beta`（`:396`），`lf=0.162, lr=0.145` |

### 1.3 采样与迭代

| 符号 | 数学定义 | 含义 | 代码对应 / 取值 |
|---|---|---|---|
| `U^ℓ_{i,t}` | 第 ℓ 轮名义序列 | 采样中心 = 上轮解 | `mean_action_seq = self._previous_action_seq`（`core:455`） |
| `ε^{(k,ℓ)}` | `∼𝒩(0, Σ_u)` | 控制扰动 | `self._action_noises`（`core:422`） |
| `Σ_u` | `diag(σ_a², σ_δ²)` | 扰动协方差 | `sigmas = [0.33544, 0.024888]` |
| `σ_a` | 加速度标准差 | 纵向探索幅度 | `sigma[0] = 0.33544` |
| `σ_δ` | 转角标准差 | 横向探索幅度 | `sigma[1] = 0.024888` |
| `ũ` | 扰动后未投影控制 | 中间量 | `mean_action_seq + noises`（`core:427`） |
| `𝓤` | 可行控制集合 | 箱式约束 | `[−a_max, a_max] × [−δ_max, δ_max]` |
| `Π_𝓤(·)` | 投影算子 | 限幅 | `torch.clamp(..., self._u_min, self._u_max)`（`core:432`） |
| `𝒞^ℓ_{i,t}` | 候选集合 | K 条 (轨迹, 序列) | `_state_seq_batch` / `_perturbed_action_seqs` |
| `k*` | 选中候选号 | 输出索引 | `selected_idx`（`post_selection.py:376`） |
| `λ` | 软最优温度 | 权重锐度（论文未在主式出现） | `lambda_ = 2.9896`，支持 MPO/LBPS/ESSPS 自适应 |

### 1.4 Frenet 与进度

| 符号 | 数学定义 | 含义 | 代码对应 |
|---|---|---|---|
| `s_{i,t}` | `s(x_{i,t}; G)` | 沿赛道弧长进度 | `_point_to_frenet_d_s` 的 `s_ref`（`game_block_cost.py:22`）；全局进度用 `_compute_raw_progresses` |
| `d_{i,t}` | `d(x_{i,t}; G)` | 相对参考线横向偏移（**左正**） | 同上第二个返回值；`:27` |
| `ŝ_{j,t+τ}` / `d̂` | 对手预测的 s / d | 带帽子=预测值 | 由 `opp_pred_traj` 逐点投影得到 |
| `Δs_{ij,t}` | `s_i − ŝ_j`（当前） | 当前纵向差（正=本车在前） | `current_s_gap`（`game_block_cost.py:125`） |
| `Δs^{(k)}_{ij,t+H}` | `s^{(k)}_{i,t+H} − ŝ_{j,t+H}` | 末端纵向差 | `future_s_gap`（`:126`） |
| `s_contest` | 竞争窗口 | 8 m 内算"还在缠斗" | `contest_s_gap = 8.0` |
| `s_role` | 角色切换阈值 | 领先超过 1 m 才有"阻挡者"身份 | `role_s_margin = 1.0` |
| `d̄^{(k)}_i` / `d̄_j` | 尾段横向均值 | 用平均值而非单点，抗抖动 | `ego_mean_d` / `opp_mean_d`（`:114` / `:98`） |
| `Δd^{(k)}_{ij,tail}` | `d̄_i − d̄_j`（**带符号**） | 横向差，正=本车在对手左侧 | 代码为 `np.abs(ego_mean_d − opp_mean_d)`（`:141`）⚠️ |
| `ε` | `1e−6` | 防除零 | `eps`（`:109`） |

### 1.5 博弈代价

| 符号 | 值 | 含义 | 代码对应 |
|---|---|---|---|
| `c_game,ij` | — | 对单个对手的成对代价 | `_compute_game_block_cost` 的 `cost[i,0]`（`game_block_cost.py:93`） |
| `c_game,i` | — | 对所有对手求和 | `total_game_cost += game_cost_i`（`planner:472`） |
| `w_contest` | `1.0` | 缠斗奖励权重 | `contest_weight` |
| `w_long` | `2.0` | 纵向优势权重 | `longitudinal_weight` |
| `w_block` | `10.0` | 阻挡对齐权重 | `block_weight` |
| `w_safety` | `50.0` | 安全惩罚权重 | `safety_weight` |
| `α_{ij,t}` | `(1+|Δs_t|/(s_contest+ε))⁻¹` | 距离越近，纵向项越重要（∈(0,1]） | `contest_alpha`（`:130`） |
| `I^{(k)}_{ij,H}` | `𝟙[|Δs_{t+H}| < s_contest]` | 末端仍在竞争窗口 | `contest_score`（`:135-136`） |
| `γ_{block,ij}` | `w_block·𝟙[Δs_t>s_role]·𝟙[Δs_t<s_contest]` | 阻挡项开关（仅"领先但未甩开"） | `if current_s_gap > role_s_margin and ... < contest_s_gap`（`:140`） |
| `g_min,ij` | `min_τ ‖p_i − p̂_j‖₂` | 时间对齐最小间距 | `dmin`（`:145`）← `_min_aligned_distance`（`:54-67`） |
| `g_safe` | `1.0 m` | 安全距离阈值 | `safe_dist = 1.0` |
| `w_game` | `60.0` | 博弈总权重（量纲对齐用） | `game_cost_weight`（`planner:77`、`:474`） |

### 1.6 跟踪代价

| 符号 | 含义 | 代码对应 / 取值 |
|---|---|---|
| `c_track,i` | 跟踪+控制+平滑总代价 | `_cost_func_torch`（`planner:262-323`） |
| `s^r_{i,t+τ}` | 参考进度（沿参考线外推） | `s_target = s_curr + v_ref_base·(k·dt)`（`:412`） |
| `v^r(s)` | 参考速度场 | `wp_v = waypoints[:,2]`（已乘 `v_scale`） |
| `x^r` | 参考状态 | `ref_traj[0..3, k]`（经 `np.interp` 在弧长上插值） |
| `Q` | 状态权重（对角） | `track=4.0, heading=47.75, speed=39.48` |
| `R` | 控制幅值权重（对角） | `u_accel=8.43, u_steer=20.0` |
| `S` | 控制变化率权重（对角） | `du_accel=1.0, du_steer=19.26` |
| `Δu` | `u_τ − u_{τ−1}` | `da`, `ddelta`（`:288-289`） |
| `c_i^{(k)}` | 总评分 | `total_costs`（`core:322` + `:338`） |

### 1.7 可行性

| 符号 | 含义 | 代码对应 / 取值 |
|---|---|---|
| `ρ_bd` | 最小边界裕度 | `min_boundary_clearance[k]`（`post_selection.py:79`） |
| `w_L(s)` / `w_R(s)` | 左/右边界宽 | `d_right_left[1, idx]` / `[0, idx]`（**离散取最近点**，`:62-63`） |
| `m_bd` | 边界裕度阈值 | `BOUNDARY_MARGIN = 0.06` |
| `half_width`（代码独有） | 半车宽 | `0.5 × args.width = 0.155`（`:205`） |
| `safety_buffer`（代码独有） | 边界安全垫 | `BOUNDARY_SAFETY_BUFFER = 0.30` |
| `ρ_obs` | 对所有对手的最小间距 | `min_obs_dist[k]`（`post_selection.py:114-115`） |
| `d_col` | 碰撞判定距离 | `COLLISION_DIST = 0.9` |
| `b^{(k)}` / `o^{(k)}` | 边界 / 碰撞通过标志 | `valid_boundary[k]` / `valid_collision[k]` |
| `h^{(k)}` | 综合可行性 | `valid_boundary[k] and valid_collision[k]`（`:136`） |
| `ν_bd`, `ν_obs`, `ν` | 边界 / 碰撞 / 总违规量 | `boundary_violation`, `collision_violation`, `safety_violation`（`:148-160`） |

---

## 2. 12 个公式 → 代码定位总表

| 式 | 论文含义 | 文件 | 函数 / 代码段 | 行号 |
|---|---|---|---|---|
| (1) | IBR 内选可行最优候选 | `players/planner/planner_generators.py` | `ocp_based_multi_generator`（IBR 主循环） | `118-194` |
| (1) | 同上（选择算子本身） | `players/planner/biased_mppi/biased_mppi_post_selection.py` | `_select_index_numba` / `_post_select_mppi_output` | `121-166` / `359-400` |
| (2) | 成对博弈代价求和 | `players/utils/game_block_cost.py` | `_compute_game_block_cost`；对手求和与加权在 `biased_mppi_planner.py` | `111-151`；`451-475` |
| (3) | 维持竞争窗口 | `players/utils/game_block_cost.py` | `_compute_game_block_cost` 第 2 项 | `133-137` |
| (4) | 纵向进度优势 | 同上 | 第 1 项 + `α` | `125-131` |
| (5) | 阻挡对齐 | 同上 | 第 3 项 | `139-142` |
| (6) | 时间对齐最小间距 | 同上；碰撞检查复用另一实现 | `_min_aligned_distance`；`_collision_validity_one_opp_numba` | `54-67`；`post:100-112` |
| (7) | 安全间距平方惩罚 | `players/utils/game_block_cost.py` | 第 4 项 | `144-148` |
| (8) | 跟踪 + 控制 + 平滑代价 | `players/planner/biased_mppi/biased_mppi_planner.py` | `_cost_func_torch`；求和与末步在 `biased_mppi_core.py::_compute_batch_costs` | `262-323`；`core:277-340` |
| (8) | 参考轨迹生成 | 同上 | `build_reference` | `405-418` |
| (9) | 总评分 | `players/planner/biased_mppi/biased_mppi_core.py` | `_compute_batch_costs`；`w_game` 在 planner | `322`、`324-338`；`planner:474` |
| (10) | 赛道边界检查 | `players/planner/biased_mppi/biased_mppi_post_selection.py` | `_boundary_validity_numba` + 包装 | `11-82`；`204-215` |
| (11) | 车辆碰撞检查 | 同上 | `_collision_validity_one_opp_numba` + 包装 | `86-118`；`252-276` |
| (12) | 最终可选最优选择 | 同上 | `_select_index_numba` 前半 | `135-142` |
| §6 | 最小违规回退 | 同上 | `_select_index_numba` 后半 | `144-166` |

**执行顺序**（按一次 `plan()` 的真实调用链）：

```
planner_generators.py:122-134   第0轮自私预测（opp_pred=None → 椭圆斥力，对应论文"基于距离的避障初始化"）
  └─ biased_mppi_planner.py:477  plan()
       ├─ :405  build_reference            → 式(8) 的参考 x^r
       ├─ core:455  mean = 上轮解          → 式(1) 的 U^ℓ
       ├─ core:422-438  采样+限幅          → Π_U、𝒞^ℓ
       ├─ core:474-499  逐步推演           → f(·)
       ├─ core:501  _compute_batch_costs
       │     ├─ :295-305  Σ 跟踪+控制+Δu   → 式(8)
       │     ├─ :316-322  末步 + 求和      → 式(9) 前半
       │     └─ :324-338  + 博弈代价       → 式(9) 后半（w_game·Σ_j 式(2)）
       │            └─ game_block_cost.py:70  → 式(3)(4)(5)(6)(7)
       └─ core:511-517  softmax 加权平均   ← SGTP 会丢弃这一步的结果
  └─ post_selection.py:359  _post_select_mppi_output
       ├─ :204-215  边界可行性             → 式(10)
       ├─ :252-276  碰撞可行性             → 式(11)
       └─ :339-357  可行集取最小/回退      → 式(12) + §6
planner_generators.py:171-182   L 轮同步最佳响应     → 式(1) 的 ℓ 递推
planner_generators.py:184-190   ego 终轮响应         → 论文"最终再响应一次"
```

---

## 3. 逐式详解

### 式 (1)｜IBR 内的可行最优选择

```python
# planner_generators.py:171-182
for ibr_iter in range(int(params_dict["num_ibr_iters"])):
    prev = curr
    curr = [plan_once(agent_idx=i,
                      opp_pred_traj=_other_trajs(prev, i),   # 上一轮全体轨迹当预测
                      post_select=MPPI_POST_SELECT)          # ← 打开式(1)的可行筛选
            for i in range(len(agent_planners))]
# :184-190 ego 再对最终 curr 响应一次
curr[ego_idx] = plan_once(ego_idx, _other_trajs(curr, ego_idx), MPPI_POST_SELECT)[0]
```

- **可行集筛选**落在 `_select_index_numba`（`post:135-142`）：`for k: if valid_boundary and valid_collision and isfinite(cost)` 中取 `cost` 最小。
- **warm start** 由 `_commit_selected_action_sequence`（`post:306-333`）写回 `mppi._previous_action_seq`，下一次 `forward` 的 `mean_action_seq` 就是它（`core:455`）——论文"所选控制序列用于 warm start"逐字对应。
- **初始化**：第 0 轮 `opp_pred_traj=None`、`bias_override='none'`（`planner_generators.py:122-129`），此时代价里只有基于当前位姿的椭圆斥力（`planner:296-323`），即论文所说的"基于距离的避碰轨迹"。
- ℓ 的递推是**同步（Jacobi）更新**：一轮内所有车都用上一轮全体轨迹，而不是顺序（Gauss-Seidel）。

### 式 (2)｜成对博弈代价与求和

```python
# game_block_cost.py:111-151  —— 一次算完 K 个候选对 1 个对手
for i in range(K):
    ego_traj = all_traj[i]
    ...            # 四项累加进 cost[i,0]

# biased_mppi_planner.py:451-475  —— 对所有对手求和，再乘 w_game
for one_opp_pred_traj in opp_pred_trajs:
    total_game_cost += self.game_block_cost(ego_xyv[:, :horizon_len, :], opp_xyv[:horizon_len, :], waypoints_np).reshape(-1)
total_game_cost *= self.game_cost_weight
return torch.tensor(total_game_cost, ...)
```

- 求和维度正确：**先对候选向量化、再对对手循环**。式 (2) 的 `Σ_j` 对应 `:451-472` 的 for 循环。
- 注意：多对手时是**逐对手独立求和**，没有做 `min` 或加权——所以 3 车时 `c_game` 的量级会翻倍。

### 式 (3)｜维持竞争窗口

```python
# game_block_cost.py:133-137
contest_score = 0.0
if np.abs(future_s_gap) < contest_s_gap:   # s_contest = 8.0
    contest_score += 1.0
cost[i, 0] -= contest_weight * contest_score     # w_contest = 1.0
```
完全一致：末端进度差（`future_s_gap`，已做环形展开）落在 ±8 m 内给 −1 的奖励。

### 式 (4)｜纵向进度优势

```python
# game_block_cost.py:128-131
contest_alpha = 1.0 / (1.0 + np.abs(current_s_gap) / (contest_s_gap + eps))
cost[i, 0] -= longitudinal_weight * contest_alpha * future_s_gap    # w_long = 2.0
```
- `α = (1+|Δs_t|/(s_contest+ε))⁻¹` 与论文一致（`:130`）。
- 符号一致：`future_s_gap > 0`（本车在前）→ 减代价。量级：Δs=1 m、α≈0.89 时奖励 ≈1.78。

### 式 (5)｜阻挡对齐 ⚠️ 唯一的语义级偏差

```python
# game_block_cost.py:139-142
if current_s_gap > role_s_margin and current_s_gap < contest_s_gap:   # 1 < Δs_t < 8
    lateral_gap = np.abs(ego_mean_d - opp_mean_d)        # ← 论文是带符号的 Δd
    cost[i, 0] -= block_weight / (1.0 + lateral_gap)     # w_block = 10.0
```

**激活条件与论文一致**（`s_role < Δs_t < s_contest`），**末段均值也一致**（`_traj_tail_mean_d_and_last_s`，`tail_ratio=0.3` → τ=9…12）。但横向差取了绝对值，导致：

| | 论文（带符号） | 代码（绝对值） |
|---|---|---|
| Δd = 0（正对） | 奖励 `w_block` | 奖励 `w_block` |
| Δd = +0.8（本车在左） | 奖励 `w_block/1.8 ≈ 5.6` | 奖励 `10/1.8 ≈ 5.6` |
| Δd = −0.8（本车在右） | 奖励 `w_block/0.2 = 50` | 奖励 `10/1.8 ≈ 5.6` |
| Δd ≤ −1（本车明显在右） | **变为惩罚**（1+Δd ≤ 0） | 仍是小奖励 |

也就是说：论文的公式是**方向性**的——越往右侧（负 d）奖励越大，且低于 −1 后翻转成惩罚；代码则是**对称**的"横向重合即奖励"。这解释了为什么论文正文强调"不要取绝对值"。§4.1 给了修正写法。

### 式 (6)｜时间对齐最小间距

```python
# game_block_cost.py:54-67
dmin = min over τ of ‖ego_traj[τ,:2] − opp_traj[τ,:2]‖₂
# 调用点 :145
dmin = _min_aligned_distance(ego_traj, opp_traj)
```
同构实现还有 `post:100-112`（供式(11)用），区别是后者对多对手累加最小值。
**差异**：论文写 `τ = 1…H`，代码从 `τ = 0` 起算（`compare_len = horizon_len`，循环 `for t in range(compare_len)`）——把"当前时刻"也计入，工程上更保守。

### 式 (7)｜安全间距平方惩罚

```python
# game_block_cost.py:144-148
if dmin < safe_dist:                       # g_safe = 1.0
    violation = safe_dist - dmin
    cost[i, 0] += safety_weight * violation * violation    # w_safety = 50.0
```
与 `w_safety·[max(0, g_safe − g_min)]²` 逐字一致（`max(0,·)` 由 `if` 承担）。1 m 缺口 0.2 m → 罚 50×0.04 = 2.0，与奖励项的 1~10 同量级，说明这一项是"软"约束；真正的硬约束靠式 (11) 的 0.9 m。

### 式 (8)｜跟踪 + 控制 + 平滑

```python
# biased_mppi_planner.py:262-323  —— 单步代价
cost_track   = self.cost_running_track   * (ex**2 + ey**2)          # Q 的位置分量 4.0
cost_heading = self.cost_running_heading * epsi**2                  # 47.75
cost_speed   = self.cost_running_speed   * ev**2                    # 39.48
cost_u       = u_accel*a**2 + u_steer*delta**2                      # R = (8.43, 20.0)
cost_du      = du_accel*da**2 + du_steer*ddelta**2                  # S = (1.0, 19.26)

# biased_mppi_core.py:277-340  —— 沿时域求和
for t in range(self._horizon):            # τ = 0…H−1
    costs[:, t] = self._cost_func(state_seq[:, t], action_seq[:, t], cost_info)
total_costs = torch.sum(costs, dim=1) + terminal_costs              # 末步用 ref[N] + 零动作
```
- `Q/R/S` 都是对角阵，代码展开成 7 个标量权重；`‖·‖²_Q` 的正确性由 `heading` 用 `atan2(sin,cos)` 归一化角差保证（`:278`）。
- **两处索引差异**：① 论文跟踪项 `τ=1…H`，代码实际算 `τ=0…H`（多一个当前时刻项）；② 论文 `Δu` 从 `τ=1` 起，代码在 `τ=0` 时 `prev_index=0` 使 `Δu≡0`，等价。
- **参考轨迹差异**：论文用递推 `s^r_{τ+1} = s^r_τ + Δt·v^r(s^r_τ)`（速度沿参考线变化），代码用**当前点的固定参考速度**外推：`s_target = s_curr + v_ref_base·(k·dt)`（`:412`）。弯道处参考点会略微偏离论文定义，但实现更省（一次 `interp` 三次）。
- 权重分布值得注意：`heading 47.75 ≈ 12×track`，即**方向误差远比位置误差重要**——这是竞速场景的一个隐含先验。

### 式 (9)｜总评分

```python
# biased_mppi_core.py:322 + :324-338
total_costs = torch.sum(costs, dim=1) + terminal_costs
extra_costs = self._trajectory_cost_func(state_seq_batch, action_seq_batch, cost_info)
total_costs = total_costs + extra_costs          # extra 已在 planner:474 乘过 w_game
```
`c_i = c_track + w_game·c_game` 逐项对应。注意 `w_game` 不在 core 里乘，而在 `_trajectory_game_cost_torch` 内部（`planner:474`）——找权重时容易在这一层绕圈。

### 式 (10)｜赛道边界检查

```python
# biased_mppi_post_selection.py:60-80
lateral_d   = -dx*sin_psi + dy*cos_psi
clear_right = d_right_left[0, best_idx] + lateral_d
clear_left  = d_right_left[1, best_idx] - lateral_d
clear_right_eff = clear_right - half_width - safety_buffer     # −0.155 −0.30
clear_left_eff  = clear_left  - half_width - safety_buffer
clear_eff   = min(clear_right_eff, clear_left_eff)
min_clearance_k = min over τ of clear_eff
valid_boundary[k] = min_clearance_k >= boundary_margin         # 0.06
```
- 结构与式 (10) 一致：`ρ_bd = min_τ min(w_R+d, w_L−d)`，`b = 𝟙[ρ_bd ≥ m_bd]`。
- **两处工程增强**：① 从裕度里额外扣掉**半车宽 0.155 m + 安全垫 0.30 m**（`:68-69`），等效阈值变成 0.515 m 的车心距边界距离；论文的 `m_bd` 只对应最后的 0.06。② `w_L, w_R` 取**最近路点的离散值**（`:62-63`），不是沿 s 插值——赛道点足够密（通常 0.1 m 级）时误差可忽略。
- `τ` 同样是 `0…H`（`horizon_len = H+1`），比论文 `1…H` 多算一步。

### 式 (11)｜车辆碰撞检查

```python
# post_selection.py:100-118
for k in range(sample_count):
    min_dist2_k = min over τ of ‖ego[k,τ,:2] − opp[τ,:2]‖²
    min_dist_k  = sqrt(min_dist2_k)
    if min_dist_k < min_obs_dist[k]: min_obs_dist[k] = min_dist_k   # 跨对手取最小
    if min_dist_k < collision_dist: valid_collision[k] = False      # d_col = 0.9
```
`ρ_obs = min_j g_min,ij`、`o = 𝟙[ρ_obs ≥ d_col]` 完全对应；`d_col = COLLISION_DIST = 0.9` 常量在 `:170`。
工程细节：无对手预测时 `min_obs_dist` 保持 `inf`、`valid_collision` 保持 `True`（`:255-261`），于是式 (11) 自动退化为"只查边界"。

### 式 (12)｜最终选择

```python
# post_selection.py:135-142
for k in range(sample_count):
    if valid_boundary[k] and valid_collision[k] and np.isfinite(costs[k]):
        if costs[k] < best_valid_cost:
            best_valid_cost, best_valid_idx = costs[k], k
if best_valid_idx >= 0: return best_valid_idx
```
`h = b·o` 用布尔与实现，等价于论文的乘积；额外加了 `isfinite` 兜底（NaN 候选直接跳过）。

### §6｜最小违规回退

```python
# post_selection.py:144-166
collision_violation = max(0, collision_dist - min_obs_dist[k])
boundary_violation  = max(0, boundary_margin - min_boundary_clearance[k])
safety_violation    = collision_violation + boundary_violation
if safety_violation < best_violation: ...      # 取总违规最小
```
与论文 `ν = ν_bd + ν_obs` 一致，**且回退时确实不看 `cost`**。注意 `min_boundary_clearance` 存的是已扣半车宽/buffer 的"有效裕度"，所以 `ν_bd` 的物理含义比论文更保守（多扣 0.455 m）。

---

## 4. 实现与论文不等价的四处（按影响排序）

### 4.1 ⚠️ 阻挡项横向差取绝对值（`game_block_cost.py:141`）

论文：`c_block = −γ·(1+Δd)⁻¹`，`Δd = d̄_i − d̄_j`（带符号）。
代码：`−w_block/(1+|Δd|)`。

后果：失去了"选择从哪一侧封堵"的能力，且丢掉了 `Δd ≤ −1` 时的惩罚分支。若要与论文对齐：

```python
# 原
lateral_gap = np.abs(ego_mean_d - opp_mean_d)
cost[i, 0] -= block_weight / (1.0 + lateral_gap)

# 对齐论文（1+Δd 可能为负甚至接近 0，必须先夹住分母）
signed_gap = ego_mean_d - opp_mean_d
denom = 1.0 + signed_gap
if denom > -1e-3 and denom < 1e-3:
    denom = 1e-3
cost[i, 0] -= block_weight / denom
```
两段都在 `@njit` 内，所以用 `if` 而不用 `math.copysign` 更稳妥。⚠️ 直接改会引入数值风险（`Δd → −1` 时爆炸），建议同时夹住分母，并重跑对照实验确认这是"复现"而非"改进"。

### 4.2 参考进度用固定速度外推（`biased_mppi_planner.py:412`）

论文递推里 `v^r` 随 `s^r` 变化；代码用当前最近点的 `v_ref_base` 一次性外推 12 步。直道无影响，连续弯（如 Monaco、Singapore 类赛道）会让参考点系统性偏前/偏后。若做消融，这里是可量化的差异源。

### 4.3 边界裕度多扣半车宽与安全垫（`post_selection.py:68-69`）

论文 `ρ_bd ≥ m_bd`；代码 `ρ_bd − 0.155 − 0.30 ≥ 0.06`。若论文实验跑的就是这份代码，则论文的 `m_bd` 应理解为"加上几何余量后的净空"，表格复现时别把 0.06 当成绝对阈值。

### 4.4 预测步起点 τ=0（`post:103`、`post:30`、`game_block_cost.py:45-46`）

论文多处写 `τ=1…H`，代码从 0 起算（含当前时刻）。影响很小（当前时刻两车必然不重合），但意味着"最小间距"永远是"含当前"的，比论文略保守。

---

## 5. 两张"开关表"：公式何时才生效

工程里公式不是无条件执行的，两个源码布尔决定它们跑不跑（`planner_generators.py:8-12`）：

| 公式组 | 生效条件 | 关掉后的行为 |
|---|---|---|
| (2)(3)(4)(5)(7) | `USE_OPP_PRED=True` **且** `opp_pred_traj` 非空 | 退化为"当前位姿椭圆斥力"（`planner:296-323`），式(6) 也不参与博弈 |
| (1)(10)(11)(12)+§6 | `MPPI_POST_SELECT=True`（`plan(post_select=True)`） | 直接用 `core:513-517` 的 softmax 加权平均结果，不做可行性筛选 |
| (6) | 同时被式(7) 与式(11) 使用 | — |
| (8)(9) | 始终执行 | — |

对照论文的四种消融：SGTP = 两开关都开 + `w_game=60`；标准 MPPI = 都关 + `w_game=0`；"只开后选择" = 只开 `MPPI_POST_SELECT`；"只开博弈代价" = 只开 `USE_OPP_PRED`。

---

## 6. 速查：论文符号 → 代码标识符

| 论文 | 代码 |
|---|---|
| `H` / `N` | `self._horizon` / `self.N`（=12） |
| `K` | `self._num_samples`（=128） |
| `Δt` | `self.dt`（=0.1，注意与 env 的 0.01 区分） |
| `Σ_u` | `sigmas` / `_noise_distribution` |
| `Π_𝓤` | `torch.clamp(..., _u_min, _u_max)` |
| `ξ` | `_state_seq_batch` |
| `U` | `_perturbed_action_seqs` |
| `c_i` | `_last_costs`（后选择直接读它，`post:285`） |
| `Δs_t` / `Δs_{t+H}` | `current_s_gap` / `future_s_gap` |
| `d̄` | `ego_mean_d` / `opp_mean_d` |
| `g_min` | `dmin` / `min_obs_dist` |
| `ρ_bd` | `min_boundary_clearance` |
| `m_bd` | `BOUNDARY_MARGIN` |
| `d_col` | `COLLISION_DIST` |
| `w_game` | `game_cost_weight` |
| `w_block` | `block_weight` |
| `s_contest` / `s_role` | `contest_s_gap` / `role_s_margin` |
| `g_safe` | `safe_dist` |
| `T_tail` | `tail_ratio`（比例，非集合） |
