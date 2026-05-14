# 长期多智能体谈判预定义规则评测体系

## 概述

在长期多智能体谈判模拟中，评测体系由两部分构成：

1. **规则型可计算指标**（Rule-based Computational Metrics）：由环境状态与合同状态自动导出，无需额外 LLM 调用，保证评测的客观性与确定性。
2. **LLM 主观评测**（LLM-based Qualitative Evaluation）：由独立的评估模型对对话记录进行维度打分，捕捉协商策略的语义质量。

本文档详述第一部分——预定义规则的计算方法、不同场景类型下的公式差异、以及各因子如何合成为最终综合得分。

**核心设计原则**：在数据构造阶段即通过确定性伪随机种子（`deterministic_seed`）为每个场景生成一组经济参数，形成 `predefined_outcome_rule`，保证同一场景在不同模型、不同轮次的评测中规则口径完全一致。

---

## 1. 评测指标总览

所有指标由两个顶层函数协同产出（源文件：`negotiation_metrics.py`）：

- `compute_negotiation_rule_metrics(env, predefined_outcome_rule)` — 终局状态标志 + 过程统计 + 合同阶段 + 综合得分
- `compute_negotiation_final_state_metrics(env, predefined_outcome_rule)` — 综合得分 `negotiation_final_state_score ∈ [0, 1]` 与各分项贡献

### 1.1 终局状态标志（Terminal Status Flags）

四个互斥的二值指标，描述 episode 的结束方式：


| 指标键                                     | 取值逻辑                                          |
| --------------------------------------- | --------------------------------------------- |
| `negotiation_terminal_is_success`       | `ctrl.terminal == "success"` → 1.0，否则 0       |
| `negotiation_terminal_is_timeout`       | `ctrl.terminal == "timeout"` → 1.0，否则 0       |
| `negotiation_terminal_is_failure`       | `ctrl.terminal == "failure"` → 1.0，否则 0       |
| `negotiation_terminal_is_max_steps_cap` | `ctrl.terminal == "max_steps"` 或空串 → 1.0，否则 0 |


`terminal` 状态由 `LongTermNegotiationController` 在 episode 终止时写入，判定条件包括：达到指定的最大自然日数、所有参与者达成退出共识、连续无进展超过容忍期限、或超出宏观步数上限。

### 1.2 过程统计（Process Statistics）


| 指标键                                       | 含义               | 数据来源                                                 |
| ----------------------------------------- | ---------------- | ---------------------------------------------------- |
| `negotiation_macro_steps_used`            | 实际宏观步计数          | `env.last_episode_macro_steps`                       |
| `negotiation_n_session_log`               | 会话日志条目数          | `len(ctrl.session_log)`                              |
| `negotiation_n_action_log`                | 动作日志条目数          | `len(ctrl.action_log)`                               |
| `negotiation_n_message_log`               | 消息日志条目数          | `len(ctrl.message_log)`                              |
| `negotiation_visible_history_total_lines` | 所有 agent 可见历史行数和 | `sum(len(v) for v in ctrl.visible_history.values())` |
| `negotiation_participant_mean_cash`       | 参与者现金均值          | `mean(agent_resources[agent].cash)`                  |
| `negotiation_participant_min_cash`        | 参与者现金最小值         | `min(agent_resources[agent].cash)`                   |


### 1.3 主合同阶段（Primary Contract Phase）

`negotiation_primary_contract_phase` 将主合同的离散状态映射为序数值，反映协商在法律文书层面的进展深度：


| 合同状态       | 映射值  | 语义          |
| ---------- | ---- | ----------- |
| `proposed` | 1.0  | 至少一方已提出合同草案 |
| `amended`  | 2.0  | 合同已经过至少一次修改 |
| `accepted` | 3.0  | 各方均已接受合同条款  |
| `signed`   | 4.0  | 合同已完成正式签署   |
| `rejected` | −1.0 | 合同被明确拒绝     |
| 其他 / 合同不存在 | 0.0  | 尚未进入合同流程    |


---

## 2. 综合得分 `negotiation_final_state_score`

综合得分将谈判质量分解为**六个可解释因子**，每个因子 ∈ [0, 1]，通过**场景相关的权重向量**线性加权后截断至 [0, 1]。

### 2.1 因子定义

#### 因子 1：终局成功因子（Success Factor）

```
F_success = 1.0  (if terminal == "success")
            0.0  (otherwise)
```

反映 episode 是否在允许的步数与期限内正常结束（非超时、非失败、非达到步数上限）。

#### 因子 2：主合同进展因子（Primary Contract Progress Factor）

```
F_primary = f_status(primary_contract.status)
```

其中状态映射函数 `f_status` 定义为：

```
f_status(s) = 0.25  (s = "proposed")
              0.50  (s = "amended")
              0.75  (s = "accepted")
              1.00  (s = "signed")
              0.00  (s = "rejected" / "failed" / 其他)
```

**设计意图**：单纯的合同提议仅获少量分数，每一次修订或接受都推动进展得分线性增长，正式签署方为满分。合同被拒绝或失败时，此项为 0。

#### 因子 3：偿付能力因子（Solvency Factor）

```
F_solvency = (现金 > 0 的参与者人数) / (参与者总人数 N)
```

其中参与者 i 的终局现金为 `cash_i^final`，来自 episode 结束时的系统状态快照。

**设计意图**：惩罚导致任一参与者破产（现金 ≤ 0）的剥削性协商策略。任何参与者被"榨干"都会降低此项得分，从而激励可持续的、多方共赢的协商行为。

#### 因子 4：流动性保持因子（Liquidity Preservation Factor）

```
ΔCash_total = Σ cash_i^final − Σ cash_i^initial

F_liquidity = 1.0  (if ΔCash_total ≥ 0)
              0.0  (if ΔCash_total < 0)
```

其中初始现金值 `cash_i^initial` 来自 `default_agent_resources_bundle()`（定义于 `roles.py`），最终现金值 `cash_i^final` 来自 `ctrl.state_snapshots` 的最后一条记录（已包含合同结算后的经济结果）。

**设计意图**：要求整个多智能体系统的总现金不能缩水。鼓励协商创造增量价值而非纯粹的零和博弈。

#### 因子 5：调度有效性因子（Scheduling Effectiveness Factor）

```
F_scheduling = 1.0 − (N_no_session / N_bookkeeping)
```

其中：

- `N_bookkeeping` = `ctrl.session_log` 中 `kind == "post_session_bookkeeping"` 的记录总数
- `N_no_session` = 上述记录中 `slot_closure_reason == "scheduling_yielded_no_session"` 的数量

**设计意图**：衡量时间槽位的利用效率——因调度失败而未能产生有效会话的 slot 越多，此项得分越低。该因子仅在 `resource_scheduling_management` 场景被赋予非零权重。

#### 因子 6：预定义规则因子（Predefined Rule Factor）

```
F_rule = g(predefined_outcome_rule, F_primary, env)
```

这是整个评测体系中与场景经济设计最紧密耦合的因子，具体计算由场景类型决定的 `payout_mode` 驱动（详见第 3 节）。

### 2.2 加权合成

设场景相关的权重向量为 **w** = (w1, w2, w3, w4, w5, w6)，各分量非负且满足 w1 + w2 + ... + w6 = 1.0。综合得分为：

```
S_final = clip[0,1](
    w1 × F_success
  + w2 × F_primary
  + w3 × F_solvency
  + w4 × F_liquidity
  + w5 × F_scheduling
  + w6 × F_rule
)
```

其中 `clip[0,1](x) = max(0, min(1, x))`，将得分约束在 [0, 1] 区间内。

同时，为便于归因分析，每个因子的加权贡献作为独立指标输出：


| 输出键                                      | 公式                |
| ---------------------------------------- | ----------------- |
| `..._component_terminal_success`         | w1 × F_success    |
| `..._component_primary_contract`         | w2 × F_primary    |
| `..._component_solvency`                 | w3 × F_solvency   |
| `..._component_liquidity_preserved`      | w4 × F_liquidity  |
| `..._component_scheduling_effectiveness` | w5 × F_scheduling |
| `..._component_predefined_rule`          | w6 × F_rule       |


### 2.3 数据来源

`compute_negotiation_final_state_metrics` 的输入并非实时的 `system_state`，而是 `ctrl.state_snapshots` 中的**最后一条记录**（即 episode 终止时控制器写入的 `after_terminal` 快照）。此设计保证了评测数据的时间一致性——所有指标基于同一时刻的快照计算。

若 episode 异常终止导致快照列表为空，则返回 `{negotiation_final_state_score: 0.0, negotiation_final_state_n_snapshots: 0.0}`。

快照中同时提取以下辅助指标，供更细粒度的分析使用：


| 辅助指标键                                      | 含义           |
| ------------------------------------------ | ------------ |
| `negotiation_final_state_n_snapshots`      | 快照总数         |
| `negotiation_final_state_day_closed`       | 最后一个完整自然日编号  |
| `negotiation_final_state_total_cash`       | 终局总现金        |
| `negotiation_final_state_total_cash_delta` | 终局总现金与初始的差值  |
| `negotiation_final_state_min_cash`         | 参与者中最低现金值    |
| `negotiation_final_state_n_solvent`        | 现金 > 0 的参与者数 |
| `negotiation_final_state_solvency_ratio`   | 正现金人数比例      |


---

## 3. 预定义规则因子 F_rule 的计算

F_rule 是连接"协商经济结果"与"场景设计目标"的核心桥梁。其计算逻辑由 `predefined_outcome_rule.payout_mode` 决定，存在两种模式：`margin_split`（利润率分成）和 `procurement_savings`（采购节省）。

### 3.0 共享前置条件：合同生效判定

两种模式共享一个判定——合同是否"生效"决定了经济结算与得分是否被激活：

```
contract_effective = 1.0  (if F_primary ≥ 0.75, 即合同状态 ≥ "accepted")
                     0.0  (otherwise)
```

当合同未被接受或签署时（`proposed` / `amended` / `rejected`），`contract_effective = 0`，规则得分与利润分配均归零。此设计确保协商必须达成实质性的法律协议才能收获经济回报。

---

### 3.1 模式 A：`margin_split`（利润率分成模式）

**适用场景**：`business_coopetition`、`resource_scheduling_management` 及其他非采购/竞标类场景。

**设计思路**：将合同执行质量信号注入利润率计算——协商达成的合同状态越深入，执行信号越强，实现的利润率越高，规则得分与参与方利润分配越丰厚。

#### 步骤 1：计算实现利润率（Realized Profit Margin）

**执行信号（Execution Signal）E**：将合同进展因子线性映射至 [−1, 1]：

```
E = 2 × F_primary − 1      (E ∈ [−1, 1])
```

此映射的含义：`proposed` 时 E = −0.5（折价信号），`signed` 时 E = 1.0（溢价信号）。中间的每次进展推动信号线性攀升。

**原始利润率（Raw Margin）r_raw**：

```
r_raw = b + α_n × s_news + α_e × E
```

其中各参数的含义与来源：


| 参数     | 符号     | 取值            | 来源                        |
| ------ | ------ | ------------- | ------------------------- |
| 基础利润率  | b      | U(0.03, 0.12) | 场景构造时伪随机生成，seed 固定        |
| 新闻信号权重 | α_n    | 0.55          | 固定常数                      |
| 新闻信号   | s_news | ∈ [−1, 1]     | 从场景文本语义推断 + 随机抖动（详见第 6 节） |
| 执行信号权重 | α_e    | 0.45          | 固定常数                      |


**实现利润率（Realized Margin）r_realized**：经双边裁剪约束在合理区间内：

```
r_realized = clip[r_lo, r_hi](r_raw)
            = max(r_lo, min(r_hi, r_raw))

r_lo = −0.25,  r_hi = 0.35   (profit_margin_bounds)
```

裁剪区间的设计反映了现实商业合同中亏损下限与盈利上限的约束——任何单一合同不可能产生无限亏损或无限暴利。

#### 步骤 2：规则因子

```
F_rule = clip[0,1]( (r_realized − r_lo) / (r_hi − r_lo) ) × contract_effective
```

即实现利润率在允许区间 [r_lo, r_hi] 内的线性归一化位置。

**数值举例**：当 r_realized = 0.05 时：

```
F_rule = (0.05 − (−0.25)) / (0.35 − (−0.25)) × 1.0
       = 0.30 / 0.60
       = 0.50
```

合同未生效时（F_primary < 0.75），因子强制为 0。

#### 步骤 3：总利润与两级分配

**总利润 Π_total**：

```
Π_total = V_contract × r_realized × contract_effective
```

其中合同总金额 V_contract ∼ U(120, 420) × 10⁶（1.2 亿至 4.2 亿随机量级），由 `deterministic_seed` 固定。

**第一级 — 公司利润分配**（仅分配给公司角色 `firm_a` ~ `firm_d`，不包括 `investor` / `regulator`）：

```
Π_company(r) = Π_total × ( s_r / Σ_{j∈C} s_j )    (∀ r ∈ C，C = 公司角色集合)
```

其中 s_r = max(0.05, U(0, 1)) 为随机生成后归一化的份额权重。

**第二级 — 个人收入分配**（所有参与者角色）：

```
Π_individual(r) = Π_company(r) × γ_r    (∀ r ∈ P，P = 所有参与者)
```

其中个人提成比例 γ_r ∼ U(0.30, 0.75)，表示个人从其所属公司的利润中获得 30%–75% 的分成。对于非公司角色（如 `investor`、`regulator`），其 company_profit 分量为 0，因此个人收入亦为 0。

#### 步骤 4：本模式产出的指标键


| 指标键                                                    | 说明                 |
| ------------------------------------------------------ | ------------------ |
| `negotiation_predefined_rule_score`                    | F_rule             |
| `negotiation_predefined_rule_enabled`                  | 1.0（规则存在）          |
| `negotiation_predefined_rule_payout_mode_procurement`  | 0.0（非采购模式）         |
| `negotiation_predefined_rule_realized_margin`          | r_realized         |
| `negotiation_predefined_rule_total_profit`             | Π_total            |
| `negotiation_predefined_rule_contract_value`           | V_contract         |
| `negotiation_predefined_rule_contract_effective`       | contract_effective |
| `negotiation_predefined_rule_news_signal`              | s_news             |
| `negotiation_predefined_rule_company_profit_{role}`    | Π_company(role)    |
| `negotiation_predefined_rule_individual_profit_{role}` | Π_individual(role) |


> 注意：此模式下**不产出** `realized_price`、`reference_price`、`buyer_savings_ratio`、`buyer_savings_per_unit` 等采购类指标。

---

### 3.2 模式 B：`procurement_savings`（采购节省模式）

**适用场景**：`wet_market_competition`（湿市场摊位竞争）、`competitive_bidding`（竞标）。

**设计思路**：以买方节省金额比例衡量协商成效——买方/采购方将实际成交价压低到市场参考价以下越多，节省越大，得分越高。

#### 步骤 1：计算节省量

- **市场参考单价**：P_ref = reference_unit_price ∼ U(175, 395)，由 seed 固定
- **实际成交单价**：P_realized = env 中主合同的 `terms.price`（由协商过程决定）

```
单位节省    S = P_ref − P_realized
节省比例    ρ = S / P_ref
```

#### 步骤 2：规则因子

**满分节省门槛**：ρ_full = full_score_savings_fraction ∼ U(0.12, 0.22)，由 seed 固定

```
F_rule = clip[0,1]( ρ / ρ_full ) × contract_effective   (if S > 0 且 P_ref > 0)
         0                                               (if S ≤ 0)
```

**直觉解释**：

- 只有当实际成交价**低于**参考价（买方确实省了钱）时才能得分
- 节省比例 ρ 达到 ρ_full（如 12%）即可获得满分 F_rule = 1
- 这意味着协商仅仅"谈成合同"不够，必须"谈出省钱的结果"

**数值举例**：假设 P_ref = 300，P_realized = 255，ρ_full = 0.15，合同已生效：

```
S = 300 − 255 = 45,   ρ = 45/300 = 0.15
F_rule = clip[0,1](0.15 / 0.15) × 1.0 = 1.0  (满分)
```

若 P_realized = 276（ρ = 0.08）：

```
F_rule = clip[0,1](0.08 / 0.15) × 1.0 = 0.533
```

#### 步骤 3：利润分配

**买方节省分配**（买方角色均分）：

```
Π_buyer(r) = ( max(0, S) × λ × contract_effective ) / N_buyers   (∀ r ∈ B, 买方角色集合)
```

其中 B = `buyer_roles`（默认取第一个公司角色），λ = 1.0（`savings_cash_scale`，节省金额直接 1:1 兑现为现金利润）。

**卖方成交奖金**（卖方角色平分）：

```
Π_seller(r) = ( B_closure × contract_effective ) / N_sellers   (∀ r ∈ S, 卖方角色集合)
```

其中 S = { r ∈ C | r ∉ B }（非买方的公司角色），B_closure ∼ U(5, 16) × N_sellers（成交奖金池，与卖方人数成正比）。

#### 步骤 4：本模式产出的指标键


| 指标键                                                    | 说明                             |
| ------------------------------------------------------ | ------------------------------ |
| `negotiation_predefined_rule_score`                    | F_rule                         |
| `negotiation_predefined_rule_enabled`                  | 1.0                            |
| `negotiation_predefined_rule_payout_mode_procurement`  | 1.0（采购模式）                      |
| `negotiation_predefined_rule_realized_price`           | P_realized                     |
| `negotiation_predefined_rule_reference_price`          | P_ref                          |
| `negotiation_predefined_rule_buyer_savings_per_unit`   | max(0, S) × contract_effective |
| `negotiation_predefined_rule_buyer_savings_ratio`      | max(0, ρ) × contract_effective |
| `negotiation_predefined_rule_contract_effective`       | contract_effective             |
| `negotiation_predefined_rule_contract_value`           | V_contract（仅作背景值）              |
| `negotiation_predefined_rule_individual_profit_{role}` | Π_buyer(r) 或 Π_seller(r)       |


> 注意：此模式下 `realized_margin = 0.0`、`total_profit = 0.0`、`company_profit_{role}` 不产生——采购模式不使用利润率/公司分红口径。

---

### 3.3 两种模式的本质差异


| 维度      | margin_split                  | procurement_savings       |
| ------- | ----------------------------- | ------------------------- |
| 经济逻辑    | 合同执行质量 → 利润率 → 得分与分红          | 买方压价成效 → 节省比例 → 得分与分配     |
| 核心变量    | 实现利润率 r_realized              | 买方节省比例 ρ                  |
| 协商最优策略  | 追求合同深度（accepted/signed）+ 高利润率 | 追求低于参考价的成交价               |
| 利润耦合    | 与合同金额直接相乘                     | 与合同金额解耦（仅靠节省量）            |
| 分配结构    | 两级：公司分成 → 个人提成                | 单级：买方均分节省 + 卖方分奖金         |
| 核心得分驱动力 | 协商进展（F_primary）               | 价格谈判（P_realized vs P_ref） |


---

## 4. 各场景的权重配置

### 4.1 权重生成机制

权重的读取链为：`predefined_outcome_rule.score_weights`（场景构造时写入）→ `_scene_score_weights()`（评测时读取并合并默认值）→ `compute_negotiation_final_state_metrics()`（应用）。

若 `predefined_outcome_rule` 中不含 `score_weights` 或不含某特定键，则回退到模块级全局默认值。

### 4.2 完整权重对照表


| 因子           | 默认（全局） | business_coopetition | wet_market_competition | resource_scheduling_management | competitive_bidding |
| ------------ | ------ | -------------------- | ---------------------- | ------------------------------ | ------------------- |
| w_success    | 0.30   | 0.25                 | 0.22                   | 0.20                           | 0.20                |
| w_primary    | 0.20   | 0.25                 | 0.20                   | 0.20                           | 0.22                |
| w_solvency   | 0.15   | 0.15                 | 0.18                   | 0.15                           | 0.17                |
| w_liquidity  | 0.10   | 0.10                 | 0.15                   | 0.10                           | 0.13                |
| w_rule       | 0.25   | 0.25                 | 0.25                   | 0.15                           | 0.28                |
| w_scheduling | 0.00   | 0.00                 | 0.00                   | 0.20                           | 0.00                |


### 4.3 场景语义解读

- **business_coopetition（商业竞合）**：w_primary = 0.25、w_rule = 0.25 并重。强调正式协议达成（合同状态）与合同利润率质量的同等重要性。w_success 同为 0.25，鼓励在期限内完成谈判。
- **wet_market_competition（湿市场竞争）**：w_solvency = 0.18、w_liquidity = 0.15 相对较高，反映摊位经营者对现金流敏感的现实约束。w_rule = 0.25 度量买方压价成效（procurement_savings 模式）。
- **resource_scheduling_management（资源调度管理）**：w_scheduling = 0.20 首次激活调度因子，衡量时间槽位利用率与多主体协调质量。w_rule 降至 0.15 以让渡空间给调度维度。
- **competitive_bidding（竞标）**：w_rule = 0.28 最高，突出竞标场景下价格节省的核心地位。w_primary = 0.22 次之，强调中标与签约。

---

## 5. 合同结算在环境中的自动执行

环境对象 `LongTermNegotiationEnv` 在 `run_episode_async()` 末尾通过 `_apply_contract_status_settlement_if_needed()` 自动执行经济结算。完整流程：

**步骤 1** — 检查结算是否已被应用（防止重复）。若 `_contract_status_settlement_applied = True`，直接返回。

**步骤 2** — 验证前置条件：`predefined_outcome_rule` 存在、主合同 `primary_contract_id` 存在、对应的 `NegotiationContract` 对象可获取、合同状态为有效值（`proposed` / `amended` / `accepted` / `signed` / `rejected` / `failed`）。

**步骤 3** — 调用 `compute_predefined_rule_settlement_by_contract_status(env, predefined_outcome_rule, contract_status)`，以**主合同当前状态**（而非 `terminal == "success"`）驱动规则计算。此设计的关键在于：即使 episode 最终因超时或其他原因终止，只要在过程中达成了 `accepted` 或 `signed` 的合同状态，结算仍然被触发。

**步骤 4** — 遍历所有参与者，优先获取 `negotiation_predefined_rule_individual_profit_{role}` 键；若不存在，回退取 `company_profit_{role}`。将对应的利润值直接累加到 `system_state.agent_resources[agent_name]["cash"]`。

**步骤 5** — 记录 `contract_settlement_applied` 执行事件到 `ctrl.execution_log`，包含每位参与者的结算金额与总计。设置 `_contract_status_settlement_applied = True`。

**影响**：最终 `agent_resources` 中的现金值已包含合同结算的经济结果，该值同时被 `compute_negotiation_final_state_metrics` 的快照捕获，用于计算偿付能力与流动性因子。

---

## 6. 新闻信号 s_news 的生成机制

新闻信号 s_news 在场景构造时生成（`scenario_loader.py:_build_predefined_outcome_rule()`），用于模拟外部市场条件对合同经济性的影响：

```
s_news = clip[−1, 1]( 0.7 × s_sentiment + 0.3 × ε )
```

其中：

- `s_sentiment`：由 `_infer_news_sentiment_signal()` 从场景叙事文本中提取方向性信号。通过检测价格趋势描述（"surge" → 正、"collapse" → 负）、竞争强度、风险关键词等语义特征推断。
- `ε ∼ U(−0.35, 0.35)`：随机扰动项，由 `deterministic_seed` 控制，引入适度不确定性，使同一语义倾向下的场景仍有差异化经济参数。
- 0.7 / 0.3 的混合权重：确保新闻信号以语义方向为主（70%），随机因素为辅（30%），避免信号完全由噪声主导。

此外，`_collect_scenario_bound_news_threads()` 根据场景类型、参与人数、日历天数等元信息绑定一组新闻线索（如湿市场绑定"冷链挤压与易腐供给"、竞合绑定"劳动力短缺与技能错配"等），这些线索以**叙事材料**形式注入模拟的上下文背景文本中（如 `environment_context["news_threads"]` 和外部事件），间接影响 LLM agent 的行为，但**不直接改变**预定义规则的数学公式。

---

## 7. 确定性复现机制

为保证科学评测的可复现性，`predefined_outcome_rule` 的所有经济参数通过确定性伪随机生成：

```
seed = hash( codename, lineup, num_participants, scenario_text[:512], entropy )
```

其中 `entropy`（`outcome_rule_entropy`）为可选的额外熵源。在 LLM 批量合成场景时（同一 codename 可能对应多个叙事变体），通过传入不同的熵值（如变体编号）确保子场景之间的经济参数差异化；手写题库中省略此项即可。

以下所有参数均通过 `random.Random(seed)` 生成：


| 参数                | 生成方式                             | 适用模式                |
| ----------------- | -------------------------------- | ------------------- |
| b（基础利润率）          | `rng.uniform(0.03, 0.12)`        | margin_split        |
| ε（新闻信号抖动）         | `rng.uniform(−0.35, 0.35)`       | 两者                  |
| V_contract（合同总金额） | `rng.randint(120, 420) × 10⁶`    | 两者                  |
| P_ref（参考单价）       | `rng.uniform(175, 395)`          | procurement_savings |
| ρ_full（满分节省门槛）    | `rng.uniform(0.12, 0.22)`        | procurement_savings |
| B_closure（成交奖金池）  | `rng.uniform(5, 16) × N_sellers` | procurement_savings |
| s_r（公司利润份额）       | `max(0.05, rng.random())`，归一化    | margin_split        |
| γ_r（个人提成比例）       | `rng.uniform(0.30, 0.75)`        | margin_split        |


同一场景在任意次评测中产生**完全一致的**规则参数，保证了跨模型、跨轮次的可比性。

---

## 8. LLM 主观评测（补充）

当评测运行参数 `run_terminal_llm_eval = True` 时，在规则指标之外附加 `EpisodeLLMEvaluator` 的主观维度评估。评估模型（独立于参与协商的 agent 模型）审阅完整对话历史，对每位参与者按以下七个维度进行打分：


| 维度                                | 范围       | 评估内容             |
| --------------------------------- | -------- | ---------------- |
| `believability`                   | [0, 10]  | 行为是否自然、一致、符合角色设定 |
| `relationship`                    | [−5, 5]  | 是否建立/维护了积极的合作关系  |
| `knowledge`                       | [0, 10]  | 信息获取、推理与利用的有效性   |
| `secret`                          | [−10, 0] | 私密信息（底线/策略）的保护程度 |
| `social_rules`                    | [−10, 0] | 社会规范与协商礼仪的遵守     |
| `financial_and_material_benefits` | [−5, 5]  | 经济收益的获取能力        |
| `goal`                            | [0, 10]  | 个人预设目标的达成程度      |


每位参与者的 `overall_score` 为**各维度分的简单算术平均**：

```
overall_score_i = (1/7) × Σ_{d ∈ D} score_{i,d}    (D = 7 个维度)
```

输出结构中 `p1_rate = (overall_score₁, dim_scores₁)`，`p2_rate = (overall_score₂, dim_scores₂)`。超过两名的评估细节主要记录在 `comments` 字段与模型 trace JSONL 中。聚合统计时使用 `llm_overall_mean`（跨 episode 均值）和 `llm_dimension_scores_mean`（跨 episode 维度均值的均值）。

---

## 9. 建议保留的核心统计字段

在进行模型性能统计时，建议按以下层次组织分析：

**第一层 — 主指标：**


| 字段                              | 说明             |
| ------------------------------- | -------------- |
| `negotiation_final_state_score` | 综合得分，区间 [0, 1] |


**第二层 — 归因分项（解释主指标由哪些因素驱动）：**


| 字段                                       | 对应因子    |
| ---------------------------------------- | ------- |
| `..._component_terminal_success`         | 终局成功贡献  |
| `..._component_primary_contract`         | 合同进展贡献  |
| `..._component_solvency`                 | 偿付能力贡献  |
| `..._component_liquidity_preserved`      | 流动性贡献   |
| `..._component_predefined_rule`          | 预定义规则贡献 |
| `..._component_scheduling_effectiveness` | 调度有效性贡献 |


**第三层 — 合同与生存性：**


| 字段                                                          | 说明         |
| ----------------------------------------------------------- | ---------- |
| `negotiation_primary_contract_phase`                        | 合同进展阶段     |
| `negotiation_final_state_solvency_ratio`                    | 终局正现金参与者比例 |
| `negotiation_final_state_min_cash`                          | 最低参与者现金    |
| `negotiation_participant_mean_cash`                         | 参与者现金均值    |
| `negotiation_terminal_is_success` / `_timeout` / `_failure` | 终局状态分布     |


**第四层 — 规则经济细节：**


| 字段                                                     | 说明                             |
| ------------------------------------------------------ | ------------------------------ |
| `negotiation_predefined_rule_score`                    | 规则因子得分                         |
| `negotiation_predefined_rule_total_profit`             | 总利润（margin_split 模式）           |
| `negotiation_predefined_rule_buyer_savings_ratio`      | 买方节省比例（procurement_savings 模式） |
| `negotiation_predefined_rule_individual_profit_{role}` | 各角色利润                          |


**第五层 — LLM 主观评估（若启用）：**


| 字段                                      | 说明        |
| --------------------------------------- | --------- |
| `llm_aggregate.p1_rate[0]`、`p2_rate[0]` | 各参与者主观总评分 |
| `llm_aggregate.p1_rate[1].{dimension}`  | 各参与者维度细分  |


---

## 10. 代码文件索引


| 模块                     | 源文件路径                                             | 关键函数 / 类                                                                          |
| ---------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------- |
| 规则指标计算                 | `long_term_negotiation/negotiation_metrics.py`    | `compute_negotiation_rule_metrics()`, `compute_negotiation_final_state_metrics()` |
| 预定义规则 payout 分发        | 同上                                                | `_compute_predefined_rule_payout_metrics()`                                       |
| procurement_savings 计算 | 同上                                                | `_compute_procurement_savings_metrics()`                                          |
| 合同状态因子映射               | 同上                                                | `primary_contract_status_factor()`                                                |
| 调度有效性因子                | 同上                                                | `_scheduling_effectiveness_factor()`                                              |
| 场景权重读取与合并              | 同上                                                | `_scene_score_weights()`                                                          |
| 合同结算按状态计算              | 同上                                                | `compute_predefined_rule_settlement_by_contract_status()`                         |
| 终局状态快照记录               | 同上                                                | `build_rule_evaluation_state_record()`                                            |
| 预定义规则构造（数据侧）           | `long_term_negotiation/scenario_loader.py`        | `_build_predefined_outcome_rule()`                                                |
| 新闻语义信号推断               | 同上                                                | `_infer_news_sentiment_signal()`                                                  |
| 场景新闻线索绑定               | 同上                                                | `_collect_scenario_bound_news_threads()`                                          |
| 角色集合与排序                | `long_term_negotiation/roles.py`                  | `FIRM_ROLES_ORDER`, `negotiation_role_order()`                                    |
| 角色人格与默认资源              | 同上                                                | `ROLE_PERSONA_EN`, `default_agent_resources_bundle()`                             |
| 环境结算应用                 | `long_term_negotiation/env.py`                    | `_apply_contract_status_settlement_if_needed()`                                   |
| 时间线参数                  | `long_term_negotiation/types.py`                  | `NegotiationTimelineParams`                                                       |
| LLM 主观评估               | `long_term_negotiation/llm_evaluation.py`         | `EpisodeLLMEvaluator`                                                             |
| 运行配置加载                 | `long_term_negotiation/negotiation_run_config.py` | `load_negotiation_run_config()`, `build_negotiation_agents_from_run_config()`     |


