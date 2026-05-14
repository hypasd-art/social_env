## 1) 规则评测（`rule_metrics`）里的所有分数

这些都来自 `compute_negotiation_rule_metrics()` + `compute_negotiation_final_state_metrics()`。

### A. 终局状态类（0/1）
- `negotiation_terminal_is_success`：`terminal == "success"` 则 1，否则 0  
- `negotiation_terminal_is_timeout`：`terminal == "timeout"` 则 1，否则 0  
- `negotiation_terminal_is_failure`：`terminal == "failure"` 则 1，否则 0  
- `negotiation_terminal_is_max_steps_cap`：`terminal == "max_steps"` 或空串则 1，否则 0  

### B. 过程统计（计数/均值）
- `negotiation_macro_steps_used`：实际宏步数  
- `negotiation_n_session_log`：session_log 条数  
- `negotiation_n_action_log`：action_log 条数  
- `negotiation_n_message_log`：message_log 条数  
- `negotiation_visible_history_total_lines`：所有 agent 可见历史行数之和  
- `negotiation_participant_mean_cash`：各参与者现金均值  
- `negotiation_participant_min_cash`：各参与者现金最小值  

### C. 主合同阶段映射
- `negotiation_primary_contract_phase` 映射：
  - proposed=1.0, amended=2.0, accepted=3.0, signed=4.0, rejected=-1.0, 其他=0.0

---

## 2) 最终综合分 `negotiation_final_state_score` 的计算方式

### A. 先算几个因子
- `success_factor`：`terminal == success` ? 1 : 0  
- `primary_factor`（主合同状态）：
  - proposed=0.25, amended=0.5, accepted=0.75, signed=1.0, rejected/failed=0.0
- `solvency_factor`：`现金>0人数 / 总人数`
- `liquidity_factor`：`final_total_cash_delta >= 0` ? 1 : 0
- `rule_factor`：来自预定义规则（见下一节）

### B. 权重加权求和（并截断到 [0,1]）
\[
\text{final\_state\_score} =
0.3\cdot success\_factor +
0.2\cdot primary\_factor +
0.15\cdot solvency\_factor +
0.1\cdot liquidity\_factor +
0.25\cdot rule\_factor
\]

并同时输出五个分项：
- `negotiation_final_state_score_component_terminal_success`
- `..._primary_contract`
- `..._solvency`
- `..._liquidity_preserved`
- `..._predefined_rule`

---

## 3) 预定义规则相关分数（`negotiation_predefined_rule_*`）

前提：`predefined_outcome_rule` 存在，否则只给 `enabled=0`。

### A. 中间量
- `negotiation_predefined_rule_contract_value`
- `negotiation_predefined_rule_news_signal`
- `negotiation_predefined_rule_realized_margin`

其中：
\[
execution\_signal = 2\cdot primary\_factor - 1
\]
\[
raw\_margin = base\_margin + news\_weight\cdot news\_signal + execution\_weight\cdot execution\_signal
\]
\[
realized\_margin = clip(raw\_margin, lo, hi)
\]

### B. 合同是否“生效”
- `negotiation_predefined_rule_contract_effective`：`primary_factor >= 0.75` 则 1，否则 0

### C. 利润与分配
\[
total\_profit = contract\_value \cdot realized\_margin \cdot contract\_effective
\]
- `negotiation_predefined_rule_total_profit`
- `negotiation_predefined_rule_company_profit_{role}`：按 `company_profit_share` 比例分公司利润
- `negotiation_predefined_rule_individual_profit_{role}`：公司利润再乘 `individual_income_share`

### D. 规则子分
\[
rule\_factor = clip\left(\frac{realized\_margin-lo}{hi-lo},0,1\right)\cdot contract\_effective
\]
- 对应键：`negotiation_predefined_rule_score`

---

## 4) LLM 终局评测分（`llm_aggregate`）

如果 `run_terminal_llm_eval=True`，会用 `EpisodeLLMEvaluator` 输出每个 agent 的维度分，再聚合。

默认维度（`SotopiaDimensions`）：
- `believability`（0~10）
- `relationship`（-5~5）
- `knowledge`（0~10）
- `secret`（-10~0）
- `social_rules`（-10~0）
- `financial_and_material_benefits`（-5~5）
- `goal`（0~10）

### 聚合方式
- 每个 agent 的 `overall_score` = 该 agent 所有维度分的**简单平均**
- 返回结构中：
  - `p1_rate = (agent_1_overall_score, agent_1各维度字典)`
  - `p2_rate = (agent_2_overall_score, agent_2各维度字典)`

> 注意：当前 `ScriptEnvironmentResponse` 只显式放 `p1_rate/p2_rate`，多于两人的细节主要在 `comments`/trace 中看。

---

## 5) 你做统计时建议保留的核心字段（最实用）

- 主指标：`negotiation_final_state_score`
- 解释项：5 个 `...score_component_*`
- 合同进展：`negotiation_primary_contract_phase`
- 生存性：`negotiation_final_state_solvency_ratio`、`...min_cash`
- 规则经济性：`negotiation_predefined_rule_score`、`...total_profit`
- 主观评估（若开 LLM eval）：`p1_rate[0]`、`p2_rate[0]` + 各维度分