# 长期谈判数据合成与评估增强

## 1. 增量多样性约束 (`--incremental-diversity`)

**文件**: `scripts/generate_long_term_negotiation_llm.py`

**问题**: 同一批次内 N 条数据全部并发执行，相互之间没有约束，容易产生批内重复。

**方案**: 新增 `--incremental-diversity` 参数。开启后：
- 按 `scene_type` 分组，**同组内串行生成**，不同组之间仍并发
- 每生成一条，提取 scenario/goals 摘要加入该 scene_type 的"批内语料"
- 下一条的 prompt 同时携带：历史 manifest 摘要 + 批内已生成语料 + 反重复约束

**使用**:
```bash
# 跨批次多样性 + 批内增量多样性
python scripts/generate_long_term_negotiation_llm.py --n 6 --tag ltr_v3 \
    --diversity-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --incremental-diversity
```

**向前兼容**: 不传 flag 时行为完全不变。

**新增函数**:
- `_extract_profile_excerpt()` — 从生成的 EnvironmentProfile 提取 scenario/goals 摘要
- `_batch_corpus_block()` — 将批内已生成条目格式化为反重复约束块
- `_generate_with_incremental_diversity()` — 分组串行生成核心逻辑

---

## 2. 扩大竞争性要求

### 2a. SCENARIO_PROMPT_GUIDE 增强

**文件**: `scripts/generate_long_term_negotiation_llm.py`

| scene_type | 改动 |
|---|---|
| `business_coopetition` | 新增 zero-sum tradeoff、winner-takes-most dynamics、直接冲突目标 |
| `wet_market_competition` | 新增 price war 螺旋、reputation 攻击、customer poaching |
| `resource_scheduling_management` | 大幅增强：从"协调"转为"争夺"，新增抢占、囤积、undercut 策略 |
| `business_outsourcing` | **新增** — milestone 博弈、rework 责任转嫁、subcontractor 压价 |
| `competitive_bidding` | **新增** — bid spread 策略、信息泄露、win-at-any-cost 压力 |

### 2b. ROLE_PERSONA_EN 增强

**文件**: `sotopia/settings/long_term_negotiation/roles.py`

| 角色 | 成本变化 (daily_fixed / short_term_debt) | 核心改动 |
|---|---|---|
| firm_a (买方) | 65→95 / 120→160 | 加入 rival buyer 竞争压力，目标 "outperform rival buyers" |
| firm_b (主供应商) | 80→110 / 90→130 | 加入 challenger poaching 威胁，目标 "defend market share" |
| firm_c (挑战者) | 72→100 / 150→200 | 债务加重驱动攻击性，目标 "poach 2+ accounts" |
| firm_d (后发者) | 68→90 / 110→160 | 加入时间窗口压力，目标 "capture premium accounts from each incumbent" |

`default_agent_resources_bundle()` 同步更新。

### 2c. scenario_loader 增强

**文件**: `sotopia/settings/long_term_negotiation/scenario_loader.py`

- `DIALOGUE_STYLE_SYNTHESIS_APPEND_EN`: 新增 competitive tone differentiation（aggressive challenger / defensive incumbent / calculating latecomer）
- `DIALOGUE_STYLE_EVAL_RUBRIC_EN`: 新增 competitive authenticity 评估维度
- `full_score_savings_fraction`: 0.08-0.16 → 0.12-0.22（加大得分难度）

---

## 3. 多方可交流 Skill 指导

**文件**: `sotopia/settings/long_term_negotiation/negotiation_llm_agent.py`

**问题**: 模型可能只与单一方谈判，不会利用多方竞争来最大化收益。

**方案**: 新增 `MULTI_PARTY_NEGOTIATION_SKILL` 常量，包含三个层面的指导：

### 核心策略 (Core Tactics)
1. **货比三家**: 在接受任何报价之前，与所有可用的对手方沟通
2. **交叉引用**: 用一方的报价向另一方施压（"B 报了 X — 你能更优惠吗？"）
3. **利用竞争**: 让竞争对手知道对方也在出价，制造竞价张力
4. **联盟意识**: 与一个对手结盟施压第三方，随时准备切换
5. **最大化总收益**: 追踪所有对手方组合中的 BATNA（最佳替代方案）
6. **信息套利**: 利用一方透露的信息作为对另一方的筹码

### 商业策略工具箱 (Business Tactics)
- **锚定效应**: 以激进但可辩护的数字开局
- **让步模式**: 每次让步比上次更小，永远不白给
- **期限杠杆**: 利用对方的时间压力获取更优条款
- **捆绑谈判**: 谈整体方案（价格+交付+质量+罚则），不只是价格
- **离开的可信度**: 愿意离开并让对方看到
- **蚕食战术**: 主要条款达成后再要小附加条件
- **沉默施压**: 出价后保持沉默，让对方先开口

### 对话模板 (Dialogue Templates)
- 开场比价: "I'm talking to a few vendors today. What's your best offer?"
- 交叉施压: "Another seller quoted me X with Y delivery. Can you do better?"
- 制造紧迫: "I need to close by [time]. Your offer or I move on."
- 让步交换: "I can come up on price if you extend warranty to 30 days."
- 防御挖角: "Their price is lower — but their delivery was late twice last month."
- 建立联盟: "If we both hold firm, the buyer can't play us against each other."
- 促成成交: "We're close. Meet me at [final offer] and we sign now."

### 角色特定策略
- **买方/采购**: 向每个卖家询价，比较总方案，让卖家知道他们在竞争
- **主供应商/防守方**: 用口碑对抗新进入者，考虑与其他防守方结盟
- **挑战者**: 公开攻击防守方弱点，先用低价吸引再靠量/速度盈利
- **后发入场者**: 找到最弱的防守方突破口，把溢价包装成可靠性保险

注入位置: `build_negotiation_social_llm_agents()` 中每个 agent 的 `goal` 块末尾。
