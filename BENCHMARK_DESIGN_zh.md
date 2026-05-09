# 长周期多智能体社会系统 Benchmark 方案（仅设计，不改代码）

> 目标：在现有 `social_env/sotopia` 基础上，从“短对话博弈”扩展为“长周期、事件驱动、状态可累积”的社会系统评测框架。  
> 本文只给实施方案与改造路径，不直接修改任何现有代码。

---

场景一：长期商业谈判
1. 基本设定
● Agent构成：公司A（买方）、公司B（卖方）、投资人、监管机构（4类角色，各司其职）
● 时间尺度：30–100 steps（对应真实世界数周，分阶段推进）
● 核心目标：模拟企业并购、长期合作等真实谈判场景，测试Agent的信息利用、声誉管理、长期策略规划能力
2. 各Agent核心目标
● 公司A（买方）：以最低合理价格完成收购/合作，控制成本，降低风险
● 公司B（卖方）：以最高合理价格出售/合作，同时维持自身市场声誉，保障长期发展
● 投资人：实现投资收益最大化，平衡风险与回报，监督谈判进程
● 监管机构：确保谈判过程合规，维护市场秩序，防范垄断、违规操作等风险
3. 核心机制
（1）合同机制（核心约束）
contract = {
    "price": ...,               # 核心交易价格
    "conditions": ...,          # 附加条件（如履约要求、违约责任）
    "deadline": ...,            # 履约期限（对应step数）
    "signatories": [...],       # 签约方（Agent列表）
    "penalty": ...              # 违约惩罚（声誉扣除、资源赔偿）
}
（2）关键外部事件（驱动谈判走向）
● Day 10（对应step10）：媒体曝光公司B财务问题，降低其议价能力，影响公司A的报价策略
● Day 15（对应step15）：监管政策收紧，新增并购合规要求，增加谈判复杂度
● Day 20（对应step20）：第三方竞争对手加入报价，改变双方议价权力平衡
● 其他：投资人撤资警告、市场需求变化等，进一步模拟真实不确定性
4. 关键挑战
● 信息不对称：Agent需通过沟通、观察，推断对方真实底线与实力
● 声誉管理：谈判过程中的行为的（如反悔、欺骗）会影响长期声誉，进而影响后续合作

下面的代码要实现这样的场景

## 1. 总体思路

现有 Sotopia 可复用的核心很强：角色画像、场景对象、异步多智能体交互、评测器框架、Redis 存储、日志体系。  
建议采用 **“保留主干 + 叠加系统层”**：

- 保留：`LLMAgent`、`ParallelSotopiaEnv`、`run_async_server` 的主流程思想
- 新增：`SystemState`（状态层）、`EventEngine`（外部事件层）、`ActionDispatcher`（结构化行为执行层）
- 扩展：`ActionType`、`EnvironmentProfile`、`AgentProfile`、`EpisodeLog`

核心闭环保持为：

`observe -> communicate -> act -> update_state`

事件不建议每步触发，改为“日终触发”（`end_of_day`）并由 `config` 控制：

`intra_day_actions -> end_of_day(config) -> apply_events -> snapshot`

---

## 2. 需要改哪些代码（模块级）

## A. 消息与动作层

- `sotopia/messages/message_classes.py`
  - 扩展 `ActionType`：
    - `propose_contract`, `accept`, `reject`
    - `transfer_resource`, `defect`
    - `invest`, `withdraw`, `vote`
  - 扩展 `AgentAction.argument`：支持 `dict`（结构化参数）

目的：把“行为”从自然语言变成可执行指令。这些行为都要对环境中的状态造成影响

## B. 环境层

- 新增文件：`sotopia/envs/social_system_env.py`
  - 基于 `ParallelSotopiaEnv` 继承实现 `SocialSystemEnv`
  - 在 `astep()` 中增加：
    1. 行为执行（ActionDispatcher）
    2. 日内状态更新（SystemState）
  - 新增 `end_of_day()`：
    1. 根据 `config` 判断是否触发外部事件（EventEngine）
    2. 执行日终结算（合约到期、惩罚、规则刷新）
    3. 状态快照持久化（SystemStateSnapshot）

目的：实现长周期系统演化，而不是仅对话回合推进。

## C. 事件层（新增）

- 新增：`sotopia/events/event_engine.py`
- 新增：`sotopia/events/effect_dsl.py`

功能：
- 读取脚本化事件（可随机扰动触发时刻/强度）
- 根据 `visibility` 做信息披露（public/partial/private）
- 对系统状态施加 effect（如信任下降、资源变动、动作禁用）
- 所有事件默认在 `end_of_day` 阶段触发；是否触发由 `config` 中的开关和时间表决定

## D. 状态与合约层（新增）

- 新增：`sotopia/state/system_state.py`
- 新增：`sotopia/state/contracts.py`

功能：
- 管理 `trust_matrix`、`market_state`、`resource_pool`、`agent_resources`、`reputation`
- 管理合同生命周期（创建、签署、违约、结算）

## E. Agent 层

- 新增：`sotopia/agents/social_agent.py`
- 新增：`sotopia/agents/memory.py`

功能：
- 在 `aact()` 中接入长期记忆检索（短期窗口 + 向量记忆）
- 感知系统状态摘要后输出结构化动作

## F. 评测层
- 对于评测要根据环境的变化统计不同agent的分数，同时还要使用大模型评测
- 在现有 `sotopia/envs/evaluators.py` 基础上新增四类评测器：
  - `individual`：utility / regret / consistency
  - `social`：total_welfare / gini / stability
  - `behavioral`：cooperation / punishment / deception / reciprocity
  - `long_horizon`：历史利用、策略调整、长期目标一致性

---

## 3. 数据部分怎么处理

> **所有数据 schema / 工厂函数已落地为独立可 import 的模块**，路径：  
> `sotopia/benchmark_v2_data_models.py`  
> 该文件**不修改任何现有代码**，只新增并列的 V2 模型与工具函数；老 `AgentProfile / EnvironmentProfile / EpisodeLog` 与老脚本完全不受影响。

## 3.1 设计要点（与 v1 的关系）

- **增量并列，不改老类**：用 `AgentProfileV2 / EnvironmentProfileV2 / EpisodeLogV2` 作为新表，老类保持不动；
- **后端无关**：沿用既有 `is_local_backend()` 双分支，本地 JSON 与 Redis JsonModel 都能用；
- **`pk` workaround 同步**：所有 JsonModel 子类都重写 `__init__` 显式传 `pk=""`，与 `persistent_profile.py` 修法一致，避免 `redis-om` 元类把 `pk` 替换成 `ExpressionProxy` 导致 pydantic 校验失败；
- **关系数据继承复用**：`RelationshipProfile` 不新增字段，作为 `SystemStateSnapshot.trust_matrix` 的初值来源即可。

## 3.2 模块覆盖范围（一一对应）

| 设计目标 | 文件中的对应物 | 说明 |
|---|---|---|
| AgentProfile 增量字段 | `AgentProfileV2` | 加 `initial_resources / initial_reputation / risk_preference / role_type` |
| EnvironmentProfile 增量字段 | `EnvironmentProfileV2` | 加 `scenario_type / n_agents / max_days / intra_day_steps / event_schedule_pk / system_state_init` |
| 事件脚本 | `EventScript` + `EffectOp` | `apply_days / intraday / step` 共同决定触发时机；`effects` 用最小 DSL（`set / delta / disable_action / broadcast`） |
| 合约对象 | `Contract` | 通用条款 `terms` + 违约惩罚 `penalty` + 状态机 `status / history`，覆盖谈判 / 借贷 / 配额三类场景 |
| 日终状态快照 | `SystemStateSnapshot` | day 粒度，含 `trust_matrix / public_opinion / market_state / resource_pool / agent_resources / agent_reputation` |
| Episode 日志（v2） | `EpisodeLogV2` | `schema_version=2` + 关联 `state_trajectory_pks / events_log_pks / contracts_pks / final_metrics`；与老 `EpisodeLog` 通过可选的 `legacy_episode_pk` 双写过渡 |

## 3.3 工厂函数（最常用入口）

| 入口 | 用途 |
|---|---|
| `upgrade_agent_profile(legacy, ...)` | 把老 `AgentProfile` 升级成 V2，老字段沿用、新字段走默认值或参数；不自动落库，便于离线脚本批处理 |
| `upgrade_environment_profile(legacy, ...)` | 同上，针对 `EnvironmentProfile` |
| `make_initial_state_snapshot(...)` | 生成 `day=0` 的 `SystemStateSnapshot`，统一处理 `trust_matrix / agent_reputation` 的默认结构 |
| `make_event_script_from_dict(spec)` | 从 JSON / dict 构造 `EventScript`，自动把 `effects` 项转 `EffectOp` |

## 3.4 EpisodeLog 兼容策略

- **不强制迁移老数据**。老 `EpisodeLog` 仍按 `schema_version=1` 处理（隐式约定）；
- 新 benchmark 一律写 `EpisodeLogV2`（`schema_version=2`），且通过 `legacy_episode_pk` 可与对应的老 `EpisodeLog` 关联（双写过渡期使用）；
- 老数据若与当前代码 schema 冲突（例如 `messages` tuple/str 类型不一致），按 `DATA_GUIDE_zh.md` 的"raw 导出"或"删除老记录"处理，**与 v2 模型无关**，互不干扰。

## 3.5 最小验证

```bash
# 在 social_env/ 目录下
SOTOPIA_STORAGE_BACKEND=local python -c "
from sotopia.benchmark_v2_data_models import (
    AgentProfileV2, EnvironmentProfileV2, EventScript, Contract,
    SystemStateSnapshot, EpisodeLogV2,
    make_initial_state_snapshot, make_event_script_from_dict,
)
a = AgentProfileV2(first_name='Alice', last_name='Z', role_type='buyer')
e = EnvironmentProfileV2(scenario='demo', scenario_type='investment',
                        n_agents=3, max_days=10, intra_day_steps=4)
es = make_event_script_from_dict({'name':'rate_hike','category':'market',
    'visibility':'public','apply_days':[3,6],
    'effects':[{'op':'delta','target':'market_state.interest_rate','value':0.02}]})
ss = make_initial_state_snapshot(episode_pk='epi_x', agent_pks=['a','b'])
print('ALL OK', a.role_type, e.scenario_type, es.name, ss.day)
"
```

Phase 0 PoC 阶段直接 `from sotopia.benchmark_v2_data_models import ...` 即可走通"造数据 → 写库 → 离线统计"的最短链路；后续 Phase 1+ 接入运行时（事件触发 / 状态更新 / 合约结算）时再扩展配套模块（§B/C/D），数据层不需要再改。

---

## 4. 交互方式如何实现

## 4.1 公开/私下信息流

沿用现有可见性逻辑并强化：

- `public`：所有 agent 可见
- `partial`：仅部分 agent 可见
- `private`：仅发送者/接收者可见

事件与动作都附带 `visibility`，统一进入 observation 渲染。

## 4.2 结构化动作输出（推荐 JSON Schema）

让模型按 schema 输出：

- `action_type`
- `argument`（按动作类型约束）
- 可选 `target_pk`（指向合同/对象）

再由 `ActionDispatcher` 执行副作用（转账、签约、违约惩罚、投票计票）。

## 4.3 时间推进

建议采用“双层时间”：

- `intra_day_step`：日内交互（多次 `astep`，默认不触发外部事件）
- `end_of_day`：日终触发事件并持久化快照（由 `config` 决定触发）

调度方式建议支持三种（用于日内交互）：

- `round-robin`（调试友好）
- `parallel`（更贴近现实）
- `scheduled-by-role`（如监管优先）

推荐默认：`parallel + end_of_day_event_mode`

### `config` 驱动触发（建议）

```yaml
time:
  max_days: 60
  intra_day_steps: 4                # 每天执行4轮 agent 交互

events:
  trigger_mode: "end_of_day"        # end_of_day | intraday | mixed
  end_of_day_enabled: true
  intraday_enabled: false
  schedule_source: "event_script"   # event_script | policy_rules
  apply_days: [1, 5, 10, 15, 20]    # 可选；为空则按 event_script.step
```

触发规则：
- `end_of_day_enabled=false`：当天不触发任何外部事件；
- `apply_days` 非空：仅在指定 day 触发；
- `apply_days` 为空：按 `EventScript` 的 `step/day` 字段触发；
- `trigger_mode=mixed`：仅允许被标记为 `intraday=true` 的事件在日内触发。

---

## 5. 三类场景的落地映射

## 场景1：长期商业谈判

- 角色：buyer / seller / investor / regulator
- 关键动作：`propose_contract`, `accept/reject`, `defect`, `speak`
- 关键事件：财务曝光（news）、监管收紧（policy）、竞争报价（market）
- 指标：deal quality、efficiency、信息利用、声誉变化

## 场景2：投资与违约

- 角色：5-10 个异构 agent
- 关键动作：`propose_contract(loan)`, `transfer_resource`, `defect`
- 关键事件：利率变动、市场崩盘、信用评级公告
- 指标：total welfare、default rate、trust stability、gini

## 场景3：公共资源管理

- 角色：5-8 个共享公共资源
- 关键动作：`transfer_resource(consume)`, `vote`, `propose_contract(quota)`
- 关键事件：干旱、政策限额、技术突破
- 指标：sustainability、fairness、cooperation、stability

---

## 6. 分阶段实施（推荐）

## Phase 0（PoC，1-2天）

- 只做一个最小场景（3 agent，10 days，每天 2-3 个 intra_day_steps）
- 只新增 1-2 个动作（`transfer_resource`, `propose_contract`）
- 只做一个事件类型（`market`，并在 `end_of_day` 触发）
- 输出状态快照，验证状态可累积

## Phase 1（单场景可用，约1周）

- 跑通“投资与违约”场景
- 加入违约逻辑与基础指标（welfare/default/gini）
- 生成第一版可复现实验脚本

## Phase 2（三场景完整，2-3周）

- 接入三类事件与四层评测
- 支持 50-200 steps
- 增加长期记忆模块（可先简化为检索 top-k）

## Phase 3（论文级评测）

- 多模型、多种子批量跑
- 统计显著性 + 案例分析 + 可视化

---

## 7. 工程风险与应对

- **LLM 调用量过大**：先小规模（3 agent x 10 steps）逐步放大；做并发与缓存
- **提示词窗口爆炸**：必须做“状态摘要 + 记忆检索”，不能把全历史硬塞进 prompt
- **事件过多噪声**：事件脚本“主干固定 + 幅度随机化”
- **可复现性不足**：固定 random seed + 固定事件模板 + 报告多种子均值
- **旧数据兼容**：用 `schema_version` 隔离，不强制迁移旧 EpisodeLog

---

## 8. 建议先做的“零风险准备”（不改主代码）

1. 新建分支：`feat/social-system-benchmark`
2. 准备 3 份 JSON 原型（不进主逻辑）：
   - `event_scripts/trust_economy_v1.json`
   - `scenarios/negotiation_v1.json`
   - `scenarios/commons_v1.json`
3. 补全一批 agent 的初始资源与风险偏好（脚本离线处理）
4. 先定义指标计算脚本（离线读取日志），再反推在线 evaluator

---

## 9. 你可以直接拍板的决策项

请先确认这 6 个点，我就能给你下一步的“任务分解清单（到文件级）”：

- 先做哪个场景（建议：投资与违约）
- 初始规模（建议：`n_agents=5`, `max_steps=50`）
- 动作集合第一版包含哪些（建议先 5 个）
- 是否第一版就接向量记忆（建议先不接）
- 评测先保留哪几个核心指标（建议 welfare/default/gini/consistency）
- 运行预算（每次实验允许多少 episode）

---

## 10. 结论

你的方案可以在 sotopia 上实现，且不需要推翻重写。  
最佳路径是：**在现有框架上叠加状态层与事件层**，先做单场景 PoC，再扩展到三场景统一 benchmark。  
数据上完全可以复用现有角色与关系数据，场景与事件通过新增字段和新表增量建设即可。

