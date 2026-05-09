# Setting 1 世界机制规格草案 v0

## 1. 基本对象

### 1.0 世界设置

Setting 1 是一个有明确时间期限的长期商业谈判世界。

整个交易窗口持续：

```text
d = 1, 2, ..., D
```

其中 `D` 是交易截止日。所有 agent 都知道：

- 当前是第几天 `d`。
- 交易总期限是 `D` 天。
- 如果在 `D` 天结束前没有形成有效协议，谈判自动失败或终止。
- 在 deadline 之后不能再签订新协议。

因此，agent 的决策需要考虑剩余时间：

```text
remaining_days = D - d + 1
```



V1 的约束来自：

- 当前 day 和剩余 deadline。
- 当前 session slot。
- 当前 active session participants。
- 当前 agent 的 action space。
- 当前合同、资源、事件、消息和历史状态。

agent 根据这些 observation 自主决定何时收集信息、何时提出合同、何时修改合同、何时接受或签署。

世界中可能存在外部冲击或外部事件。事件不是 agent 的行动，也不是谈判 phase；它是 控制器 对世界状态和可见信息的外生更新。事件机制在 [外部事件](#8-外部事件) 中详细定义。

从 agent 的角度看，整个世界的行动机制可以概括为：

```text
observe -> schedule -> join session -> communicate/act -> world update -> next slot/day
```

具体来说：

1. 在每个 session slot 的 scheduling window 中，agent 可以邀请其他 agent 开启一个 session，也可以接受或拒绝收到的 session invitation。
2. 控制器 根据 request 和 response 决定当前 slot 是否形成 session。
3. session 成立后，只有 session participants 能看到 session 内信息，也只有他们能在该 session 中发言或行动。
4. session 内 agent 可以发送 message，也可以在预算允许时执行 formal action，例如提出合同、修改合同、接受合同、签署合同、融资承诺或监管批准。
5. agent 也可以通过 session control action 离开当前 session。若剩余 participants 少于 2 人，控制器 结束该 session。
6. session 结束后，控制器 更新状态并进入下一个 session slot 或 end-of-day。
7. end-of-day 时，控制器 可以结算状态、保存 snapshot，并按配置触发外部事件。

整个世界在以下情况下终止：交易主体达成有效合同并完成必要签署，且在需要时满足融资承诺或监管批准；到达 deadline 仍未形成有效协议；交易主体执行 `terminate.negotiation` 且 控制器 判定整体谈判终止；或 控制器 判定不存在可行协议或继续谈判路径。终止机制在 [终止条件](#9-终止条件) 中详细定义。

关键机制分别在后续章节展开：

- agent 的 utility、初始禀赋和预算约束见 [参与者](#11-参与者)。
- day、session slot 和 end-of-day 的推进见 [时间结构](#2-时间结构)。
- session request、response 和 resolution 见 [Scheduling Window](#3-scheduling-window)。
- session 内信息、turn order 和离开规则见 [Session](#4-session)。
- 合同账本、合同字段、合同可见性和合同修改规则见 [Contract 世界状态](#5-contract-世界状态)。
- agent action space 和每个时点可用 action 集合见 [Action Space 与 Budget](#6-action-space-与-budget)。
- 不同 session participants 组合下的业务直觉见 [不同 Session 类型中的可行动作](#7-不同-session-类型中的可行动作)。
- 外部事件与 shock 机制见 [外部事件](#8-外部事件)。
- world-level 终止条件见 [终止条件](#9-终止条件)。

### 1.1 参与者

设参与者集合为：

```text
A = {firm_a, firm_b, investor, regulator}
```

其中：

- `firm_a`：买方或收购方。
- `firm_b`：卖方或目标公司。
- `investor`：融资方。
- `regulator`：监管方。

每个 agent `i in A` 有自己的 private objective、utility function、threshold、可见历史和私有记忆。agent 的目标不是服从脚本，而是在规则允许下最大化自己的目标函数。

每个 agent `i` 有一组初始禀赋：

```text
B_i(0) = {
  cash_i(0),
  asset_i(0),
  liability_i(0),
  reputation_i(0),
  private_information_i(0)
}
```

不同角色可以拥有不同的禀赋维度。例如 firm 和 investor 通常有 cash、asset、liability；regulator 不一定有 money，但可以有 policy constraint、institutional credibility 或 public mandate。

每个 agent 有自己的 utility function：

```text
u_i = U_i(C, omega_T, h_i)
```

其中：

- `C` 是最终合同；若没有达成合同，则为空。
- `omega_T` 是终止时的 world state。
- `h_i` 是 agent `i` 在整个过程中的可见历史和私有信息。

每个 agent 还有个体理性约束或最低可接受收益：

```text
u_i >= theta_i
```

其中 `theta_i` 是 agent `i` 的 private threshold。agent 不应公开泄露自己的 `U_i` 或 `theta_i`，但应基于它们决策。

预算约束可以写成：

```text
g_i(C, omega_t, B_i(t)) <= 0
```

含义是 agent 不能承诺超出自身资源、权限或约束的合同义务。例如：

- firm_a 不能承诺无法支付的现金结构。
- investor 不能承诺超过可部署资本或风险约束的融资。
- firm_b 不能承诺自己无法履行的交割或担保义务。
- regulator 不能批准违反 hard constraints 或 policy constraints 的合同。

控制器 在执行 formal action 时校验预算约束、角色权限和 hard constraints。违反约束的 action 可以被标记为 invalid，或在后续版本中产生惩罚。

buyer 是否需要 investor 不是固定设定，而由预算约束和合同价格决定。设合同 `C` 的支付义务为：

```text
price(C)
```

buyer 自有可用资金为：

```text
cash_firm_a(t)
```

若：

```text
cash_firm_a(t) >= price(C)
```

则 buyer 可以不依赖外部融资完成交易；investor 不是该合同的必要参与方。

若：

```text
cash_firm_a(t) < price(C)
```

则合同需要外部融资或付款结构调整。此时 investor 可以成为该合同的 contingent required party，只有当 investor 执行 `commit.finance_commit`，或合同被修改为不再需要外部融资时，交易才可能成功。

regulator 是否需要参与也不是固定设定，而由 regulatory requirement 决定。设：

```text
R(C, omega_t) in {0, 1}
```

表示合同 `C` 在当前 world state `omega_t` 下是否需要监管批准。`R(C, omega_t)` 可以由合同规模、合规条款、事件冲击、监管政策或 regulator 的 utility/threshold 变化决定。

若：

```text
R(C, omega_t) = 0
```

则 regulator 不是该合同的必要参与方。

若：

```text
R(C, omega_t) = 1
```

则 regulator 成为该合同的 contingent required party，合同成功需要 regulator 执行 `commit.approve`。外部事件可以把 `R(C, omega_t)` 从 0 改为 1，也可以改变 regulator 对合同的可接受性。

### 1.2 控制器

除 agent 外，环境中有一个 deterministic 控制器。它不是谈判参与者，而是规则执行者和状态记录者。

控制器 负责：

- 推进 day、schedule window、session、post-session、end-of-day。
- 收集 session request 和 response。
- 决定哪些 session 成立，以及 session 顺序。
- 记录 active session set 及每个 session 的 participants。
- 限制非 session participants 不能插话或被邀请。
- 校验 action schema、receiver、visibility 和 formal state mutation。
- 记录 message log、action log、session log、event log、state snapshot。
- 按 event trigger 在指定时点注入外部事件。

## 2. 时间结构

时间以 day 为单位推进：

```text
d = 1, 2, ..., D
```

每个 day `d` 由以下部分组成：

```text
Day d
  1. Session Slot 1
     1.1 Scheduling Window
     1.2 Session Execution, if one or more sessions are formed
     1.3 Post-session bookkeeping
  2. Session Slot 2
     2.1 Scheduling Window
     2.2 Session Execution, if one or more sessions are formed
     2.3 Post-session bookkeeping
  ...
  S. Session Slot S_max
  3. End-of-day update
```

其中 `S_max` 是每天最多 session slot 数，当前候选值是 3，但本文先保留为符号。day 内的 session slot 可以表示为：

```text
k = 1, 2, ..., S_max
```

每个 slot 是一个并行时间段。一个 slot 内可以形成多个互不重叠的 session：

```text
Sessions(d, k) = {s_1, s_2, ..., s_m}
```

但同一个 agent 在同一个 slot 中最多参加一个 session：

```text
for any s_a, s_b in Sessions(d, k), s_a != s_b:
  participants(s_a) ∩ participants(s_b) = empty
```

因此，`S_max` 更准确地表示每天最多有多少个正式会谈时间段，而不是每天最多只能发生多少个 session。若一个 agent 每个 slot 最多参加一个 session，则它每天最多参加 `S_max` 个 session。

如果某个 slot 的 request 没有被接受，则该 slot 对相关 agent 可以空过，但不能在同一个 slot 内反复邀请新对象。

### 2.1 Post-session Bookkeeping

Post-session bookkeeping 是 控制器-only 阶段。它不是 agent action phase，也不调用 agent 产生新的 message、formal action、session control action 或 private note。

这个限制很重要：V1 中所有主动沟通都必须发生在 active session 内。如果 post-session 允许 agent 再主动发言、补充承诺或写入可影响状态的 private reflection，它就会变成一个隐形 action phase，破坏 session 机制的边界。

当一个 session slot 内的所有 active sessions 都结束后，控制器 执行 post-session bookkeeping。它负责：

1. 关闭所有已结束 session，记录 `t_end` 和结束原因。
2. 把 session 内 message、formal action、session control action 写入 session log 和 action log。
3. 更新每个 agent 的可见历史：
   - session participants 收到自己参与期间可见的 session transcript。
   - 中途退出者只收到退出前的 session 内容。
   - 非 participants 不收到该 session 内容。
4. 更新合同状态、daily formal-action budget、session message budget、session-control 状态。
5. 生成给下一 slot 的 observation summary，例如刚刚参与了什么、还剩多少 budget、当前可见合同状态是什么。
6. 检查是否触发 world-level terminal condition。
7. 检查是否有 post-session 或 end-of-slot event 需要触发。

agent 的 reflection 或 memory update 在 V1 中只能作为 observation construction 的结果出现：控制器 可以把可见历史、预算、合同状态和事件摘要拼接进下一次 prompt，但 agent 不会在 post-session 单独输出可改变状态的 private note。

后续版本可以把 self-reflection 作为实验变量引入，例如：

```text
private_reflection_enabled in {true, false}
```

但 V1 默认：

```text
private_reflection_enabled = false
```

## 3. Scheduling Window

Scheduling window 是每个 session slot 开始时用于形成该 slot session set 的有限流程。它不是正式谈判 session。

Scheduling window 可以异步调用所有 agent：每个 agent 根据自己的当前 observation 生成 request 或 response，控制器 收集后统一 resolve。resolve 后，控制器 再把结果拼接回每个 agent 的 observation，告知它：

- 自己发起的 request 是否被接受或拒绝。
- 自己收到的 request 中哪些成立、哪些未成立。
- 接下来是否进入某个 session。
- 若进入 session，自己所在 session 的 participants 是谁。
- 若未进入 session，自己知道哪些相关结果。

每个 Scheduling Window 的内部流程为：

1. **Invitation Round**：各 agent 提交 session request。
2. **Response Round**：被邀请者对收到的 request 做 accept/decline。
3. **Resolution**：控制器 汇总 request 与 response，决定该 slot 形成哪些 session。

v0 暂定只有一轮 request + 一轮 response，不支持多轮协商式调度。

### 3.1 Invitation Round

在 invitation round 中，每个 agent `i` 可以提交至多 `Q_i` 个 session request。v0 暂定：

```text
Q_i = 1
```

即每个 agent 在一个 session slot 中最多发起一次邀请，不能同时发起多个候选 session。

```text
session_request r = {
  requester: i,
  proposed_participants: P_r,
  purpose: text
}
```

约束：

- `i in P_r`
- `P_r subset A`
- `|P_r| >= 2`
- `purpose` 只能描述会话目的，不能夹带实质合同报价或正式承诺。
- 同一个 slot 中，如果 request 被拒绝，requester 不能立刻改邀其他 agent。
- 后续 slot 中，requester 可以重新发起新的 request。

request 不使用 `public | private | semi_private` 这类 message visibility 字段。它的可见性由 控制器 的固定 scheduling visibility rule 决定，见 [Request Visibility](#32-request-visibility)。

例子：

```text
firm_a requests a session with firm_b to discuss valuation and payment.
investor requests a session with regulator to discuss financeability and approval risks.
```

### 3.2 Request Visibility

request 对 requester 和被邀请对象可见，对未被邀请对象不可见。这是 scheduling request 的固定可见性规则：

```text
visible_to_request(r) = {requester(r)} ∪ proposed_participants(r)
```

由于 requester 必须属于 `proposed_participants(r)`，也可以简写为：

```text
visible_to_request(r) = proposed_participants(r)
```

例如：

```text
firm_a invites firm_b and investor to form a session.
```

则：

- `firm_b` 知道 `firm_a` 邀请自己和 `investor`。
- `investor` 知道 `firm_a` 邀请自己和 `firm_b`。
- `regulator` 不知道这三方具体在计划讨论什么。
- `regulator` 最多只能在后续 observation 中知道自己未进入当前 session，或某些 agent 当前不可用。

因此 控制器 必须区分 private observation 和 shared observation，不能把所有 scheduling 信息广播给所有 agent。

非被邀请者在 v0 中不会看到：

- request 是否存在。
- requester 是谁。
- proposed participants 是谁。
- purpose 是什么。

如果当前 slot 最终形成了一个不包含某 agent 的 session，该 agent 最多知道自己未进入该 slot 的 session，或者知道当前 slot 对自己不可用；它不自动知道谁和谁正在 session 中。

### 3.3 Response Round

控制器 将每个 request 通知给被邀请者。每个被邀请 agent 对每个 request 返回：

```text
session_response = accept | decline
```

发起人默认接受自己的 request。

如果 `firm_a` 邀请 `firm_b`，而 `firm_b` decline，则该 request 不成立。`firm_b` 可以同时接受或拒绝其他 request，但不能在 response round 中重新发起新的 request，除非配置允许额外的 scheduling round。

v0 进一步约束：

```text
每个 agent 在一个 session slot 中最多接受一个 session invitation。
```

这里的 invitation 指其他 agent 发给自己的 request，不包括自己发起 request 时的默认接受。如果 `firm_b` 同时收到多个邀请，它只能接受其中一个，其他必须 decline 或保持不接受。这个规则避免同一个 agent 在同一个 session slot 中被多个 session 同时占用。

response round 的有效 response 集合为：

```text
response_i(r) in {accept, decline}
```

如果 agent 未响应、输出无效 response、或 response 无法被 控制器 解析，则记为：

```text
raw_response_i(r) = no_response
effective_response_i(r) = decline
```

控制器 不等待、不重试、不追加协商轮。`no_response` 会作为工具调用失误或无效调度行为写入 log / trajectory，但在 session resolution 中按 decline 处理。

如果某个 agent 在同一个 session slot 中 accept 多个 request，则该 agent 在该 slot 的所有 accept 都无效：

```text
if count_accept_i(k) > 1:
  effective_response_i(r) = decline for all r received by i in slot k
```

该情况也写入 log / trajectory，作为违反 response 约束的工具调用失误。

### 3.4 Session Resolution

控制器 根据 request 和 response 生成当前 session slot 的 session set。

一个 request 成立的必要条件：

```text
all agents in proposed_participants accept
```

设所有成立 request 的集合为：

```text
R_valid(d, k) = {r | all agents in proposed_participants(r) accept r}
```

一个 session slot 可以创建多个 session，但这些 session 的 participants 必须两两不重叠。控制器 使用 deterministic resolver 将 `R_valid(d, k)` 转换为 `Sessions(d, k)`。

v0 resolver 规则如下：

1. **相同 participants 合并**：若多个 accepted request 的 `proposed_participants` 完全相同，则合并为一个 session。
2. **包含关系取大**：若两个 accepted request 有重叠，且一个 participant set 严格包含另一个，则保留更大的 participant set，较小 request 被视为被更完整的 session 吸收。
3. **复杂交叉冲突失败**：若两个 accepted request 有重叠，但不存在包含关系，则这些冲突 request 在 v0 中全部失败，并写入 log / trajectory。
4. **互不重叠并行成立**：经过上述处理后，所有 participants 互不重叠的 request 都可以在同一个 slot 中并行形成 session。

形式化地说，最终 session set 必须满足：

```text
for any s_a, s_b in Sessions(d, k), s_a != s_b:
  P_{s_a} ∩ P_{s_b} = empty
```

例子：

```text
firm_a invites firm_b
investor invites regulator
```

若两个 request 都成立，则同一 slot 中形成两个并行 session：

```text
Sessions(d, k) = {
  {firm_a, firm_b},
  {investor, regulator}
}
```

双向邀请会被合并。例如：

```text
firm_a invites firm_b
firm_b invites firm_a
```

若两个 request 都成立，则只形成一个 session：

```text
Sessions(d, k) = {
  {firm_a, firm_b}
}
```

若小 session 和大 session 竞争，例如：

```text
firm_a invites firm_b
firm_b invites firm_a, investor
```

且两个 request 都成立，则保留更大的三方 session：

```text
Sessions(d, k) = {
  {firm_a, firm_b, investor}
}
```

若出现无法通过包含关系解决的交叉冲突，例如：

```text
firm_a invites firm_b, investor
firm_b invites firm_a, regulator
```

且两个 request 都成立，则两个冲突 request 在 v0 中都失败。控制器 不替 agent 主观选择其中一组。

## 4. Session

### 4.1 Session 定义

一个 session 为：

```text
session s = {
  session_id,
  day: d,
  participants: P_s,
  t_start,
  t_end,
  status
}
```

约束：

- `P_s` 在 session 开始时固定。
- session 内不可加入新参与者。
- session 内不可邀请 session 外 agent 加入当前 session。
- 同一个 session slot 内可以有多个并行 active sessions。
- 同一个 agent 在同一个 session slot 内最多属于一个 active session。
- 同一 slot 的所有 active sessions 都结束后，控制器 才进入 post-session bookkeeping 和下一个 slot。

### 4.2 非参与者

对于任一 active session，如果其 participants 为 `{firm_a, firm_b}`，则不在该 session 内的 `investor` 和 `regulator`：

- 不会被调用做 session action。
- 不能给 session 内 agent 发消息。
- 不能邀请 session 内 agent 开新 session。
- 不消耗 session 内 action budget。

他们不能插入该 session；若同一 slot 内没有属于自己的并行 session，则只能等待后续 slot 或 next day 的 scheduling window。

session 内信息默认只对 session participants 可见。多个 agent 参与同一个 session 时，v0 不再细分 session 内 public/private；session 内容对所有当前 participants 共享，对非 participants 不可见。

如果某个 participant 中途退出 session，则退出之后发生的内容对它不可见。

### 4.3 Session 内 Turn

每个 session 内同时有两个 turn 上限：

```text
N_s <= T_s
K_i(s) <= K_s
```

其中：

- `N_s` 是 session `s` 已经执行的总 turn 数。
- `T_s` 是 session `s` 允许的最大总 turn 数。
- `K_i(s)` 是 agent `i` 在 session `s` 中已经执行的 turn 数。
- `K_s` 是每个 participant 在 session `s` 中允许的最大 turn 数。

两个限制同时生效。只要任一 session 结束条件被触发，控制器 就结束该 session。

一次 turn 中，active agent 可以输出：

```text
turn_output = {
  message: optional message,
  formal_action: optional formal action,
  session_control_action: optional session control action
}
```

三类行为独立，互不占用对方的 budget：

- **message**：沟通、解释、询问、披露信息、说服对方。
- **formal_action**：改变世界状态或合同状态的动作（如 `contract.propose`）。
- **session_control_action**：控制 session 本身（如 `leave_session`、`terminate.session`）。

约束：

- v0 中 session 内 message 默认对当前所有 participants 可见。
- message receiver 可记为 `ALL_IN_SESSION`，或省略后由 控制器 自动解释为当前 participants。
- session 内暂不支持只发给部分 participants 的 private message。
- `formal_action` 至多一个。
- `formal_action` 必须符合当前 agent、当前 session 和当前 world state 的可用 schema。
- `session_control_action` 至多一个，不计入 `formal_action` 上限。
- 一个 turn 中可以同时输出 message + formal_action + session_control_action，三者互不冲突。

turn 消耗规则：

- active agent 被调用并返回一次 `turn_output`，就消耗一个 turn。
- 只发送 message、不执行 formal action，也消耗一个 turn。
- 只执行 formal action、不发送 message，也消耗一个 turn。
- 只执行 session control action，也消耗一个 turn。
- 什么都不做是允许的，记为 `pass`，也消耗一个 turn。
- 如果 formal action invalid，该 turn 仍然被消耗；控制器 只是不应用 invalid formal state mutation。

`pass` 可以表示为：

```text
turn_output = {
  message: null,
  formal_action: null,
  session_control_action: null
}
```

message 只能在 session 内发送。session 外不允许 agent 主动发送 message。

session 内 agent 发言是串行的。v0 候选规则：

```text
1. 每个 session 的第一个行动者按 deterministic participant order 决定。
2. 之后按同一个 deterministic participant order 轮流行动。
3. 如果有 agent 退出，则从后续 turn order 中移除。
4. 如果某个 agent 已达到 `K_i(s) = K_s`，则跳过该 agent。
```

v0 暂时使用固定 role order：

```text
firm_a -> firm_b -> investor -> regulator
```

如果 session participants 不包含 role order 中靠前的 agent，则从该 session 中实际存在的最靠前 agent 开始。例如 `{firm_b, investor}` 的第一发言人是 `firm_b`。

这样可以处理双向 invitation 或相同 participants request 合并后的发言顺序问题：合并后的 session 不再依赖 requester 决定谁先说，而是由固定 role order 决定。

这一点仍需继续讨论：多方 session 中是否应该严格串行，还是允许一轮中多个 agent 基于同一 observation 并行 response。

session 自动结束条件：

```text
N_s >= T_s
or all active participants satisfy K_i(s) >= K_s
or |P_s| < 2
or world-level terminal_state is reached
```

达到 `T_s` 或 `K_s` 上限时，session 是自然结束，不是惩罚，也不是 invalid action。

#### 4.3.1 Turn 排列示例

**两人 session：`{firm_a, firm_b}`，`T_s = 6`，`K_s = 3`**

```text
turn 1: firm_a   — message + contract.propose   (消耗 1 turn)
turn 2: firm_b   — message                      (消耗 1 turn)
turn 3: firm_a   — message                      (消耗 1 turn)
turn 4: firm_b   — message + contract.accept    (消耗 1 turn)
turn 5: firm_a   — message + commit.sign        (消耗 1 turn)
turn 6: firm_b   — message + commit.sign        (消耗 1 turn)
```

总 turn 数达到 `T_s = 6`，双方各用满 `K_s = 3`，session 自然结束。

**两人 session 提前结束：`{firm_a, firm_b}`，`T_s = 6`，`K_s = 3`**

```text
turn 1: firm_a — message + contract.propose
turn 2: firm_b — terminate.session              (firm_b 退出)
```

`firm_b` 退出后 `|P_s| = 1 < 2`，控制器 结束 session。总 turn 数为 2，未达 `T_s`，但 session 因 participants 不足而结束。`firm_b` 退出后看不到任何后续内容（此处无后续）。

**三人 session：`{firm_a, firm_b, investor}`，`T_s = 6`，`K_s = 3`**

```text
turn 1: firm_a   — message                      (消耗 1 turn)
turn 2: firm_b   — message                      (消耗 1 turn)
turn 3: investor — message                      (消耗 1 turn)
turn 4: firm_a   — message                      (消耗 1 turn)
turn 5: firm_b   — message                      (消耗 1 turn)
turn 6: investor — message                      (消耗 1 turn)
```

总 turn 数达到 `T_s = 6`，session 自然结束。此时每人只用了 `K_s = 2`，未用满个人上限，但 session 总上限已到。

**三人 session 中途退出：`{firm_a, firm_b, investor}`，`T_s = 6`，`K_s = 3`**

```text
turn 1: firm_a   — message + contract.propose
turn 2: firm_b   — message
turn 3: investor — terminate.session            (investor 退出)
turn 4: firm_a   — message
turn 5: firm_b   — message + contract.accept
turn 6: firm_a   — message + commit.sign        (已无 investor，无需 finance_commit)
```

investor 在 turn 3 退出后，后续 turn 顺序跳过 investor，由 firm_a 和 firm_b 继续轮流。investor 看不到 turn 4-6 的任何内容。若合同不需要融资，交易仍可由 firm_a 和 firm_b 直接完成。

### 4.4 离开 Session 与 Session 自然结束

agent 可以在 session 内通过 session control action 离开当前 session：

```text
leave_session
terminate.session
```

在 v0 中，`terminate.session` 的语义不是强制关闭整个 session，而是表示发出该 action 的 agent 自己离开当前 session。它等价于更显式的 `leave_session`。

当一个 agent 选择 `leave_session` 或 `terminate.session`：

- 它从当前 session participants 中移除。
- 它不再收到该 session 后续 message 或 action observation。
- 它不能在该 session 内再次发言或行动。

如果 session 中剩余 participants 少于 2 人，则 控制器 结束该 session。

如果三人 session 中一人离开，剩下两人可以继续 session；离开者听不到剩余两人的后续交流。

## 5. Contract 世界状态

合同是 formal action 的核心操作对象。控制器 维护一个全局合同账本：

```text
ContractBook_t = {C_1, C_2, ..., C_n}
```

每个合同 `C` 至少包含：

```text
C = {
  contract_id,
  parent_contract_id,
  created_by,
  created_at,
  terms,
  parties,
  status,
  acceptances,
  signatures,
  financing,
  regulatory,
  visibility_set,
  history
}
```

其中：

```text
created_at = {
  day,
  slot_id,
  session_id,
  turn_id
}
```

在 Setting 1 中，合同条款 `terms` 可以写成：

```text
terms = {
  valuation,
  payment,
  closing,
  compliance,
  penalty
}
```

`parties` 至少包含交易主体：

```text
parties = {firm_a, firm_b}
```

如果合同需要融资或监管批准，investor / regulator 不一定成为合同交易主体，但会成为 contingent required parties。

合同状态可以取：

```text
status in {
  proposed,
  amended,
  accepted,
  rejected,
  signed,
  superseded,
  failed
}
```

交易主体接受状态为：

```text
acceptances = {
  firm_a: true | false | null,
  firm_b: true | false | null
}
```

签署状态为：

```text
signatures = {
  firm_a: true | false,
  firm_b: true | false
}
```

融资状态为：

```text
financing = {
  required: 0 | 1,
  status: not_required | pending | committed | declined,
  actor: investor optional
}
```

监管状态为：

```text
regulatory = {
  required: 0 | 1,
  status: not_required | pending | approved | blocked,
  actor: regulator optional
}
```

合同可见性由 `visibility_set` 决定：

```text
visible_to(C, i, t) = true iff i in visibility_set(C, t)
```

默认情况下，合同只对创建它的 session participants 可见：

```text
visibility_set(C, created_at) = P_s
```

session 结束后，合同不会自动公开，也不会自动对非 participants 可见。如果 `firm_a` 和 `firm_b` 在二人 session 中提出或修改了合同 `C`，则 session 结束后：

```text
visibility_set(C) = {firm_a, firm_b}
```

`investor` 或 `regulator` 在后续 session 中默认看不到该合同，也不能引用该 `contract_id`。只有能看到合同的 agent 才能引用它：

```text
i can reference C iff i in visibility_set(C, t)
```

如果后续执行 review 或 share 类 formal action，控制器 可以把新的 agent 加入 `visibility_set`。例如：

```text
review.request_financing(contract_id)
  => investor is added to visibility_set(C)

review.request_regulatory(contract_id)
  => regulator is added to visibility_set(C)

contract.share(contract_id, receiver)
  => receiver is added to visibility_set(C)
```

这些 action 必须发生在 active session 内，并且 `receiver` 必须是当前 session participant。控制器 还必须校验 action actor 本身已经能看到该合同。

已签署合同也不会自动变成 public。V1 中，signed contract 仍只对 signatories、required parties 和已经在 `visibility_set` 中的 agent 可见，除非外部事件、监管规则或后续 disclosure action 要求公开。

合同修改不原地覆盖旧合同，而是创建新合同：

```text
contract.amend(C)
  => create C'
  => C'.parent_contract_id = C.contract_id
  => C.status = superseded
```

这样 控制器 可以保留完整合同历史，后续评估也能追踪 agent 是否记住旧合同、是否合理修改合同。

## 6. Action Space 与 Budget

### 6.1 Action Space 总体定义

设完整 action space 为：

```text
X = X_sched ∪ X_msg ∪ X_formal ∪ X_ctrl
```

其中：

- `X_sched`：scheduling action，用于形成 session。
- `X_msg`：message action，用于 session 内沟通。
- `X_formal`：formal action，用于改变合同状态或世界状态。
- `X_ctrl`：session control action，用于控制 session 本身。

一次 agent 输出可以表示为：

```text
x_i(t) = {
  action_class,
  action_type,
  receiver,
  content,
  meta
}
```

其中：

- `action_class in {sched, msg, formal, ctrl}`
- `action_type` 是该 action class 下的有限枚举。
- `receiver subset A`，但 session 内 action 还必须满足 `receiver subset P_s`。
- `content` 是该 action 的主要业务内容。
- `meta` 是 routing、visibility、object reference、audit flag 等控制信息。

### 6.2 Scheduling Action Space

Scheduling action space 定义为：

```text
X_sched = X_request ∪ X_response
```

包括：

- `session_request`
- `session_response`

它们只用于形成 session，不直接改变合同状态，不计入 daily formal-action budget。

#### 6.2.1 Session Request

```text
x in X_request:
  action_class = sched
  action_type = session_request
  receiver = P_r \ {i}
  content = {
    proposed_participants: P_r,
    purpose: text
  }
  meta = {
    slot_id,
    visibility
  }
```

约束：

```text
i in P_r
P_r subset A
|P_r| >= 2
|{x in X_request by i in slot k}| <= Q_i
```

v0 中：

```text
Q_i = 1
```

#### 6.2.2 Session Response

```text
x in X_response:
  action_class = sched
  action_type = session_response
  receiver = {requester(r)}
  content = {
    request_id,
    decision: accept | decline
  }
  meta = {
    slot_id
  }
```

约束：

```text
decision in {accept, decline}
accepted_requests_i(k) <= 1
```

其中 `accepted_requests_i(k)` 表示 agent `i` 在 session slot `k` 接受的 request 数量。

### 6.3 Message Action Space

Message action space 定义为：

```text
X_msg = {message}
```

message 用于沟通、解释、询问、披露信息、说服对方。

message 只能发生在 session 内。第一版中，session 外不允许 agent 主动发送 message。

```text
x in X_msg:
  action_class = msg
  action_type = message
  receiver = P_s \ {i} or ALL_IN_SESSION
  content = text
  meta = {
    session_id,
    intent
  }
```

约束：

```text
i in P_s
receiver subset P_s
x can only be emitted while session s is active
```

v0 中 session 内 message 默认对当前所有 participants 可见，不再细分 session 内 public/private。因此 `receiver` 可以写作：

```text
receiver = ALL_IN_SESSION
```

可以为每个 session 设置：

```text
M_i(s) <= M_max
```

表示 agent `i` 在 session `s` 中最多发送 `M_max` 条 message。

message 不计入 daily formal-action budget，但受 session turn/message budget 限制。

### 6.4 Formal Action Space

Formal action space 定义为：

```text
X_formal =
  X_contract ∪ X_commit ∪ X_review ∪ X_terminate_negotiation
```

formal action 是会改变世界状态或合同状态的动作，例如：

- `contract.propose`
- `contract.amend`
- `contract.share`
- `contract.accept`
- `contract.reject`
- `commit.sign`
- `commit.finance_commit`
- `commit.finance_decline`
- `commit.approve`
- `commit.block`
- `terminate.negotiation`

`terminate.session` 不属于 `X_formal`，也不消耗 daily formal-action budget `F_max`。它属于 [Session Control Action Space](#65-session-control-action-space)，只能在 active session 内由当前 session participant 调用。

通用形式为：

```text
x in X_formal:
  action_class = formal
  action_type in T_formal
  receiver subset P_s
  content = formal payload
  meta = {
    session_id,
    target_object_id optional
  }
```

其中：

```text
T_formal = {
  contract.propose,
  contract.amend,
  contract.share,
  contract.accept,
  contract.reject,
  review.request_financing,
  review.request_regulatory,
  commit.sign,
  commit.finance_commit,
  commit.finance_decline,
  commit.approve,
  commit.block,
  terminate.negotiation
}
```

#### 6.4.1 Contract Actions

```text
X_contract = {
  contract.propose,
  contract.amend,
  contract.share,
  contract.accept,
  contract.reject
}
```

`contract.propose`:

```text
content = {
  contract: C
}
meta = {
  session_id
}
```

`contract.amend`:

```text
content = {
  contract_id,
  amended_contract: C'
}
meta = {
  session_id
}
```

`contract.accept` and `contract.reject`:

```text
content = {
  contract_id
}
meta = {
  session_id
}
```

`contract.share`:

```text
content = {
  contract_id,
  receiver
}
meta = {
  session_id
}
```

`contract.share` 用于把一个当前可见合同显式披露给当前 session 中的其他 participant。控制器 必须校验：

```text
actor in visibility_set(C, t)
receiver subset P_s
receiver not empty
```

执行成功后：

```text
visibility_set(C, t + 1) = visibility_set(C, t) ∪ receiver
```

合同对象 `C` 的完整字段、可见性和修改规则见 [Contract 世界状态](#5-contract-世界状态)。

#### 6.4.2 Review Actions

```text
X_review = {
  review.request_financing,
  review.request_regulatory
}
```

```text
content = {
  contract_id
}
meta = {
  session_id
}
```

这些 action 用于请求 investor 或 regulator 对已有合同进行融资或监管审查。执行成功时，控制器 会把对应 reviewer 加入该合同的 `visibility_set`，使其能够读取合同并在后续执行 `commit.finance_commit`、`commit.finance_decline`、`commit.approve` 或 `commit.block`。

#### 6.4.3 Commit Actions

```text
X_commit = {
  commit.sign,
  commit.finance_commit,
  commit.finance_decline,
  commit.approve,
  commit.block
}
```

`commit.sign`:

```text
content = {
  contract_id
}
meta = {
  session_id,
  final: true
}
```

`commit.finance_commit` and `commit.finance_decline`:

```text
content = {
  contract_id,
  reason optional
}
meta = {
  session_id
}
```

`commit.approve` and `commit.block`:

```text
content = {
  contract_id,
  reason optional
}
meta = {
  session_id
}
```

#### 6.4.4 Negotiation Termination

```text
X_terminate_negotiation = {
  terminate.negotiation
}
```

```text
content = {
  reason
}
meta = {
  session_id,
  final: true
}
```

`terminate.negotiation` 表示 agent 退出整个交易谈判。控制器 根据执行者角色决定其后果：

```text
if actor in {firm_a, firm_b}:
  terminal_state = terminated_by_agent
else if actor = investor:
  mark investor as withdrawn from financing path
else if actor = regulator:
  mark regulator as withdrawn from regulatory review path
```

因此，investor 或 regulator 的 `terminate.negotiation` 不必然终止整个世界，但会影响当前合同路径是否还能满足 contingent required conditions。

每日 formal-action budget 已在世界设置中定义为：

```text
F_i(d) <= F_max
```

formal action 发生在 active session 内。session 结束后，agent 不再主动执行 formal action，除非后续明确引入 post-session action。

V1 同时使用 daily formal-action budget 和 per-session-per-agent formal-action budget：

```text
F_i(d) <= F_max
H_i(s) <= H_max
```

其中：

- `F_i(d)` 是 agent `i` 在 day `d` 已执行的 formal action 数。
- `H_i(s)` 是 agent `i` 在 session `s` 中已执行的 formal action 数。
- `F_max` 控制一天内的正式行动总额度。
- `H_max` 控制同一个 session 内单个 agent 不能无限执行 formal action。

某个 agent 在 session `s` 中还能执行的 formal action 数为：

```text
available_formal_actions_i(d, s)
  = min(F_max - F_i(d), H_max - H_i(s))
```

若 `available_formal_actions_i(d, s) <= 0`，则 agent `i` 在该 session 中不能再执行 formal action，但仍可在 turn/message budget 允许时发送 message 或执行 session control action。

### 6.5 Session Control Action Space

Session control action space 定义为：

```text
X_ctrl = {
  leave_session,
  terminate.session
}
```

session control action 用于控制 session 本身，例如：

- `leave_session`
- `terminate.session`

v0 中，`terminate.session` 表示发出该 action 的 agent 退出当前 session；它不会自动强制其他 participants 离开。若退出后剩余 participants 少于 2 人，控制器 才结束该 session。

`leave_session`:

```text
action_class = ctrl
action_type = leave_session
receiver = WORLD
content = {
  session_id
}
meta = {
  reason optional
}
```

`terminate.session`:

```text
action_class = ctrl
action_type = terminate.session
receiver = WORLD
content = {
  session_id,
  reason
}
meta = {}
```

这些 action 不改变合同状态，但会改变 session participants。它们不计入 `X_formal`，也不计入 formal-action budget。v0 倾向于把它们单独记录为 session-control budget。

Session control action 只能在 active session 内调用。session 已经结束后，任何 agent 都不能再调用 `leave_session` 或 `terminate.session`；它们也不能在 scheduling window 或 end-of-day 阶段调用。

### 6.6 Agent-Specific Available Action Set

完整 action space `X` 是所有可能动作的全集。某个 agent 在具体时点真正可用的动作是一个子集：

```text
X_i(d, k, s, omega) subset X
```

其中：

- `i` 是 agent。
- `d` 是当前 day。
- `k` 是当前 session slot。
- `s` 是当前 active session，若不在 session 内则为空。
- `omega` 是当前 world state，包括合同状态、资源状态、event log、session log 和 agent budget。

可用性由以下约束共同决定：

```text
X_i(d, k, s, omega)
  = RoleAllowed_i
    ∩ TimeAllowed(d, k)
    ∩ SessionAllowed_i(s)
    ∩ BudgetAllowed_i(d, s)
    ∩ StateAllowed(omega)
```

解释：

- `RoleAllowed_i`：角色权限，例如 regulator 可以 approve/block，但不能代表 firm 签署合同。
- `TimeAllowed(d, k)`：当前处于 scheduling window 还是 active session。scheduling action 只能在 scheduling window 中发生，message/formal/control action 只能在 active session 中发生。
- `SessionAllowed_i(s)`：agent 必须是当前 session participant 才能在 session 内发言或执行 action。
- `BudgetAllowed_i(d, s)`：agent 的 daily formal-action budget `F_i(d)`、per-session formal-action budget `H_i(s)`、session message budget、session-control budget 尚未耗尽。
- `StateAllowed(omega)`：动作必须引用存在、可见且可操作的对象，例如只能 amend 自己可见的已存在合同，只能 sign 可签署合同，只能 approve/block 自己可见且需要审查的合同。

这意味着 V1 不依赖 phase 来裁剪 action space，而是由 role、time、session、budget 和 world state 决定每个 agent 当前可以做什么。

## 7. 不同 Session 类型中的可行动作

不同 participants 组合可以拥有不同的 action schema。下面先记录业务直觉，后续再转成 JSON schema。

### 7.1 firm_a 与 firm_b

可能发生：

- 交换估值、付款、交割、合规、违约责任相关信息。
- 提出合同。
- 修改合同。
- 接受或拒绝合同。
- 签署合同，若处于 finalization 且条件满足。

### 7.2 firm_a 与 investor

可能发生：

- 讨论融资条件和风险。
- 请求融资 review。
- investor 表达融资偏好。
- investor 对已有合同进行 finance_commit 或 finance_decline。

### 7.3 firm_b 与 investor

可能发生：

- 讨论付款结构、风险保护、卖方回购或担保安排。
- investor 说明哪些条款会影响融资。
- 对已有合同提出融资相关修改建议。

### 7.4 firm_a 或 firm_b 与 regulator

可能发生：

- 讨论合规风险、披露要求、审批条件。
- 请求 regulatory review。
- regulator 对已有合同 approve 或 block。

### 7.5 investor 与 regulator

可能发生：

- 讨论融资可行性与监管可接受性之间的冲突。
- 澄清某些合同条款是否同时满足融资和监管条件。
- 通常不直接提出买卖双方合同，但可提出约束性意见或审查结论。

### 7.6 全体 Session

可能发生：

- 汇总多方约束。
- 提出或修改多方都可见的合同。
- 多方接受、融资承诺、监管审批、签署。
- 结束谈判。

## 8. 外部事件

外部事件不是 phase。它是 控制器 按 trigger 注入的状态变化或信息披露。

事件可以：

- 改变 market state。
- 改变 hard constraints。
- 改变 agent resources。
- 产生 public、private 或 semi-private information。
- 使旧合同、旧承诺或旧策略变得不再合适。

agent 不应预先知道未来事件会发生。agent 只能根据当前 observation 中已经可见的信息做决策。

### 8.1 Event Schema

外部事件由 控制器 执行，而不是由 agent 发起。事件对象可以表示为：

```text
event e = {
  event_id,
  event_type,
  trigger,
  visibility_set,
  payload,
  state_effect,
  observation_text
}
```

字段含义：

- `event_id`：事件唯一标识。
- `event_type`：事件类型，例如 `market_update`、`financial_disclosure`、`regulatory_change`、`financing_risk_update`。
- `trigger`：事件触发条件。
- `visibility_set subset A`：哪些 agent 能观察到该事件。
- `payload`：事件携带的结构化信息。
- `state_effect`：控制器 应用到 `omega_t` 的状态更新。
- `observation_text`：注入给可见 agent 的自然语言描述。

事件同时有两个层次：

- **机制层**：控制器 根据 `state_effect` 真实改变 world state `omega_t`。
- **观察层**：agent 只看到自己有权限观察的 `observation_text` 和可见状态变化。

agent 不需要自己推断事件是否改变了 world state；控制器 必须直接执行 `state_effect`。

### 8.2 Trigger

事件触发器可以抽象为：

```text
trigger = {
  mode: scheduled | condition_based | stochastic
}
```

V1 默认只实现：

```text
mode in {scheduled, condition_based}
```

`stochastic` 先不启用，除非后续明确要引入随机环境。

scheduled event 表示事件在预设时间触发：

```text
trigger = {
  mode: scheduled,
  day: d,
  slot_id: optional,
  timing: start_of_day | before_scheduling | after_session | end_of_slot | end_of_day
}
```

condition-based event 表示事件在某个 world state 条件满足时触发：

```text
trigger = {
  mode: condition_based,
  condition: predicate_name,
  timing: after_formal_action | after_session | end_of_slot | end_of_day
}
```

其中 `predicate_name` 是由 控制器 实现的确定性谓词，例如：

```text
contract_valuation_above_threshold
regulatory_approval_becomes_required
buyer_cash_below_required_payment
financing_risk_above_threshold
```

控制器 在对应 `timing` 检查 trigger。若 trigger 满足，则事件被触发并写入 event log。

### 8.3 State Effect

`state_effect` 是事件对世界状态的真实更新。V1 中可以先支持以下类型：

```text
state_effect = {
  update_market_state optional,
  update_agent_budget optional,
  update_regulatory_requirement optional,
  update_contract_feasibility optional,
  add_private_information optional
}
```

例子：

```text
update_market_state:
  market_condition = tighter_credit

update_agent_budget:
  cash_firm_a(t) = cash_firm_a(t) - shock_amount

update_regulatory_requirement:
  R(C, omega_t) = 1

update_contract_feasibility:
  feasible(C, omega_t) = false

add_private_information:
  private_information_firm_b += disclosed_liability
```

如果事件改变公共 world state，例如公开监管规则变化，则所有 agent 后续可见的公共状态也随之改变。如果事件只改变某个 agent 的 private information，则只有该 agent 的 observation 会显示该信息。

### 8.4 Event Observation

事件对 agent 的可见性由 `visibility_set` 决定：

```text
visible_to_event(e) = visibility_set(e)
```

如果：

```text
i in visibility_set(e)
```

则 agent `i` 下一次被调用时，observation 中出现：

```text
External Event Observed:
- event_id
- event_type
- description visible to you
- affected contract/resource/rule if disclosed
```

如果：

```text
i notin visibility_set(e)
```

则 agent `i` 不知道该事件发生，也看不到该事件的 `observation_text`。但是，如果事件改变了公共 world state，agent 可能在后续公共状态中看到更新后的结果，而不是看到事件本身的私有描述。

### 8.5 Event Timing and Logs

控制器 可以在以下时点检查并触发事件：

- `start_of_day`
- `before_scheduling`
- `after_formal_action`
- `after_session`
- `end_of_slot`
- `end_of_day`

每个触发事件都写入 event log：

```text
event_log_entry = {
  event_id,
  triggered_at,
  visible_to,
  state_effect_applied,
  observation_text_by_agent
}
```

其中 `observation_text_by_agent` 可以因 agent 可见性不同而不同。未进入 `visibility_set` 的 agent 不应收到该字段。

## 9. 终止条件

本节专门定义 world-level termination。它不同于 session termination：session termination 只结束当前 session，world-level termination 会结束整个谈判世界。

整个世界的终止状态记为：

```text
terminal_state in {success, failure, terminated_by_agent, timeout}
```

成功终止的核心条件是：

```text
当合同 C 被交易主体接受并签署，且所有 contingent required parties 的必要条件都被满足时，游戏结束。
```

具体到 Setting 1，可以写成：

```text
firm_a signs
firm_b signs
if FinancingRequired(C, omega_t) = 1:
  investor finance_commit
if RegulatoryApprovalRequired(C, omega_t) = 1:
  regulator approve
```

其中：

```text
FinancingRequired(C, omega_t) = 1
  iff cash_firm_a(t) < price(C)
  or C explicitly contains financing-contingent terms

RegulatoryApprovalRequired(C, omega_t) = R(C, omega_t)
```

因此，investor 和 regulator 不一定天然是每个合同的必要参与者。它们是否成为 required parties，取决于 buyer 的预算约束、合同条款、监管规则和外部事件后的 world state。

失败或非成功终止包括：

```text
timeout:
  d > D，且在 deadline 前没有形成有效协议。

terminated_by_agent:
  firm_a 或 firm_b 执行 terminate.negotiation，控制器 判定交易主体退出，谈判整体终止。

failure:
  控制器 判定长期没有任何可行 session、合同进展或可行协议。
```

`terminate.negotiation` 的含义是退出整个交易谈判，而不是退出当前 session。V1 中，只有交易主体 `firm_a` 或 `firm_b` 的 `terminate.negotiation` 会直接触发 world-level termination，因为没有 buyer 或 seller，交易本身无法成立。

如果 investor 执行 `terminate.negotiation`，控制器 将 investor 标记为退出融资谈判。若当前合同需要融资且没有替代融资路径，则该合同路径失败；但 buyer 和 seller 可以继续谈判，尝试降低价格、调整付款结构，或在扩展版本中寻找替代 investor。

如果 regulator 执行 `terminate.negotiation`，控制器 将 regulator 标记为退出当前监管沟通或拒绝继续参与审查。若当前合同需要监管批准且没有替代审批路径，则该合同路径失败；但 buyer 和 seller 可以继续谈判，尝试修改合规条款，使合同不再触发监管批准或重新满足监管要求。

进入任一 world-level terminal state 后，系统不再开启新的 scheduling window 或 session，也不再允许 message、formal action 或新合同签署。

## 10. Agent State Variables

每个 agent 有一组可配置状态变量。具体取值先不讨论，只记录变量类型。

### 10.1 firm_a

可能包括：

- utility function
- threshold
- money / cash
- assets or expected acquisition value
- liabilities
- reputation
- private information
- memory

### 10.2 firm_b

可能包括：

- utility function
- threshold
- money / cash
- asset value
- liabilities
- reputation
- private information
- memory

### 10.3 investor

可能包括：

- utility function
- threshold
- money / deployable capital
- risk exposure
- reputation
- private information
- memory

### 10.4 regulator

可能包括：

- utility function
- approval threshold
- public mandate or compliance objective
- policy constraints
- reputation or institutional credibility
- private information
- memory

regulator 不一定需要 money，但需要有自己的 objective 和 constraints。

## 11. 尚未拍板的问题

以下参数和规则需要后续继续讨论：

- `S_max`：每天最多 session slot 数；由于同一 slot 可有多个互不重叠 session，它不再表示每天最多 session 总数。
- `Q_i`：每个 agent 每个 session slot 最多可发起多少 session request。v0 暂定为 1。
- scheduling window 是否允许多轮 request/response，还是固定一轮。v0 暂定固定一轮。
- `T_s` 和 `K_s` 的具体取值。
- `F_max`：每个 agent 每天最多几个 formal action。
- `M_max`：每个 agent 每个 session 最多几条 message。
- `H_max`：每个 agent 在每个 session 中最多几个 formal action。
- message 是否完全免费，还是也有每日上限。
- 后续是否引入 `private_reflection_enabled = true` 作为实验变量；V1 默认 post-session 为 world-model-only，不允许 agent 主动 reflection 或 private note。
- event trigger 的具体 predicate 集合如何配置，例如哪些合同条款或资源状态会触发 condition-based event。
- 是否需要在 v0 resolver 之外，引入 urgency、合同状态或 agent priority 来处理更复杂的 scheduling conflict。
- 同一 slot 内的并行 sessions 在代码执行上是顺序模拟还是并行调用；机制上 v0 允许 participants 互不重叠的 sessions 并行成立。
- 多方 session 中 agent 是严格串行发言，还是允许同一轮中多个 agent 并行 response。
- session-control budget 是否需要单独设置上限。
- 起始禀赋如何设计，包括 cash、asset、liability、reputation、private information。