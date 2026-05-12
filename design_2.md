# Setting 1 世界机制规格草案 v1

> 目的：记录我们要实现的长期商业谈判世界机制。本文先写机制，不急着写代码。具体轮数、上限和参数先用符号表示，后续可以作为可扩展配置。

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

整个世界在以下情况下终止：交易主体签署合同且该合同在 closing / final execution check 中被判定可执行；到达 deadline 仍未形成可 closing 的有效协议；交易主体执行 `terminate.negotiation` 且 控制器 判定整体谈判终止；或 控制器 判定不存在可行协议或继续谈判路径。终止机制在 [终止条件](#9-终止条件) 中详细定义。

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
- scenario generation 接口与原则见 [Scenario Generation Interface](#13-scenario-generation-interface)。

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

V1 先不定义完整最终 `U_i`。World mechanism 不在 episode 中途结算 expected utility、折现风险、expected penalty 或 no-deal cost；它只推进状态、记录 trajectory，并在 closing / terminal condition 触发时冻结 final snapshot。未来 evaluator 若需要数值 utility，应从完整 trajectory 和 final state 后置复原 realized utility：只有实际 closing success、closing failure、timeout 或 termination 触发的条件才产生对应收益或成本。

Agent 实际看到多少 objective / constraint 信息由 generator / runner 侧的 `PromptObjectiveModule` 和 `ExperimentRunConfig.objective_level` 控制，不属于 world mechanism 的状态转移规则。同一个 `ScenarioInstance` 应能在 L0 / L1 / L2 / L3 objective prompt level 下重复运行，控制器 只负责执行相同的 visibility、action validity、closing 和 trajectory 记录规则。

对于需要立即可执行的角色约束，world mechanism 可以定义 role-specific `utility_proxy_i` 和 hard feasibility predicates。当前 V1 先落地 investor 的简单融资收益 proxy：investor 的收益只来自成功 closing 后 buyer 使用其融资所支付的利息；interest rate 由 `ScenarioInstance.initial_state.financing_market.interest_rate` 给定，不由 agent 在合同中谈判。

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

buyer 在当前 world state `omega_t` 下的自有可用资金为：

```text
cash_firm_a(t)
```

合同当前需要的 upfront cash 为：

```text
upfront_cash_required(C)
```

若：

```text
cash_firm_a(t) >= upfront_cash_required(C)
```

则 buyer 可以不依赖外部融资完成交易；investor 不是该合同的必要参与方。

若：

```text
cash_firm_a(t) < upfront_cash_required(C)
```

则合同需要外部融资或付款结构调整。此时 investor 可以成为该合同的 contingent required party，只有当 investor 执行 `commit.finance_commit`，或合同被修改为不再需要外部融资时，交易才可能成功。

regulator 是否需要参与也不是固定设定，而由 regulatory requirement 决定。V1 不让 regulator 自己决定是否入场，也不让入场依赖 regulator 的 private utility 或 threshold；入场由 控制器 根据 generator 初始化状态、event shock 和合同离散条款确定。

```text
RegulatoryRequired(C, omega_t) in {0, 1}
```

表示合同 `C` 在当前 world state `omega_t` 下是否需要监管批准。V1 的直觉规则是：generator 或 event 明确要求监管审查；或合同合规太低；或交易估值很高但没有 enhanced compliance。

若：

```text
RegulatoryRequired(C, omega_t) = 0
```

则 regulator 不是该合同的必要参与方。

若：

```text
RegulatoryRequired(C, omega_t) = 1
```

则 regulator 成为该合同的 contingent required party，合同成功需要 regulator 执行 `commit.approve`。外部事件可以把 `omega_t.regulatory_state.review_required` 从 false 改为 true，也可以改变 regulator 对合同的可接受性。

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

### 1.3 Scenario Generation Interface

Setting 1 的 world mechanism 不假定固定单一场景。每个 episode 由一个 scenario instance config 初始化。Scenario generator 负责从受控参数空间中生成 playable instances，但 generator 不能改变 world mechanism 的 action semantics、state transition、visibility、event timing 或 terminal semantics。

Scenario generator 可以决定：

- agent profiles。
- initial resources。
- market and regulatory state。
- investor capacity and risk preference。
- contract parameter mappings。
- event scripts / shock timing / shock magnitude / shock visibility。
- time config。
- private goals and thresholds。
- metadata and difficulty label。

Scenario generator 不可以覆盖：

- action class 和 action schema。
- scheduling resolution rules。
- session turn order semantics。
- contract transition functions。
- contract visibility rules。
- event application semantics。
- closing / final execution check semantics。
- terminal state semantics。
- observation filtering rules。

一个 scenario instance 至少需要提供：

```text
ScenarioInstance = {
  instance_id,
  template_id,
  agent_profiles,
  initial_state,
  contract_parameter_mappings,
  event_scripts,
  time_config,
  private_goals_and_thresholds,
  metadata
}
```

完整 generator 规则、semantic catalogs、numeric scopes、playability diagnostics、difficulty labels 和 rejection / resampling policy 见：

```text
ideas/setting-1-scenario-generator-spec-v0.md
```

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

### 2.1 Controller Execution Order

V1 使用确定性的 controller order。它的目标是消除同一时点多个机制同时触发时的顺序歧义，使 replay 和不同实现之间保持一致。

核心约定如下：

```text
parallel_sessions = logical_parallel_with_slot_barrier
same_slot_terminal_policy = do_not_interrupt_other_sessions_no_rollback
end_of_day_order = events_then_refresh_then_closing_then_timeout_then_snapshot
after_formal_action_policy = immediate_state_refresh_no_closing
```

#### 2.1.1 Day-Level Order

```text
Day d:

  start_of_day:
    1. apply trigger.timing = start_of_day events by event_order
    2. refresh_required_flags for affected active contracts
    3. if terminal_state is set, freeze world and stop

  for k = 1 to S_max:

    before_scheduling:
      1. apply trigger.timing = before_scheduling events by event_order
      2. refresh_required_flags for affected active contracts
      3. if terminal_state is set, freeze world and stop

    scheduling_window:
      1. build invitation observations from invitation-round snapshot
      2. collect and batch-commit session requests
      3. build response observations from response-round snapshot
      4. collect and batch-commit session responses
      5. resolve Sessions(d, k)

    session_execution:
      1. create slot_start snapshot
      2. execute all sessions in Sessions(d, k) with logical-parallel semantics
      3. within each session, apply each turn immediately to that session path
      4. formal action triggers after_formal_action micro-commit
      5. no attempt_close happens during session execution
      6. any world-terminal formal action creates pending_world_terminal

    slot_barrier / post-session:
      1. close all sessions in this slot
      2. commit session logs and action logs
      3. merge valid state deltas in deterministic order
      4. update visible histories
      5. refresh_required_flags for affected active contracts
      6. if pending_world_terminal is set:
           set terminal_state, save terminal snapshot, freeze world, and stop
      7. apply trigger.timing = after_session events by event_order
      8. refresh_required_flags for affected active contracts
      9. apply trigger.timing = end_of_slot events by event_order
      10. refresh_required_flags for affected active contracts
      11. if terminal_state is set, save terminal snapshot, freeze world, and stop

  end_of_day:
    1. apply trigger.timing = end_of_day events by event_order
    2. refresh_required_flags for affected active contracts
    3. for each active signed contract ordered by contract_id:
         if d >= C.closing_state.scheduled_day:
           refresh_required_flags(C, omega_t)
           attempt_close(C, omega_t, d)
           if terminal_state in {success, failure}:
             break
    4. if terminal_state is not set and d = D:
         set terminal_state = timeout
    5. save end-of-day or terminal snapshot
    6. if terminal_state is set, freeze world
```

`event_order` is deterministic. V1 默认按 `(day, slot_id, timing, event_id)` 排序；如果 scenario config 显式提供 `event_order_index`，则先按 `event_order_index`，再按 `event_id` 打破平局。

#### 2.1.2 Logical-Parallel Sessions

同一 session slot 内的多个互不重叠 sessions 是逻辑并行的。实现可以为了简单按 `session_id` 顺序模拟，但语义上所有 sessions 都从同一个 `slot_start snapshot` 开始。

因此：

- 一个 session 中产生的 message、formal action、contract update 或 visibility update，不会在同一 slot 内暴露给另一个并行 session。
- 同一 slot 内不同 sessions 不能依赖彼此刚刚产生的状态。
- 同一 slot 内所有 session 的 logs 和 state deltas 在 slot barrier 统一提交。
- 如果两个并行 session 的 state deltas 意外写同一个 world object，V1 不做自动冲突解决；该情况应作为 invalid scenario 或 implementation error 失败。按 V1 scheduling 约束，正常情况下互不重叠 participants 不应产生同一对象的并发写冲突。

#### 2.1.3 Terminal During a Slot

如果某个 session 内出现 world-terminal formal action，例如交易主体执行 `terminate.negotiation`，则：

- 该 formal action 对当前 session 立即生效。
- 当前 session 立刻结束。
- 控制器 设置 `pending_world_terminal`，记录 terminal actor、action 和 reason。
- 同一 slot 中已经形成的其他 sessions 不被回滚，也不被中断。
- 其他 sessions 已发生的 turn、message、formal action 和 state delta 保留在 log 中。
- slot barrier 时先提交同一 slot 的 logs / deltas，再把 `pending_world_terminal` 提升为 `terminal_state`。
- 进入 `terminal_state` 后，不再开启后续 slot、end-of-day event、closing check、new session、message、formal action 或新合同签署。

这个规则与 logical-parallel sessions 一致：同一 slot 表示同一逻辑时间段内发生的互不重叠会谈，某个 session 的 terminal action 不会 retroactively 取消其他同时发生的会谈。

#### 2.1.4 After-Formal-Action Micro-Commit

`after_formal_action` 是 session 内的微型状态提交点，不是 V1 外部事件触发点，也不执行 closing。

每个 valid formal action 的应用顺序为：

```text
1. validate action schema, actor, role, session membership, visibility, budget, and state predicates
2. write action log and state delta
3. apply formal state mutation
4. refresh_required_flags for affected active contracts
5. update visible_contracts and available_actions for future turns in the same session
6. if the action has immediate session/path/world-terminal semantics:
     apply session/path effect immediately
     mark pending_world_terminal if needed
7. do not attempt_close
```

`commit.sign` 只会把合同推进到 `signed`，并在双方都签署后写入 `closing_state.scheduled_day`。它不会立即触发 success 或 failure。`attempt_close` 只在 end-of-day order 中执行。

#### 2.1.5 Implementation-Oriented Pseudocode

下面的伪代码是 V1 controller contract 的实现形态。它不引入新机制，只把上面的 controller order 写成更接近代码的步骤。

```text
run_episode(initial_state):
  omega = initial_state

  for d in 1..D:
    run_day(d, omega)

    if omega.terminal_state is set:
      freeze_world(omega)
      return omega

  if omega.terminal_state is not set:
    omega.terminal_state = timeout
    save_terminal_snapshot(omega)
    freeze_world(omega)

  return omega
```

```text
run_day(d, omega):
  apply_events(timing = start_of_day, day = d, slot_id = null, omega)
  refresh_affected_active_contracts(omega)
  if stop_if_terminal(omega):
    return

  for k in 1..S_max:
    run_slot(d, k, omega)

    if omega.terminal_state is set:
      return

  run_end_of_day(d, omega)
```

```text
run_slot(d, k, omega):
  apply_events(timing = before_scheduling, day = d, slot_id = k, omega)
  refresh_affected_active_contracts(omega)
  if stop_if_terminal(omega):
    return

  invitation_snapshot = snapshot(omega)
  requests = collect_session_requests(invitation_snapshot, d, k)
  requests = normalize_invalid_or_missing_requests(requests)

  response_snapshot = snapshot_with_committed_requests(omega, requests)
  responses = collect_session_responses(response_snapshot, d, k, requests)
  responses = normalize_invalid_or_missing_responses(responses)
  sessions = resolve_sessions(requests, responses)

  slot_start_snapshot = snapshot(omega)
  slot_results = []

  for s in deterministic_session_order(sessions):
    # Implementation may run this loop sequentially, but each session's
    # observation must be based on slot_start_snapshot plus its own session path,
    # not on another same-slot session's updates.
    slot_results.append(run_session(s, slot_start_snapshot, omega))

  slot_barrier_commit(slot_results, omega)

  if omega.pending_world_terminal is set:
    commit_pending_world_terminal(omega)
    save_terminal_snapshot(omega)
    freeze_world(omega)
    return

  apply_events(timing = after_session, day = d, slot_id = k, omega)
  refresh_affected_active_contracts(omega)

  apply_events(timing = end_of_slot, day = d, slot_id = k, omega)
  refresh_affected_active_contracts(omega)

  if stop_if_terminal(omega):
    save_terminal_snapshot(omega)
    freeze_world(omega)
```

```text
run_session(session, slot_start_snapshot, omega):
  session_state = initialize_session_path(session, slot_start_snapshot)

  while not session_end_condition(session_state):
    active_agent = next_agent_in_turn_order(session_state)
    observation = build_observation(active_agent, session_state, slot_start_snapshot)
    turn_output = call_agent(active_agent, observation)

    apply_turn_output(turn_output, session_state, omega)

    if session_state.session_terminal is set:
      break

  return collect_session_result(session_state)
```

```text
apply_turn_output(turn_output, session_state, omega):
  increment_turn_counters(session_state)

  if turn_output.action_class = msg:
    write_message_log(turn_output, session_state)
    return

  if turn_output.action_class = pass:
    write_pass_log(turn_output, session_state)
    return

  if turn_output.action_class = ctrl:
    apply_session_control(turn_output, session_state)
    return

  if turn_output.action_class = formal:
    apply_formal_action(turn_output, session_state, omega)
    return
```

```text
apply_formal_action(action, session_state, omega):
  validate action schema, actor, role, session membership, visibility, budget, and state predicates

  if action is invalid:
    write_invalid_action_log(action, session_state)
    return

  write_action_log(action, session_state)
  delta = apply_formal_state_mutation(action, session_state, omega)
  write_state_delta(delta, session_state)

  refresh_required_flags for affected active contracts
  update same-session visible_contracts and available_actions

  if action creates session/path terminal effect:
    apply that effect immediately

  if action creates world terminal effect:
    omega.pending_world_terminal = terminal_reason_from(action)
    session_state.session_terminal = true

  # No closing here.
  # commit.sign can set C.status = signed and C.closing_state.scheduled_day,
  # but success/failure is only decided by run_end_of_day.
```

```text
slot_barrier_commit(slot_results, omega):
  close all sessions in slot_results
  commit session logs and action logs
  merge valid state deltas in deterministic (session_id, turn_id, action_id) order
  update visible histories
  refresh_required_flags for affected active contracts
```

```text
run_end_of_day(d, omega):
  apply_events(timing = end_of_day, day = d, slot_id = null, omega)
  refresh_affected_active_contracts(omega)

  for C in active_signed_contracts_ordered_by_contract_id(omega):
    if d >= C.closing_state.scheduled_day:
      refresh_required_flags(C, omega)
      attempt_close(C, omega, d)

      if omega.terminal_state in {success, failure}:
        break

  if omega.terminal_state is not set and d = D:
    omega.terminal_state = timeout

  save_end_of_day_or_terminal_snapshot(omega)

  if omega.terminal_state is set:
    freeze_world(omega)
```

### 2.2 Post-session Bookkeeping

Post-session bookkeeping 是 world-model-only 阶段。它不是 agent action phase，也不调用 agent 产生新的 message、formal action、session control action 或 private note。

这个限制很重要：V1 中所有主动沟通都必须发生在 active session 内。如果 post-session 允许 agent 再主动发言、补充承诺或写入可影响状态的 private reflection，它就会变成一个隐形 action phase，破坏 session 机制的边界。

当一个 session slot 内的所有 active sessions 都结束后，控制器 在 slot barrier 执行 post-session bookkeeping。它负责：

1. 关闭所有已结束 session，记录 `t_end` 和结束原因。
2. 把 session 内 message、formal action、session control action 写入 session log 和 action log。
3. 更新每个 agent 的可见历史：
   - session participants 收到自己参与期间可见的 session transcript。
   - 中途退出者只收到退出前的 session 内容。
   - 非 participants 不收到该 session 内容。
4. 合并本 slot 的有效 state deltas，并更新合同状态、session turn counters、daily formal-action counters 和 session-control 状态。
5. 刷新受影响 active contracts 的 required-party cache。
6. 处理 `pending_world_terminal`；若存在，则设置 `terminal_state`、保存 terminal snapshot，并停止后续 world progression。
7. 按 controller order 触发 `after_session` 和 `end_of_slot` events，并在每组事件后刷新受影响合同。
8. 如果没有 terminal，生成给下一 slot 的 observation input，例如刚刚参与了什么、还剩多少 budget、当前可见合同状态是什么。

Post-session bookkeeping 不执行 `attempt_close`。Closing / final execution check 只属于 end-of-day controller phase。

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

Scheduling window 使用阶段同步的 semantic-parallel round。也就是说，Invitation Round 和 Response Round 是两个顺序 phase；每个 phase 内，所有需要被调用的 agent 都基于同一个 round-start snapshot 产生输出，控制器 收集该 phase 的全部输出后才进入下一 phase。

工程实现可以真正并发调用 LLM，也可以按 role order 串行调用以便调试、限流或复现；但串行调用不得改变机制语义：

```text
for each scheduling phase:
  build observation_i from phase_start_snapshot
  collect output_i for all eligible agents
  normalize invalid / missing outputs
  commit phase results as a batch
```

因此，较早返回或较早被串行调用的 agent 不能看到同一 phase 中其他 agent 的 request / response。role order 只可用于 tie-breaking、日志排序、稳定 replay 或工程调用顺序，不表示机制上的先后行动。

每个 agent 根据自己的当前 observation 生成 request 或 response，控制器 收集后统一 resolve。resolve 后，控制器 再把结果拼接回每个 agent 的 observation，告知它：

- 自己发起的 request 是否被接受或拒绝。
- 自己收到的 request 中哪些成立、哪些未成立。
- 接下来是否进入某个 session。
- 若进入 session，自己所在 session 的 participants 是谁。
- 若未进入 session，自己知道哪些相关结果。

每个 Scheduling Window 的内部流程为：

1. **Invitation Round**：各 agent 提交 session request。
2. **Response Round**：被邀请者对收到的 request 做 accept/decline。
3. **Resolution**：控制器 汇总 request 与 response，决定该 slot 形成哪些 session。

v1 暂定只有一轮 request + 一轮 response，不支持多轮协商式调度。

### 3.1 Invitation Round

在 invitation round 中，每个 agent `i` 可以提交至多 `Q_i` 个 session request。v1 暂定：

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

如果 agent 在 invitation round 中没有提交合法 `session_request`，则记为 `no_request`，本 slot 不产生 outgoing request。`no_request` 的细化触发条件见 [Session Request](#621-session-request)。

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

非被邀请者在 v1 中不会看到：

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

如果 `firm_a` 邀请 `firm_b`，而 `firm_b` decline，则该 request 不成立。`firm_b` 可以接受或拒绝收到的 request，但不能在 response round 中重新发起新的 request，除非配置允许额外的 scheduling round。

v1 进一步约束：

```text
每个 agent 在一个 session slot 中最多保留一个 slot commitment。
```

slot commitment 是 agent 对当前 slot 要参加哪一个 participant set 的排他选择。它可以来自：

1. agent 自己在 invitation round 发起的 outgoing request；
2. agent 在 response round 接受的 incoming request；
3. 与自己 outgoing request 拥有完全相同 `proposed_participants` 的 reciprocal incoming request。

因此 response round 不只是逐条 accept / decline，而是在确定 agent 对当前 slot 的唯一 commitment。

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

`no_response` 的触发条件包括但不限于：

- LLM / tool call 超时。
- LLM / tool call 抛出异常或返回空输出。
- 输出无法解析为合法 action schema。
- `action_class != sched`。
- `action_type != session_response`。
- `request_id` 缺失、不可见、不是当前 response round 中发给该 agent 的 request，或引用了已失效 request。
- `decision` 缺失，或不在 `{accept, decline}` 中。
- agent 输出了 message / formal / ctrl / pass，而不是 scheduling response。
- agent 输出自然语言闲聊、实质谈判内容、合同条款、融资承诺、监管意见等，但没有给出可解析的 scheduling decision。

这些情况表示 agent 没有提供可用于 scheduling resolution 的有效回复；它不等价于主动 `decline` 的策略表达，但在 resolution 中具有同样效果。

如果某个 agent 在同一个 session slot 中 accept 多个 `proposed_participants` 不同的 incoming request，则该 agent 对所有 incoming request 的 accept 都无效：

```text
if count_distinct_accepted_participant_sets_i(k) > 1:
  effective_response_i(r) = decline for all r received by i in slot k
```

该情况也写入 log / trajectory，作为违反 response 约束的工具调用失误。

若 agent 接受一个 incoming request，且该 incoming request 的 `proposed_participants` 与自己发起的 outgoing request 不同，则自己发起的 outgoing request 自动撤回：

```text
if agent i accepts incoming request r_in
and outgoing_request_i exists
and P_{outgoing_request_i} != P_{r_in}:
  outgoing_request_i.status = withdrawn_by_requester_commitment
  slot_commitment_i = P_{r_in}
```

直觉是：agent 在 response round 中决定接受别人邀请，就表示它认为那个会谈优先级更高，不能同时保留自己发起的另一个会谈。

若 agent 拒绝所有 incoming request，且自己有 outgoing request，则保留自己的 outgoing request 作为当前 slot commitment：

```text
if agent i declines all incoming requests
and outgoing_request_i exists:
  slot_commitment_i = P_{outgoing_request_i}
```

若 agent 没有 outgoing request，也没有接受任何 incoming request，则当前 slot idle。

双向邀请或完全相同 participant set 的 reciprocal request 不需要再次确认，直接视为相同 slot commitment。例如：

```text
firm_a invites {firm_a, firm_b}
firm_b invites {firm_a, firm_b}
```

则 `firm_a` 和 `firm_b` 都自动 committed to `{firm_a, firm_b}`，两个 request 在 resolution 中合并为一个 session。更一般地，如果 agent 收到的 incoming request 与自己 outgoing request 的 `proposed_participants` 完全相同，则该 incoming request 对该 agent 视为 accepted，而不是撤回自己的 outgoing request：

```text
if outgoing_request_i exists
and P_{outgoing_request_i} = P_{r_in}:
  effective_response_i(r_in) = accept
  slot_commitment_i = P_{r_in}
```

### 3.4 Session Resolution

控制器 根据 request 和 response 生成当前 session slot 的 session set。

一个 request 成立的必要条件：

```text
all agents in proposed_participants accept
and no requester has withdrawn the request
and all agents in proposed_participants have slot_commitment_i = proposed_participants
```

设所有成立 request 的集合为：

```text
R_valid(d, k) = {
  r |
    r.status != withdrawn_by_requester_commitment
    and all agents i in proposed_participants(r):
      slot_commitment_i = proposed_participants(r)
}
```

一个 session slot 可以创建多个 session，但这些 session 的 participants 必须两两不重叠。控制器 使用 deterministic resolver 将 `R_valid(d, k)` 转换为 `Sessions(d, k)`。

v1 resolver 规则如下：

1. **相同 participants 合并**：若多个 accepted request 的 `proposed_participants` 完全相同，则合并为一个 session。
2. **包含关系取大**：若两个 accepted request 有重叠，且一个 participant set 严格包含另一个，即 `P_a ⊂ P_b`，则保留更大的 participant set，较小 request 被视为被更完整的 session 吸收。
3. **复杂交叉冲突失败**：若两个 accepted request 有重叠，但不存在包含关系，则这些冲突 request 在 v1 中全部失败，并写入 log / trajectory。
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

由于双向邀请自动视为相同 slot commitment，两个 request 会被合并，只形成一个 session：

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

若 `firm_a` 在 response round 接受 `firm_b` 的三方 request，则 `firm_a` 自己发起的二人 request 自动撤回。若 `investor` 也接受三方 request，则只形成更大的三方 session：

```text
Sessions(d, k) = {
  {firm_a, firm_b, investor}
}
```

若 `firm_a` 拒绝三方 request，而 `firm_b` 接受 `firm_a` 的二人 request，则 `firm_b` 自己发起的三方 request 自动撤回，只形成二人 session：

```text
Sessions(d, k) = {
  {firm_a, firm_b}
}
```

若出现无法通过包含关系解决的交叉冲突，例如：

```text
firm_a invites firm_b, investor
firm_b invites firm_a, regulator
```

若两个 request 在 slot commitment 规则后仍同时进入 `R_valid(d, k)`，则两个冲突 request 在 v1 中都失败。控制器 不替 agent 主观选择其中一组。

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
- 同一 slot 内多个 active sessions 采用 logical-parallel semantics：它们共享同一个 slot-start snapshot，跨 session 状态和信息只在 slot barrier 后提交，不在同一 slot 内相互可见。

### 4.2 非参与者

对于任一 active session，如果其 participants 为 `{firm_a, firm_b}`，则不在该 session 内的 `investor` 和 `regulator`：

- 不会被调用做 session action。
- 不能给 session 内 agent 发消息。
- 不能邀请 session 内 agent 开新 session。
- 不消耗 session 内 action budget。

他们不能插入该 session；若同一 slot 内没有属于自己的并行 session，则只能等待后续 slot 或 next day 的 scheduling window。

session 内信息默认只对 session participants 可见。多个 agent 参与同一个 session 时，v1 不再细分 session 内 public/private；session 内容对所有当前 participants 共享，对非 participants 不可见。

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

一次 session turn 中，active agent 必须且只能选择一种 session action class：

```text
turn_output = {
  action_class: msg | formal | ctrl | pass,
  payload: object | null
}
```

四类 session turn 含义如下：

- **msg**：沟通、解释、询问、披露信息、说服对方。
- **formal**：改变世界状态或合同状态的动作（如 `contract.propose`）。
- **ctrl**：控制 session 本身（如 `leave_session`、`terminate.session`）。
- **pass**：当前 agent 不发送 message，也不执行任何状态改变。

约束：

- v1 中 session 内 message 默认对当前所有 participants 可见。
- message receiver 可记为 `ALL_IN_SESSION`，或省略后由 控制器 自动解释为当前 participants。
- session 内暂不支持只发给部分 participants 的 private message。
- 每个 session turn 恰好属于一个 `action_class`；不能在同一个 turn 中同时输出 message、formal action 和 session control action。
- `formal_action` 必须符合当前 agent、当前 session 和当前 world state 的可用 schema。
- `session_control_action` 不属于 `formal_action`；V1 中二者都消耗统一 session turn。

turn 消耗规则：

- active agent 被调用并返回一次 `turn_output`，就消耗一个 turn。
- 发送 message 消耗一个 turn。
- 执行 formal action 消耗一个 turn。
- 执行 session control action 消耗一个 turn。
- `pass` 也消耗一个 turn。
- 如果 formal action invalid，该 turn 仍然被消耗；控制器 只是不应用 invalid formal state mutation。

V1 的 session 内行动额度由统一 turn budget 控制，而不是分别给 message 和 formal action 设置独立的 per-session turn budget。也就是说，agent 必须在有限的 `K_s` 次行动机会中自行分配：用来聊天、提出合同、修改合同、签署、退出或 pass。

```text
session_turn_budget_i(s) = K_s - K_i(s)
```

message count 和 formal action count 可以被记录为行为统计，但 V1 不使用 `M_max` 或 `H_max` 作为额外 per-session hard constraint。

`pass` 可以表示为：

```text
turn_output = {
  action_class: pass,
  payload: null
}
```

message 只能在 session 内发送。session 外不允许 agent 主动发送 message。

session 内 agent 发言是串行的。v1 候选规则：

```text
1. 每个 session 的第一个行动者按 deterministic participant order 决定。
2. 之后按同一个 deterministic participant order 轮流行动。
3. 如果有 agent 退出，则从后续 turn order 中移除。
4. 如果某个 agent 已达到 `K_i(s) = K_s`，则跳过该 agent。
```

v1 暂时使用固定 role order：

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

**两人 session：`{firm_a, firm_b}`，`T_s = 10`，`K_s = 5`**

```text
turn 1: firm_a   — message                      (解释报价意图，消耗 1 turn)
turn 2: firm_b   — message                      (回应报价方向，消耗 1 turn)
turn 3: firm_a   — contract.propose             (正式提出合同，消耗 1 turn)
turn 4: firm_b   — message                      (回应条款，消耗 1 turn)
turn 5: firm_a   — contract.accept              (正式接受合同，消耗 1 turn)
turn 6: firm_b   — contract.accept              (正式接受合同，消耗 1 turn)
turn 7: firm_a   — commit.sign                  (正式签署，消耗 1 turn)
turn 8: firm_b   — commit.sign                  (正式签署，消耗 1 turn)
```

双方未必用满 `T_s` 或 `K_s`；双方签署后合同进入 pending closing。只有后续 closing / final execution check 通过，才会触发 world-level success condition。

**两人 session 提前结束：`{firm_a, firm_b}`，`T_s = 6`，`K_s = 3`**

```text
turn 1: firm_a — message
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
turn 1: firm_a   — contract.propose
turn 2: firm_b   — message
turn 3: investor — terminate.session            (investor 退出)
turn 4: firm_a   — message
turn 5: firm_b   — contract.accept
turn 6: firm_a   — commit.sign                  (已无 investor，无需 finance_commit)
```

investor 在 turn 3 退出后，后续 turn 顺序跳过 investor，由 firm_a 和 firm_b 继续轮流。investor 看不到 turn 4-6 的任何内容。若合同不需要融资，交易仍可由 firm_a 和 firm_b 直接完成。

### 4.4 离开 Session 与 Session 自然结束

agent 可以在 session 内通过 session control action 离开当前 session：

```text
leave_session
terminate.session
```

在 v1 中，`terminate.session` 的语义不是强制关闭整个 session，而是表示发出该 action 的 agent 自己离开当前 session。它等价于更显式的 `leave_session`。

当一个 agent 选择 `leave_session` 或 `terminate.session`：

- 它从当前 session participants 中移除。
- 它不再收到该 session 后续 message 或 action observation。
- 它不能在该 session 内再次发言或行动。

如果 session 中剩余 participants 少于 2 人，则 控制器 结束该 session。

如果三人 session 中一人离开，剩下两人可以继续 session；离开者听不到剩余两人的后续交流。

退出后的可见性是 session-scope，不是 episode-scope。agent 离开 session 不会删除它已经看到的历史，也不会让它在后续 slot/day 失去对退出前可见内容的记忆。

如果 agent `i` 在 session `s` 的 turn `tau` 离开：

```text
visible_session_logs_i(s) include turns <= tau that were visible to i
visible_session_logs_i(s) exclude turns > tau
future observations in later slots/days may include the retained visible prefix of s
```

若 `i` 在后续 slot 加入另一个 session，它的 observation 可以包含 `s` 中退出前可见的 transcript、action log 和当时可见的 contract objects；但不包含退出后 `s` 中发生的 message、formal action 或 contract mutation，除非这些信息后来通过 `contract.share`、`review.request_*`、message disclosure、public event 或其他正常 visibility mechanism 再次对 `i` 可见。

同一 slot 内的 logical-parallel sessions 仍然不能在 slot barrier 前互相观察更新。退出前可见前缀只会在该 slot 的 slot-barrier bookkeeping 后，进入后续 slot/day 的 observation construction。

## 5. Contract 世界状态

合同是 formal action 的核心操作对象。为了让机制能够直接转化成代码，本节把合同定义为 typed state object，并把合同条款定义为 finite domains、derived variables、validity predicates 和 transition functions。

控制器 维护一个全局合同账本：

```text
ContractBook_t = {C_1, C_2, ..., C_n}
```

### 5.1 Contract Data Model

每个合同 `C` 是一个结构化对象：

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

其中 `created_at` 记录合同产生时点：

```text
created_at = {
  day,
  slot_id,
  session_id,
  turn_id
}
```

`parties` 至少包含交易主体：

```text
parties = {firm_a, firm_b}
```

如果合同需要融资或监管批准，investor / regulator 不一定成为合同交易主体，但会成为 contingent required parties。

合同状态取值为：

```text
ContractStatus = {
  proposed,
  amended,
  accepted,
  rejected,
  signed,
  closed_success,
  closed_failed,
  superseded,
  failed
}
```

`signed` 不等于交易成功。它表示交易主体已经做出签署承诺，但交易还需要经过 closing / final execution check。只有当 closing check 通过后，合同才进入 `closed_success`，world-level `terminal_state` 才能变成 `success`。

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

closing 状态为：

```text
closing_state = {
  status: not_started | pending | closed_success | closed_failed,
  scheduled_day: day optional,
  checked_at: time optional,
  failure_reasons: list
}
```

### 5.2 Terms Data Model

Setting 1 的合同条款 `terms` 是合同的核心可谈判对象：

```text
Terms = {
  valuation,
  payment,
  closing,
  compliance,
  penalty
}
```

V1 采用“枚举选择 + 控制器 派生数值”的设计。agent 在合同中填写有限枚举值，控制器 根据映射表计算价格、现金需求、融资需求、监管风险和违约保护等派生变量。

#### 5.2.1 Valuation

`valuation` 表示交易总价水平，而不是会计估值口径。V1 不引入 `valuation_basis`，除非后续明确建模现金、债务、营运资本或价格调整。

```text
ValuationDomain = {low, medium, high}
```

派生价格：

```text
price(valuation) = {
  low: 80,
  medium: 100,
  high: 120
}
```

其中数值单位可以理解为 normalized monetary unit，真实实现中可配置。

#### 5.2.2 Payment

`payment` 表示付款结构、付款确定性和资金来源。它不同于 `valuation`：`valuation` 决定总价，`payment` 决定怎么付、什么时候付、是否需要外部融资。

```text
PaymentDomain = {
  all_cash,
  staged_cash,
  investor_financed,
  seller_note
}
```

派生变量：

```text
upfront_ratio(payment) = {
  all_cash: 1.0,
  staged_cash: 0.6,
  investor_financed: 0.4,
  seller_note: 0.3
}

deferred_ratio(payment) = {
  all_cash: 0.0,
  staged_cash: 0.4,
  investor_financed: 0.0,
  seller_note: 0.7
}

external_financing_flag(payment) = {
  all_cash: 0,
  staged_cash: 0,
  investor_financed: 1,
  seller_note: 0
}
```

给定合同 `C`：

```text
price(C) = price(C.terms.valuation)
upfront_cash_required(C) = price(C) * upfront_ratio(C.terms.payment)
deferred_payment(C) = price(C) * deferred_ratio(C.terms.payment)
```

#### 5.2.3 Closing

`closing` 表示交易何时正式完成，以及完成前需要满足哪些条件。付款时间可以和 closing 相关，但付款结构本身属于 `payment`。

```text
ClosingDomain = {fast, standard, delayed}
```

派生变量：

```text
closing_delay(closing) = {
  fast: 1,
  standard: 2,
  delayed: 3
}

base_conditions(closing) = {
  fast: {},
  standard: {seller_disclosure},
  delayed: {seller_disclosure, enhanced_due_diligence}
}
```

给定签署日 `d_sign`：

```text
closing_day(C, d_sign) = d_sign + closing_delay(C.terms.closing)
```

合同在双方完成 `commit.sign` 时写入具体 closing day；写入后该日期固定，不随后续 day 漂移：

```text
C.closing_state.scheduled_day = closing_day(C, d_sign)
```

合同 scheduled closing 必须发生在交易 deadline 内：

```text
valid_closing_schedule(C) iff
  C.closing_state.scheduled_day <= D
```

closing 条件由基础条件、融资条件和监管条件共同决定：

```text
required_conditions(C, omega_t)
  = base_conditions(C.terms.closing)
    ∪ financing_condition(C, omega_t)
    ∪ regulatory_condition(C, omega_t)
```

#### 5.2.4 Compliance

`compliance` 表示合规承诺和监管风险处理。它影响 regulator 入场规则、监管批准直觉、以及 buyer/seller 承担的合规成本。

```text
ComplianceDomain = {minimal, standard, enhanced}
```

派生变量。`regulatory_risk(compliance)` 是 compliance 的解释性 shorthand，不是 regulator 入场使用的额外连续分数；入场规则由 `RegulatoryRequired(C, omega_t)` 明确定义。

```text
regulatory_risk(compliance) = {
  minimal: high,
  standard: medium,
  enhanced: low
}

compliance_cost(compliance) = {
  minimal: 0,
  standard: 5,
  enhanced: 10
}
```

#### 5.2.5 Penalty

`penalty` 表示交易失败、延迟、融资失败或监管失败时的风险分配和违约保护。它影响 seller 的交易确定性，也影响 buyer 的潜在成本。

```text
PenaltyDomain = {low, medium, high}
```

派生变量：

```text
breakup_fee(penalty) = {
  low: 2,
  medium: 5,
  high: 10
}

seller_certainty_bonus(penalty) = {
  low: 0,
  medium: 3,
  high: 6
}

buyer_penalty_cost(penalty) = {
  low: 0,
  medium: 3,
  high: 6
}
```

### 5.3 Validity and Feasibility Predicates

合同条款合法性：

```text
ValidTerms(C) iff
  C.terms.valuation in ValuationDomain
  and C.terms.payment in PaymentDomain
  and C.terms.closing in ClosingDomain
  and C.terms.compliance in ComplianceDomain
  and C.terms.penalty in PenaltyDomain
```

融资需求：

```text
FinancingRequired(C, omega_t) = 1 iff
  external_financing_flag(C.terms.payment) = 1
  or cash_firm_a(t) < upfront_cash_required(C)
```

融资缺口：

```text
financing_gap(C, omega_t)
  = max(0, upfront_cash_required(C) - cash_firm_a(t))
```

若 `external_financing_flag(C.terms.payment) = 1` 但 `cash_firm_a(t) >= upfront_cash_required(C)`，V1 仍允许合同被标记为需要 investor financing；此时可部署融资额取：

```text
financing_amount(C, omega_t)
  = max(financing_gap(C, omega_t),
        upfront_cash_required(C) * external_financing_flag(C.terms.payment))
```

V1 investor commit 硬约束：

```text
investor_can_commit(C, omega_t) iff
  FinancingRequired(C, omega_t) = 1
  and financing_amount(C, omega_t) > 0
  and financing_amount(C, omega_t)
      <= investor_available_capital(t)
```

V1 investor utility proxy：

```text
prospective_utility_proxy_investor(C, omega_t)
  = financing_amount(C, omega_t) * interest_rate
    if investor_can_commit(C, omega_t)
    else 0

realized_utility_proxy_investor(C, omega_T)
  = financing_amount(C, omega_T) * interest_rate
    if terminal_state = success
       and C.financing.status = committed
       and C.financing.actor = investor
    else 0
```

其中 `interest_rate = omega_t.financing_market.interest_rate`，由 scenario generator 初始化。`prospective_utility_proxy_investor` 只用于 agent reasoning 或 generator diagnostics，不是 控制器 在中途结算的 expected utility。`realized_utility_proxy_investor` 只有在 terminal state 后才能从 final trajectory / snapshot 计算。V1 不把 regulatory risk、default risk、relationship value 等项放入 investor utility；这些可以在后续版本扩展。

融资条件：

```text
financing_condition(C, omega_t) =
  {financing_commitment} if FinancingRequired(C, omega_t) = 1
  else {}
```

监管需求：

```text
RegulatoryRequired(C, omega_t) = 1 iff
  omega_t.regulatory_state.review_required = true
  or C.terms.compliance = minimal
  or (
       C.terms.valuation = high
       and C.terms.compliance != enhanced
     )
```

`RegulatoryRequired` 使用 OR 语义。只要 generator / event shock 让当前 world state 要求审查，或合同自己的离散条款触发审查，合同就需要监管批准。

```text
external_review_required =
  (omega_t.regulatory_state.review_required = true)

contract_induced_review_required =
  (C.terms.compliance = minimal)
  or (
       C.terms.valuation = high
       and C.terms.compliance != enhanced
     )

RegulatoryRequired(C, omega_t) =
  external_review_required OR contract_induced_review_required
```

因此，`compliance = minimal` 一定触发监管批准；`valuation = high` 在 `compliance != enhanced` 时也触发监管批准；`compliance = enhanced` 可以避免 high-valuation 的 contract-induced review，但不能覆盖或取消外部 world state 已经触发的监管审查。`regulatory_state.strictness` 可以保留为 generator 和 diagnostic 使用的背景变量，但 V1 的 regulator 入场判定不依赖连续 strictness 阈值。

监管条件：

```text
regulatory_condition(C, omega_t) =
  {regulatory_approval} if RegulatoryRequired(C, omega_t) = 1
  else {}
```

买方预算可行性：

```text
BuyerBudgetFeasible(C, omega_t) iff
  cash_firm_a(t) >= upfront_cash_required(C)
  or FinancingRequired(C, omega_t) = 1
```

合同 closing 可执行性：

```text
ClosingExecutable(C, omega_t, d) iff
  ValidTerms(C)
  and valid_closing_schedule(C)
  and d >= C.closing_state.scheduled_day
  and BuyerBudgetActuallyFeasible(C, omega_t)
  and FinancingConditionSatisfied(C, omega_t)
  and RegulatoryConditionSatisfied(C, omega_t)
  and SellerPerformanceFeasible(C, omega_t)
```

其中：

```text
BuyerBudgetActuallyFeasible(C, omega_t) iff
  cash_firm_a(t) >= upfront_cash_required(C)
  or (
    FinancingRequired(C, omega_t) = 1
    and C.financing.status = committed
  )

FinancingConditionSatisfied(C, omega_t) iff
  FinancingRequired(C, omega_t) = 0
  or C.financing.status = committed

RegulatoryConditionSatisfied(C, omega_t) iff
  RegulatoryRequired(C, omega_t) = 0
  or C.regulatory.status = approved
```

`SellerPerformanceFeasible(C, omega_t)` 是保留的 seller 侧 closing 谓词，用来表示卖方是否能够完成其承诺的交易交割。

V1 明确定义为：

```text
SellerPerformanceFeasible(C, omega_t) := true
```

也就是说，Setting 1 V1 不把 seller 侧资产不存在、资产被质押或冻结、卖方破产、title defect、披露失败、交割权限缺失等作为 hard closing blocker。V1 的 closing 重点是买方预算 / 融资、监管批准、合同条款和外部事件后的可执行性。

这个谓词保留在 `ClosingExecutable` 中，只作为后续版本的扩展锚点。未来版本可以把它展开为：

```text
SellerPerformanceFeasible(C, omega_t) iff
  seller_disclosure_satisfied(C, omega_t)
  and seller_asset_transferable(C, omega_t)
  and seller_liability_within_limit(C, omega_t)
  and seller_authority_valid(C, omega_t)
```

V1 实现不得自行发明额外 seller-side closing blockers，除非这些 blocker 已经被显式编码为合同条件、事件效果或后续版本的机制规则。

最终成功条件由 [终止条件](#9-终止条件) 定义。`ClosingExecutable` 只在 closing / final execution check 中用于判断签署合同是否真正能完成交易；它不用于阻止 agent 提出、接受或签署一个后续可能失败的合同。

### 5.4 Contract Visibility

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

Visibility only grants reference rights, not modification rights. An agent who can reference a contract may read it and may execute role-allowed actions on it, but cannot necessarily change its main terms, acceptance state, signature state, or closing state.

In V1:

```text
firm_a / firm_b:
  may propose, amend, accept, reject, sign, share, and request review.

investor:
  may finance_commit or finance_decline on visible contracts.
  may not propose, amend, accept, reject, or sign the acquisition contract.

regulator:
  may approve or block on visible contracts.
  may not propose, amend, accept, reject, or sign the acquisition contract.
```

`contract.amend` 是 acquisition contract 的核心条款修改动作。V1 要求正式 amend 发生在同时包含交易双方的 active session 中：

```text
contract.amend allowed only if:
  actor in {firm_a, firm_b}
  firm_a in P_s
  firm_b in P_s
```

因此，`{firm_a, firm_b}`、`{firm_a, firm_b, investor}`、`{firm_a, firm_b, regulator}`、`{firm_a, firm_b, investor, regulator}` 都可以承载正式 amend；`{firm_a, investor}`、`{firm_a, regulator}`、`{firm_b, investor}`、`{firm_b, regulator}`、`{investor, regulator}` 不能承载正式 amend。第三方可以在包含交易双方的 session 中参与讨论、表达融资或监管意见，但不能作为 actor 修改 acquisition terms。

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

V1 中，contract visibility 是单调的：

```text
visibility_set(C, t + 1) ⊇ visibility_set(C, t)
```

没有 agent action 可以从 `visibility_set` 中移除 agent。V1 不支持：

```text
contract.revoke_visibility(C, receiver)
```

如果某个 agent 已经看过同一合同路径中的旧版本，它不会因为后续 amendment 而失去查看最新版本的权利。

普通外部事件不会直接改变合同的 `visibility_set`。例如 `regulation_tightened` 可以改变：

```text
RegulatoryRequired(C, omega_t)
```

但不会自动导致：

```text
regulator in visibility_set(C)
```

regulator 或 investor 只有通过 `review.request_*` 或 `contract.share` 被纳入 `visibility_set` 后，才能看到合同并执行其角色允许的动作。

已签署合同也不会自动变成 public。V1 中，signed contract 仍只对 signatories、required parties 和已经在 `visibility_set` 中的 agent 可见。若后续版本需要更多私有信息或信息泄露机制，可以在 V2 引入显式 `visibility_policy` 或 disclosure / leak event，例如：

```text
visibility_policy in {inherited, session_only, explicit_disclosure}
```

但 V1 固定采用 inherited visibility。

### 5.5 Contract Transition Functions

contract formal actions 是对 `ContractBook_t` 的状态转移。V1 中每个 transition 都必须校验 actor role、visibility、active session membership、budget 和 state predicates。

V1 的重要安全边界是：agent action payload 只能表达意图、业务条款和 canonical object reference，不能提交 authoritative world-state fields。所有 authoritative fields 都由 控制器 生成、校验或修改。

Agent 可以提交：

```text
terms
contract_id
receiver
reason
note
```

Agent 不可以在 payload 中提交或覆盖：

```text
new contract_id for created objects
parent_contract_id generated by 控制器
created_by
created_at
parties
status
acceptances
signatures
financing.required
financing.status
regulatory.required
regulatory.status
closing_state
visibility_set
history
```

若 agent payload 包含这些 authoritative fields，控制器 必须忽略这些字段并记录 schema violation，或直接判定该 action invalid。V1 推荐 fail-fast：将该 formal action 记为 invalid，不应用状态变化，但仍消耗该 session turn。

#### 5.5.1 propose

```text
contract.propose(terms):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require ValidTerms(terms)
  create C_new in ContractBook
  C_new.contract_id = generated by 控制器
  C_new.parent_contract_id = null
  C_new.created_by = actor
  C_new.created_at = current_time
  C_new.terms = terms
  C_new.parties = {firm_a, firm_b}
  C_new.status = proposed
  C_new.visibility_set = P_s
  C_new.acceptances = {firm_a: null, firm_b: null}
  C_new.signatures = {firm_a: false, firm_b: false}
  C_new.financing.required = FinancingRequired(C_new, omega_t)
  C_new.financing.status = pending if C_new.financing.required = 1 else not_required
  C_new.regulatory.required = RegulatoryRequired(C_new, omega_t)
  C_new.regulatory.status = pending if C_new.regulatory.required = 1 else not_required
  C_new.closing_state.status = not_started
  C_new.closing_state.scheduled_day = null
  C_new.closing_state.checked_at = null
  C_new.closing_state.failure_reasons = []
  C_new.history = [action_id]
```

#### 5.5.2 amend

合同修改不原地覆盖旧合同，而是创建新合同：

```text
contract.amend(C_old, terms_new):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require firm_a in P_s
  require firm_b in P_s
  require actor can reference C_old
  require active_contract(C_old)
  require C_old.status in amendable_status
  require C_old.status != signed or current_day < C_old.closing_state.scheduled_day
  require ValidTerms(terms_new)
  create C_new in ContractBook
  C_new.contract_id = generated by 控制器
  C_new.parent_contract_id = C_old.contract_id
  C_new.created_by = actor
  C_new.created_at = current_time
  C_new.terms = terms_new
  C_new.parties = C_old.parties
  C_new.status = amended
  C_new.visibility_set = visibility_set(C_old, t) ∪ P_s
  C_old.status = superseded
  C_new.acceptances = {firm_a: null, firm_b: null}
  C_new.signatures = {firm_a: false, firm_b: false}
  C_new.financing.required = FinancingRequired(C_new, omega_t)
  C_new.financing.status = pending if C_new.financing.required = 1 else not_required
  C_new.regulatory.required = RegulatoryRequired(C_new, omega_t)
  C_new.regulatory.status = pending if C_new.regulatory.required = 1 else not_required
  C_new.closing_state.status = not_started
  C_new.closing_state.scheduled_day = null
  C_new.closing_state.checked_at = null
  C_new.closing_state.failure_reasons = []
  C_new.history = C_old.history + [action_id]
```

如果任一前置条件失败，控制器 必须把该 turn 记录为 invalid formal action：

```text
action_result = invalid_action
reason in {
  unauthorized_actor,
  missing_transaction_counterparty,
  invisible_contract,
  stale_or_inactive_contract,
  not_amendable_status,
  closing_day_already_reached,
  invalid_terms
}
consume current session turn
apply no state mutation
```

其中 `missing_transaction_counterparty` 表示当前 session 不同时包含 `firm_a` 和 `firm_b`；`stale_or_inactive_contract` 表示 `C_old` 不是该合同族的 latest active contract，或已经处于 `superseded`、`rejected`、`failed`、`closed_success`、`closed_failed` 等不可操作状态。V1 不允许 控制器 自动把 stale `contract_id` 改写为 latest active contract；agent 必须引用正确版本。

V1 的 active session 使用单 actor turn model：每个 turn 只调用一个 active agent，并且只接收一个 `turn_output`。因此，同一 session 内不存在两个 agent 在同一个 turn 同时提交 `contract.amend` 的情形。same-slot logical-parallel sessions 又要求 participants 不重叠，所以两个合法的 `contract.amend` 也不能在同一 slot 中由不同 session 同时作用到同一对交易主体的同一合同族。

Amendment visibility uses inherited visibility in V1:

```text
visibility_set(C_new, created_at)
  = visibility_set(C_old, t) ∪ P_s
```

Thus all agents who could see the old version can also see the amended latest version. This rule keeps the linear contract family as one continuing deal path rather than creating a new information island at every amendment.

这样 控制器 可以保留完整合同历史，后续评估也能追踪 agent 是否记住旧合同、是否合理修改合同。

#### 5.5.3 share and review

```text
contract.share(C, receiver):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require receiver subset P_s
  visibility_set(C, t + 1) = visibility_set(C, t) ∪ receiver

review.request_financing(C):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require investor in P_s
  visibility_set(C, t + 1) = visibility_set(C, t) ∪ {investor}
  C.financing.status = pending if C.financing.required = 1

review.request_regulatory(C):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require regulator in P_s
  visibility_set(C, t + 1) = visibility_set(C, t) ∪ {regulator}
  C.regulatory.status = pending if C.regulatory.required = 1
```

#### 5.5.4 accept, reject, sign

```text
contract.accept(C):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require C.status in {proposed, amended, accepted}
  C.acceptances[actor] = true
  if C.acceptances[firm_a] = true and C.acceptances[firm_b] = true:
    C.status = accepted

contract.reject(C):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require C.status in {proposed, amended, accepted}
  C.acceptances[actor] = false
  C.status = rejected

commit.sign(C):
  require actor in {firm_a, firm_b}
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require C.status in {accepted, signed}
  require C.acceptances[actor] = true
  C.signatures[actor] = true
  if C.signatures[firm_a] = true and C.signatures[firm_b] = true:
    C.status = signed
    C.closing_state.status = pending
    C.closing_state.scheduled_day = closing_day(C, d)
```

`contract.accept` 和 `commit.sign` 不要求合同在当前 `omega_t` 下已经可以 closing。agent 可以接受或签署一个仍需融资、监管批准、披露或后续状态改善的条件性合同。控制器 不在这些动作上提前阻止不可执行合同；不可执行性会在 closing check 中暴露，并在 evaluation 中惩罚。

#### 5.5.5 finance and regulatory commit

```text
commit.finance_commit(C):
  require actor = investor
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require C.financing.required = 1
  require investor_can_commit(C, omega_t)
  C.financing.status = committed
  C.financing.actor = investor

commit.finance_decline(C):
  require actor = investor
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  C.financing.status = declined
  C.financing.actor = investor

commit.approve(C):
  require actor = regulator
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  require C.regulatory.required = 1
  C.regulatory.status = approved
  C.regulatory.actor = regulator

commit.block(C):
  require actor = regulator
  require actor in P_s
  require actor can reference C
  require active_contract(C)
  C.regulatory.status = blocked
  C.regulatory.actor = regulator
```

Investor and regulator actions affect only the corresponding contingent substate:

```text
commit.finance_commit / commit.finance_decline
  mutate C.financing.status and C.financing.actor only

commit.approve / commit.block
  mutate C.regulatory.status and C.regulatory.actor only
```

如果 `commit.finance_commit` 的 `investor_can_commit(C, omega_t)` 前置条件失败，控制器 必须返回 `invalid_action`，消耗该 turn，不应用融资状态变化。V1 中 investor capacity 是 hard feasibility constraint，不是 utility 里的软扣分。

外部事件降低 `investor_available_capital` 后，已有 financing commitment 默认不自动失效；只有当事件包含明确的 `invalidate contract.<contract_id>.financing.status` op，或后续版本定义自动重检规则时，旧 commitment 才会被置为 `declined`。closing check 仍会在 `before attempt_close` 刷新 required flags，并检查当前 financing condition 是否满足。

They do not mutate:

```text
C.terms
C.acceptances
C.signatures
C.status in {proposed, amended, accepted, signed}
```

Thus a contract may be accepted and signed by `firm_a` and `firm_b` even if financing is missing, declined, or regulatory approval is missing / blocked. The world controller does not block signature for that reason; it detects infeasibility at `attempt_close` and assigns `closed_failed` plus evaluation penalties if required financing or regulatory approval is absent or negative.

#### 5.5.6 closing / final execution

closing 是 world-model-only transition，不是 agent 在 session 中直接发起的 action。它是当前合同路径的 hard settlement point：到达 `C.closing_state.scheduled_day` 后，world controller 必须结算该已签署合同；结算成功或失败都会进入 world-level terminal state，并冻结 episode 内 world state。

V1 默认只在 end-of-day controller phase 中调用 `attempt_close`，并且调用顺序固定为 `end_of_day events -> refresh -> attempt_close -> timeout -> snapshot`：

```text
attempt_close_timing = end_of_day_only
```

确定性检查规则为：

```text
at each end_of_day d:
  apply end_of_day events first
  refresh_required_flags for affected active contracts
  for each active signed contract C:
    if d >= C.closing_state.scheduled_day:
      refresh_required_flags(C, omega_t)
      attempt_close(C, omega_t, d)
```

因此，`end_of_day` 事件可以影响同一天的 closing 结果。例如日终信贷市场收紧或监管规则改变，会先进入 world state，再由 `attempt_close` 基于更新后的 `omega_t` 判断合同能否 closing。

在该 day 的 closing 结算之后，不再继续执行后续 agent turn、session scheduling、external event 或 formal action。

```text
attempt_close(C, omega_t, d):
  require active_contract(C)
  require C.status = signed
  require d >= C.closing_state.scheduled_day
  refresh_required_flags(C, omega_t)
  C.closing_state.checked_at = current_time

  if ClosingExecutable(C, omega_t, d):
    C.status = closed_success
    C.closing_state.status = closed_success
    terminal_state = success
  else:
    C.status = closed_failed
    C.closing_state.status = closed_failed
    C.closing_state.failure_reasons = closing_failure_reasons(C, omega_t, d)
    terminal_state = failure
```

可能的 `closing_failure_reasons` 包括：

```text
buyer_budget_infeasible
financing_required_but_not_committed
regulatory_approval_required_but_missing
regulatory_blocked
seller_performance_infeasible
closing_after_deadline
contract_superseded_or_failed
```

V1 默认：

```text
closing_failure_terminal = true
```

因此，closing 成功表示交易完成；closing 失败表示当前 episode 失败。closing 之后不再允许 `contract.amend`、新 session、message 或其他 formal action。world state 不再继续推进，只保留 final snapshot 供 evaluator 读取。若需要模拟 post-closing dispute、重新交易或并购后整合，应作为新 episode 或新 setting，而不是当前合同生命周期中的 amend。

### 5.6 Required-Party Refresh and Cached State

`FinancingRequired(C, omega_t)` 和 `RegulatoryRequired(C, omega_t)` 是基于当前 world state 的实时查询函数。这里的 `omega_t` 表示 控制器 在当前逻辑时刻已经应用完所有 formal action、external event、bookkeeping update 后的状态。因此，同一份合同在不同 `omega_t` 下可能产生不同 required-party 判断。

例如，若 `contract.propose` 时 buyer cash 足以覆盖 `upfront_cash_required(C)`，则：

```text
FinancingRequired(C, omega_t) = 0
```

但如果之后外部事件降低 buyer cash，则在新的 world state 下：

```text
FinancingRequired(C, omega_{t+1}) = 1
```

这不是 bug，而是 Setting 1 的核心 long-horizon / shock 机制：早前看似稳定的合同路径可能因为后续世界状态变化而需要重新融资、重新审批或重新谈判。

合同对象中的 required 字段是缓存状态，而不是最终真值来源：

```text
C.financing.required
C.regulatory.required
```

它们表示最近一次 控制器 refresh 后写入合同对象的派生状态。机制上的 authoritative predicates 始终是：

```text
FinancingRequired(C, omega_t)
RegulatoryRequired(C, omega_t)
```

控制器 必须通过统一函数刷新缓存字段：

```text
refresh_required_flags(C, omega_t):
  financing_required_new = FinancingRequired(C, omega_t)
  regulatory_required_new = RegulatoryRequired(C, omega_t)

  C.financing.required = financing_required_new
  C.regulatory.required = regulatory_required_new
```

刷新融资状态时，控制器 同步维护 `C.financing.status`：

```text
if C.financing.required changes from 0 to 1:
  C.financing.status = pending

if C.financing.required changes from 1 to 0:
  C.financing.status = not_required
```

若 `C.financing.status = committed` 但 refresh 后 `FinancingRequired(C, omega_t) = 1` 仍然成立，则已有 financing commitment 可以继续计为满足 financing condition，除非外部事件或 investor action 明确使该承诺失效。

刷新监管状态时，控制器 同步维护 `C.regulatory.status`：

```text
if C.regulatory.required changes from 0 to 1:
  C.regulatory.status = pending

if C.regulatory.required changes from 1 to 0:
  C.regulatory.status = not_required
```

若 `C.regulatory.status = approved` 但 refresh 后 `RegulatoryRequired(C, omega_t) = 1` 仍然成立，则已有 regulatory approval 可以继续计为满足 regulatory condition，除非外部事件或 regulator action 明确使该批准失效。

控制器 至少在以下时点调用 `refresh_required_flags`：

```text
contract.propose
contract.amend
after_formal_action for affected active contracts
after_event for affected active contracts
post-session / slot_barrier for affected active contracts
before attempt_close for the target signed contract
```

其中 `contract.propose` 和 `contract.amend` 会创建或替换合同条款，因此必须立即计算缓存 required 字段；`after_event` 捕捉外部事件造成的资源、监管规则、市场状态或合同可行性变化；`after_formal_action` 捕捉 session 内 formal action 造成的合同、visibility、financing、regulatory 或 termination-path 变化；`post-session / slot_barrier` 是跨 session / 跨 slot 的兜底同步点；`before attempt_close` 是最终执行校验点，不能依赖过期缓存。

V1 不要求单独的 `before contract.accept` 或 `before commit.sign` semantic refresh hook。`contract.accept` 和 `commit.sign` 使用当前最新 cached required-party state；该 cache 必须已经由 `contract.propose`、`contract.amend`、`after_formal_action`、`after_event` 或 slot-barrier bookkeeping 刷新。实现可以在 accept / sign 前做幂等的 defensive refresh，但它不构成独立的机制时点，也不能触发 closing 或其他额外 transition。

### 5.7 Contract Lifecycle and Linear Versioning

V1 中，同一个交易路径下的合同版本是线性的。`contract.amend` 不允许 fork，也不允许从 rejected、superseded 或 closed-success 合同中复活旧路径。

合同版本关系由 `parent_contract_id` 记录：

```text
C_root -> C_1 -> C_2 -> ... -> C_latest
```

每次 `contract.amend(C_old, terms_new)` 都必须以当前合同族的 latest active contract 为输入：

```text
require active_contract(C_old)
```

若 `active_contract(C_old) = false`，该 action 必须返回：

```text
action_result = invalid_action
reason = stale_or_inactive_contract
consume current session turn
apply no state mutation
```

控制器 不得自动查找并替换为 `latest_active_contract_in_family(C_root)`。自动替换会掩盖 agent 对合同版本的误引用，而版本跟踪本身是 Setting 1 V1 要评估的长程状态能力之一。

其中：

```text
latest_contract_in_family(C) =
  the unique contract C_latest such that
    C_latest is in the same parent-child chain as C
    and no other contract has parent_contract_id = C_latest.contract_id
```

active contract 定义为：

```text
active_contract(C) iff
  C = latest_contract_in_family(C)
  and C.status notin {superseded, rejected, failed, closed_success, closed_failed}
```

等价地说，latest contract 只表示合同版本链上没有 child 的最新版本；active contract 进一步要求该最新版本仍处于可操作状态。`closed_success` 和 `closed_failed` 都不是 active contract，因为 closing 是 hard settlement point，结算后 episode 已经终止。

对于一个合同族，也可以直接定义：

```text
latest_active_contract_in_family(C_root) =
  C_latest if active_contract(C_latest)
  else null
```

若返回 `null`，表示该合同族已经没有可继续操作的当前版本。例如最新版本已经 `rejected`、`failed`、`closed_success` 或 `closed_failed`。

因此，如果 `C_1` 已经被 `C_2` 替代：

```text
C_1.status = superseded
C_2.status = amended
```

agent 不能再对 `C_1` 执行 `contract.amend`、`contract.accept`、`commit.sign`、`review.request_*` 或 `commit.*`。后续动作必须引用 `C_2`。

V1 中，所有改变合同状态、围绕合同产生承诺、或结算合同的动作 / transition 都必须作用在 latest active contract 上：

```text
contract.amend
contract.share
review.request_financing
review.request_regulatory
contract.accept
contract.reject
commit.sign
commit.finance_commit
commit.finance_decline
commit.approve
commit.block
attempt_close
```

这些 action 还需要满足各自动作的状态前提。例如 `active_contract(C)` 只说明 `C` 是当前可操作版本；`contract.accept(C)` 仍额外要求 `C.status in {proposed, amended, accepted}`，`commit.sign(C)` 仍额外要求 `C.status in {accepted, signed}`。

可被 amend 的旧合同状态为：

```text
amendable_status = {
  proposed,
  amended,
  accepted,
  signed
}
```

不可被 amend 的旧合同状态为：

```text
not_amendable_status = {
  closed_success,
  closed_failed,
  superseded,
  rejected,
  failed
}
```

各状态的生命周期语义如下：

```text
proposed:
  新提案，等待接受、拒绝或修改。

amended:
  由旧版本派生的新版本，等待重新接受、拒绝或继续修改。

accepted:
  firm_a 和 firm_b 已接受当前版本，但尚未完成签署。

signed:
  firm_a 和 firm_b 已签署当前版本，但 closing 尚未成功。
  signed 在 `current_day < C.closing_state.scheduled_day` 时仍可 amend，用于支持 shock 后、closing 前的 renegotiation。

closed_failed:
  当前版本签署后 closing 失败。
  它只由 `attempt_close(C, omega_t, d)` 产生，不由 agent 直接选择。
  closing 失败即 world-level failure，不可 amend。

closed_success:
  closing 成功，交易完成，不可 amend。

superseded:
  已被新版本替代，不可再作为任何后续合同动作的目标。

rejected:
  某个交易主体通过 `contract.reject(C)` 主动拒绝当前合同版本。
  它是 agent-level refusal，不是 控制器 对合同可执行性的判定。
  rejected contract 不可 amend 复活。
  若双方要继续谈判，应通过新的 contract.propose 创建新合同路径。

failed:
  控制器 判定该合同版本、合同路径或 contract family 不可继续，不可 amend。
  它不是交易主体主动拒绝某个版本，也不是 signed contract 在 closing 时执行失败。
  failed 通常用于规则性终止、路径失效、required-party 路径不可恢复、或系统级不可继续状态。
```

`contract.amend` 生成的新版本不会继承旧版本的 acceptance、signature、financing commitment 或 regulatory approval，除非某个字段被 控制器 显式标记为可迁移。V1 默认不迁移这些状态：

```text
C_new.acceptances = {firm_a: null, firm_b: null}
C_new.signatures = {firm_a: false, firm_b: false}
C_new.financing.status = not_required | pending
C_new.regulatory.status = not_required | pending
```

这意味着：

- amend accepted contract 后，双方必须重新 `contract.accept(C_new)`。
- amend signed contract 必须发生在 scheduled closing day 之前；amend 后双方必须重新 `contract.accept(C_new)` 和 `commit.sign(C_new)`。

最终成交版本定义为：

```text
final_contract = C iff C.status = closed_success
```

失败或 timeout episode 中，evaluator 可以记录：

```text
terminal_contract = latest_contract_in_family(C_root)
latest_active_contract = latest_active_contract_in_family(C_root)
last_signed_contract = latest contract in family with status reached signed
last_closed_failed_contract = latest contract in family with status = closed_failed
```

其中 `terminal_contract` 记录合同族最后停在哪个版本，哪怕它已经不可操作；`latest_active_contract` 记录 terminal state 之前是否还有可继续谈判或补救的合同版本。这些记录用于区分 agent 是完全没有达成协议，还是签署了不可执行合同，或在 shock 后没有及时修改旧合同。

## 6. Action Space 与 Budget

### 6.1 Action Space 总体定义

设完整 action space 为：

```text
X = X_sched ∪ X_msg ∪ X_formal ∪ X_ctrl ∪ X_pass
```

其中：

- `X_sched`：scheduling action，用于形成 session。
- `X_msg`：message action，用于 session 内沟通。
- `X_formal`：formal action，用于改变合同状态或世界状态。
- `X_ctrl`：session control action，用于控制 session 本身。
- `X_pass`：active session 中的空行动。

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

- `action_class in {sched, msg, formal, ctrl, pass}`
- `action_type` 是该 action class 下的有限枚举。
- `receiver subset A`，但 session 内 action 还必须满足 `receiver subset P_s`。
- `content` 是该 action 的主要业务内容。
- `meta` 是 routing、visibility、object reference、audit flag 等控制信息。

在 active session turn 中，agent 每次被调用只能输出一个 action class：

```text
action_class in {msg, formal, ctrl, pass}
```

因此，session turn 中不能同时输出 message 和 formal action，也不能同时输出 formal action 和 session control action。若 agent 想先解释再执行正式动作，必须在不同 turn 中完成。

### 6.2 Scheduling Action Space

Scheduling action space 定义为：

```text
X_sched = X_request ∪ X_response
```

包括：

- `session_request`
- `session_response`

它们只用于形成 session，不直接改变合同状态，不属于 session turn，也不计入 formal action count。

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

v1 中：

```text
Q_i = 1
```

如果 agent 在 invitation round 没有产生可用的 `session_request`，则该 agent 在本 round 视为未发起邀请：

```text
raw_request_i = no_request
effective_request_i = null
```

`no_request` 的触发条件包括但不限于：

- LLM / tool call 超时。
- LLM / tool call 抛出异常或返回空输出。
- 输出无法解析为合法 action schema。
- `action_class != sched`。
- `action_type != session_request`。
- `proposed_participants` 缺失、不是合法 agent set、未包含 requester、人数小于 2，或超出 `A`。
- `purpose` 缺失、不是 text，或违反 request payload 约束。
- 同一个 invitation round 中提交多个 request 且无法按 schema 唯一化。

控制器 不等待、不重试、不追加 scheduling round。`no_request` 写入 log / trajectory，但不形成 session request。

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
count_distinct_accepted_participant_sets_i(k) <= 1
```

其中 `count_distinct_accepted_participant_sets_i(k)` 表示 agent `i` 在 session slot `k` 接受的不同 participant set 数量。若 agent 接受的 incoming request 与自己 outgoing request 的 participant set 不同，自己发起的 outgoing request 会按 [Response Round](#33-response-round) 的 slot commitment 规则自动撤回。

如果 response round 输出不满足 schema 或调度约束，控制器 统一规范化为：

```text
raw_response_i(r) = no_response
effective_response_i(r) = decline
```

`no_response` 覆盖调用超时、调用异常、空输出、schema 解析失败、非 `sched` action、非 `session_response` action、`request_id` 无效、`decision` 非法、以及 agent 瞎回无法映射为 `accept | decline` 的情况。该结果写入 log / trajectory，但不触发重试或额外 scheduling round。

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

v1 中 session 内 message 默认对当前所有 participants 可见，不再细分 session 内 public/private。因此 `receiver` 可以写作：

```text
receiver = ALL_IN_SESSION
```

一个 `msg` turn 恰好包含一个 message payload：

```text
turn_output = {
  action_class: msg,
  payload: {
    content: text
  }
}
```

如果 agent 想表达多个理由、条件或解释，它仍然写在同一个 `content` 字段中；控制器 不把同一 `msg` turn 内的多句文本拆成多条 message。

V1 不设置独立的 per-session message budget `M_max`。message 的上限由统一行动轮次决定：

```text
M_i(s) <= K_i(s) <= K_s
```

因此 message 不计入 formal-action counters，但每条 message 必须消耗一次 session turn。

### 6.3.1 Pass Action

`pass` 表示 active agent 在当前 turn 不发送 message，也不执行 formal action 或 session control action。

```text
X_pass = {pass}
```

```text
x in X_pass:
  action_class = pass
  action_type = pass
  receiver = WORLD
  content = null
  meta = {
    session_id
  }
```

`pass` 只消耗一个 session turn，不改变合同、资源、participant set 或 terminal state。

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

`terminate.session` 不属于 `X_formal`。它属于 [Session Control Action Space](#65-session-control-action-space)，只能在 active session 内由当前 session participant 调用，并消耗一次统一 session turn。

Formal action 必须单独占用一个 session turn。执行 formal action 的同一 turn 中不能附带 message；如果 agent 需要解释该 formal action 的理由，应在前一个或后一个 message turn 中表达。

Formal action payload 只表达 agent 的 intent。Agent 可以引用 `contract_id`、提交 `terms`、指定 `receiver` 或给出 `reason`，但不能提交完整 canonical world objects。控制器 是唯一可以创建合同 ID、设置合同状态、修改 visibility、写入签名、更新 financing/regulatory substate、安排 closing 和写 history 的主体。

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
  terms: {
    valuation,
    payment,
    closing,
    compliance,
    penalty
  },
  note optional
}
meta = {
  session_id
}
```

`contract.amend`:

```text
content = {
  contract_id,
  terms: {
    valuation,
    payment,
    closing,
    compliance,
    penalty
  },
  note optional
}
meta = {
  session_id
}
```

`contract.amend` 的 payload 只表达新条款意图。控制器 必须额外校验：`actor in {firm_a, firm_b}`，当前 session 同时包含 `firm_a` 和 `firm_b`，`contract_id` 指向 latest active contract，且 actor 能看见该合同。若校验失败，返回 `invalid_action`，消耗当前 turn，不应用状态变化。

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
  reason,
  target_contract_id optional
}
meta = {
  session_id,
  final: true
}
```

`terminate.negotiation` 表示 agent 退出交易谈判或当前审查路径。控制器 根据执行者角色决定其后果：

```text
if actor in {firm_a, firm_b}:
  terminal_state = terminated_by_agent
else if actor = investor:
  mark investor as withdrawn from current financing path
else if actor = regulator:
  mark regulator as withdrawn from current regulatory review path
```

因此，investor 或 regulator 的 `terminate.negotiation` 不必然终止整个世界，但会影响当前合同路径是否还能满足 contingent required conditions。

#### 6.4.5 Termination Application Semantics

`terminate.negotiation` 是 formal action，因此它必须单独占用一个 session turn。该 turn 不能同时包含 message。如果 agent 想解释退出理由，需要在之前的 message turn 中说明；若它直接执行 `terminate.negotiation`，控制器 只记录 formal action 的结构化 `reason`。

控制器 对 `terminate.negotiation` 的应用顺序为：

```text
1. validate actor, session membership, schema, budget, and target path
2. if valid, apply current-session or current-path termination effect immediately
3. write action log and state delta
4. if actor is a transaction party, mark pending_world_terminal
5. close affected session/path as required
```

若执行者是交易主体：

```text
if actor in {firm_a, firm_b}
and terminal_state is not already set:
  pending_world_terminal = {
    terminal_state: terminated_by_agent,
    actor,
    action_id,
    reason
  }
```

该动作对当前 session 立即生效，但 world-level terminal state 在 slot barrier / post-session bookkeeping 时提交。这样可以保持同一 slot 内互不重叠 sessions 的 logical-parallel 语义。

- 当前 session 立刻结束。
- 同一 slot 中其他已经形成的 active sessions 不被回滚，也不被中断。
- 其他同 slot sessions 已发生或后续发生的 turn、message、formal action 和 state delta 保留在 log 中。
- slot barrier 会先提交本 slot 的 logs / deltas，再把 `pending_world_terminal` 提升为 `terminal_state`。
- 进入 world-level terminal state 后，后续 scheduling window、session、message、formal action、新合同签署、end-of-day event 和 closing check 都不再发生。
- post-session bookkeeping 仍会运行，但只负责关闭 session、写入 log、snapshot 和 terminal reason，不再调用 agent 产生新 action。

若执行者是 `investor`，则退出范围默认是当前 financing path，而不是永久退出整个世界：

```text
if actor = investor:
  target_path = target_contract_id if provided else current_visible_financing_path(actor, session)
  mark investor as withdrawn from target_path
  if target_path.financing.required = 1:
    target_path.financing.status = declined
```

若执行者是 `regulator`，则退出范围默认是当前 regulatory review path，而不是永久退出整个世界：

```text
if actor = regulator:
  target_path = target_contract_id if provided else current_visible_regulatory_path(actor, session)
  mark regulator as withdrawn from target_path
  if target_path.regulatory.required = 1:
    target_path.regulatory.status = blocked
```

V1 默认：

```text
withdrawal_scope = current_contract
```

因此，investor / regulator 对合同 `C` 执行 `terminate.negotiation` 后，`C` 的融资或监管路径失败；但如果 buyer 和 seller 后续通过 `contract.amend` 创建新合同 `C_new`，控制器 可以允许重新执行 `review.request_financing(C_new)` 或 `review.request_regulatory(C_new)`，除非配置显式把 withdrawal scope 扩展为 `contract_family` 或 `global`。

`terminate.negotiation` 是幂等的：

```text
if actor in {firm_a, firm_b}
and terminal_state is already set:
  action_result = not_applied_due_to_prior_terminal_state
  state unchanged

if actor = investor
and investor already withdrawn from target_path:
  action_result = no_op_already_withdrawn
  state unchanged

if actor = regulator
and regulator already withdrawn from target_path:
  action_result = no_op_already_withdrawn
  state unchanged
```

所有幂等 no-op 都写入 action log，便于 replay 和 evaluator 区分“没有发生”与“重复尝试但状态不变”。

V1 的 session 内 formal action 上限由统一行动轮次控制。formal action 发生在 active session 内，并且单独占用一个 session turn。session 结束后，agent 不再主动执行 formal action，除非后续明确引入 post-session action。

V1 不设置额外的 per-session formal-action budget `H_max`：

```text
H_max = disabled
```

每个 formal action 都满足：

```text
K_i(s) increases by 1
```

formal action count