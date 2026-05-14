# `sotopia.settings` 说明

本目录存放**按场景拆分**的世界逻辑与谈判运行时代码（独立于 `sotopia.envs` 中的并行 gym 风格环境）。当前主要场景为 **长期商业谈判**（设计见仓库内 `design_1.md` 等文档）。

## 目录结构

| 路径 | 作用 |
|------|------|
| `__init__.py` | 从 `long_term_negotiation` 再导出规则侧常用符号（环境、控制器、指标、dummy agent 等）。**不**在顶层导出 LLM 评测入口，避免仅跑规则路径时拉起 LLM 依赖。 |
| `long_term_negotiation/` | 谈判世界实现：时间线、调度、会话、动作解析、指标、外部事件、LLM 参与者与评测。 |

## 长期谈判子包 `long_term_negotiation/`（文件速览）

| 文件 | 职责 |
|------|------|
| `types.py` | 时间线参数、阶段枚举、合同与会话相关数据类型。 |
| `roles.py` | 规范角色名、资源 bundle、roster 校验。 |
| `controller.py` | `NegotiationWorldController`：日程、会话轮次、动作日志与终止条件。 |
| `env.py` | `LongTermNegotiationEnv`：宏观相位循环，调用各参与者的 `aact`，对接 `SystemState` / 事件引擎。 |
| `negotiation_metrics.py` | 规则向标量指标 `compute_negotiation_rule_metrics`（已合并 `compute_negotiation_final_state_metrics`，含中间状态最后一帧的 `negotiation_final_state_score`）。 |
| `dummy_agents.py` | 规则策略 dummy，用于无 LLM 冒烟或对照实验。 |
| `negotiation_llm_agent.py` | `NegotiationSocialLLMAgent` 与 `build_negotiation_social_llm_agents`（结构化 JSON 动作）。 |
| `llm_evaluation.py` | **单次** LLM episode + 可选终局 `EpisodeLLMEvaluator`：`run_llm_negotiation_episode_evaluation`；可选 ``negotiation_run_config``。 |
| `batch_evaluation.py` | **批量**异步调度、并发限流、可 JSON 序列化的 episode 记录：`run_long_term_negotiation_eval_batch`；支持 ``scenario_environment_pks``、``negotiation_run_config``。 |
| `negotiation_run_config.py` | JSON 运行配置：``load_negotiation_run_config`` / ``build_negotiation_agents_from_run_config``（选用谈判 LLM Agent 变体与 ``memory.backend``：plain / summarizing）。 |
| `eval_logging.py` | CLI / 程序化侧统一配置：`configure_negotiation_cli_logging`、`episode_*_line`、`sotopia.negotiation.batch` logger。 |
| `scenario_loader.py` | ``build_negotiation_game_metadata_bundle``；从库里还原 ``NegotiationTimelineParams``。 |
| `external_events.py` | 外部事件规格与 runner。 |
| `agent_state_variables.py` | 心理/状态变量与 prompt 附加段。 |
| 其余 | 调度解析、会话 roster 分类等支撑模块。 |

## 对话风格（数据合成与终局 LLM 评测）

- **数据合成（`generate_long_term_negotiation_llm.py`）**：每条 `agenerate_env_profile` 的 inspiration 末尾会追加英文块
  `DIALOGUE_STYLE_SYNTHESIS_APPEND_EN`（定义于 `scenario_loader.py`），要求 `scenario` / `agent_goals` 写明各角色**可区分的口语身份**
  （语体、节奏、口头禅/避讳、拒绝与让步的说话方式），且随场景类型（菜场/合作竞争/调度等）调整语气，避免所有人同一套「公文腔」。
- **`game_metadata.dialogue_style`**：`build_negotiation_game_metadata_bundle` 写入 `version`、`synthesis_requirements_en`、
  `evaluation_requirements_en`（与上述常量同源）。旧 manifest 若无该字段，评测侧仍使用代码内默认 rubric。
- **终局评测（`llm_evaluation.format_negotiation_episode_for_llm_eval`）**：在送入 `EpisodeLLMEvaluator` 的历史文本**最前**插入
  「Dialogue-style rubric」，指导将**对话风格执行**并入 `SotopiaDimensions` 的 believability / knowledge / goal 等维度
  （惩罚多人同质化措辞、或与场景承诺的声线严重不符）。

### 各角色说话示例（参考，造数 / 写 `agent_goals` 时可模仿其节奏与用词，勿整段照抄）

以下示例中的人名为占位；实际 episode 以 Environment 里的人名为准。同一桌**各角色应用不同节奏**（一人短促、一人好铺垫等）。

**公司侧（`firms_only`：firm_a / firm_b / firm_c / firm_d）**

- **firm_a（偏谨慎、条款导向）**：短句、先确认再承诺。  
  - 例：「周三前我只能给到这批量；再往上走得单独签补充条款，我今天拍不了板。」  
  - 例：「你把违约那条改成‘延迟超过 48 小时’再说，现在这写法我们法务不会过。」
- **firm_b（偏关系、缓冲词多）**：先寒暄再入题，常用「咱们」「说实话」。  
  - 例：「说实话我也难办，但你要是能把账期缩到三十天，我回去跟老板也好说话。」  
  - 例：「咱们先对齐一下交付窗口吧，价格我可以再让半步，前提是你们那边别临时改规格。」
- **firm_c（防守、强调风险）**：爱用「底线」「不能再退」。  
  - 例：「这已经是底线价了；再压我就只能撤标，后面排队的还有两家。」
- **firm_d（快攻、抢窗口）**：节奏快、敢要结论。  
  - 例：「今天就定：要么签这个数，要么我转去跟 B 家谈，他们刚松口。」

**菜场/个体竞争（`wet_market_competition`：摊主 vs 买主）**

- **摊主甲（热情、感官词）**：  
  - 例：「这排肋条今早刚到的，你摸一下油花；要半扇我给你片薄点，炖汤更出味。」  
  - 例：「隔壁喊得凶，我这斤两只多不少，你回去上秤不对你来找我。」
- **摊主乙（冷淡、靠老客）**：  
  - 例：「就这个价，不还价；要就称，不要我让给后面排队。」
- **买主/团长（砍价、算总账）**：  
  - 例：「你两家加起来还比楼下团购贵五块呢，要么各让一点我一起拿。」

**资源调度（`resource_scheduling_management`）**

- **排班协调人**：时间窗、责任切分。  
  - 例：「冷库 4–6 点已经锁给水产了；你们要么改到七点档，要么自己找临时库，我这边签不了冲突单。」  
  - 例：「车回来晚了一小时，下一棒装卸顺延，谁有意见现在提，写进纪要里。」
- **被调度方（抱怨但讲条件）**：  
  - 例：「顺延可以，但超时的罚金不能算我们头上，昨晚你们装车就晚了。」

**机构侧（`with_institutional`：仅部分阵容出现）**

- **investor（融资评审）**：正式、问现金流与抵押。  
  - 例：「在现有现金流假设下，请说明若回款延迟两周，你们计划如何覆盖缺口；否则我们无法进入 commit 流程。」
- **regulator（合规）**：中性、引用规则而非情绪。  
  - 例：「若条款保留该促销表述，请补充对误导性宣传的免责声明，否则监管路径上我会标注为需修订。」

各模块内仍有更细的 docstring；评测主链路的「文件 → 函数 → 顺序」以 `llm_evaluation.py`、`batch_evaluation.py` 及 CLI `sotopia/cli/benchmark/negotiation_batch.py` 的模块说明为准。

## 运行 LLM 评测 / 生成汇总 JSON：调用顺序

### 1. 命令行（推荐）

- 安装可执行入口：`pip install -e .`（在 `social_env` 根目录），主命令名为 **`sotopia`**（见 `pyproject.toml` 的 `[project.scripts]`）。
- 子命令 **`negotiation-batch`** 在 Typer 中注册；若本机显示为根级选项，以 `sotopia --help` 为准。
- 实现文件：`sotopia/cli/benchmark/negotiation_batch.py`，入口函数 **`negotiation_batch`**。

顺序概括：

1. `negotiation_batch` 解析参数；无场景 pk 时构造默认 `NegotiationTimelineParams`，有场景 pk 时单场参数由存储层提供。参与者数量默认：无场景时由 `--quartet`（2 或 4）或显式 `--num-participants`；有场景时读 `game_metadata.num_participants`（缺省由 `quartet` 推断），CLI 的 `--num-participants` 可覆盖每场。
2. 调用 `sotopia.settings.long_term_negotiation.batch_evaluation.run_long_term_negotiation_eval_batch`。
3. 内部 `run_long_term_negotiation_eval_batch_async` 对每个 `(agent_model, repeat)` 限流并发，单任务调用 `run_llm_negotiation_episode_evaluation`。
4. 汇总后可选 `-o/--output`：写入 **带时间戳的独立 JSON 文件**（缩进格式；根级含 `aggregate_means` 与 `rows`）。

**``--run-config``（选用 Agent / 记忆后端）**

- 传入指向 JSON 的路径，例如 ``--run-config sotopia/settings/long_term_negotiation/run_config_examples/summarizing_memory.json``。
- 字段语义见 ``negotiation_run_config.py`` 模块 docstring；示例 JSON 在同目录 ``run_config_examples/`` 下。
- 不传时与历史行为一致（plain 滑动窗口记忆）。传入时，每条 `rows` 里的 episode 记录会多一个 ``negotiation_run_config`` 字段便于复现实验。

**日志与可读输出（与 `--output` 汇总 JSON 区分）**

- **`--output` 汇总 JSON**：一次 batch run 一个文件；含 `run_started_at` / `run_timestamp` / `tag` / `agent_models` / `evaluator_model` / **`aggregate_means`**（跨 episode 数值均值等）/ **`rows`**（每条 episode 一条记录：`terminal`、`rule_metrics`、`rule_evaluation_state`、`llm_aggregate`、`llm_dimension_scores`、场景 pk 等）。
- **stderr 进度条**：批量跑时用 `tqdm` 显示完成进度与速率，不写进 `-o`，也不写入 `--log-file`。
- **文本 log（可读）**：`--print-logs` 用 Rich 在控制台打出 INFO；`--log-file PATH` 向该路径 **追加** UTF-8 纯文本行（格式 `时间 | LEVEL | logger名 | message`），每条 episode 有 `episode_start` / `episode_done`（及失败时的 traceback）。配置入口：`long_term_negotiation.eval_logging.configure_negotiation_cli_logging`。
  - 若不传 `--log-file`，CLI 会自动落盘到 `logs/negotiation_batch_YYYYMMDD_HHMMSS.log`（文件名显式带运行时间）。
  - 启动时会打印 `[negotiation-batch] run_started_at=... log_file=...` 便于追踪本次运行日志文件。
  - 第三方库（如 LiteLLM）在 INFO 仍可能刷屏，可按需调高其日志级别。
- **程序化**：若直接从 Python 调用 `run_long_term_negotiation_eval_batch*`，同样需要自行对上述函数调用一次以保持与 CLI 行为一致；传入 `negotiation_run_config=load_negotiation_run_config(Path("..."))` 与 CLI `--run-config` 等价；每场模型 I/O 可传 `model_trace_dir=` 或仅 `execution_trace_dir=`（后者作 JSONL 根目录）；单次评测见 `run_llm_negotiation_episode_evaluation` 的 `model_trace_dir` / `execution_trace_dir` / `write_execution_record` 参数。

#### 落盘文件分别是什么（`negotiation-batch` / 单次 `run_llm_negotiation_episode_evaluation`）

以下以「一次 episode」为粒度说明；路径常含 `--artifact-root` 或默认嵌套的 `{agent_model}/{run_timestamp}/`（见 CLI `--trace-flat`）。

| 产物 | 文件名规律（示例） | 内容是什么 |
|------|-------------------|------------|
| **批量汇总结果** | `negotiation_eval_<tag>_<YYYYMMDD_HHMMSS>.json`（`--output` 为目录时）或 `<stem>_<时间戳>.json` | 整次 run 一条 JSON：`aggregate_means` + `rows[]` 每条 episode 的规则分、可选 LLM 聚合、终局状态快照等。 |
| **文本运行日志** | `logs/negotiation_batch_<时间戳>.log`（或 `--log-file`） | 人类可读的 INFO 行：`episode_start` / `episode_done` 摘要；不是结构化评测结果本体。 |
| **全局执行档案（可选）** | `<experiment_tag>.execution.json` | 仅 ``write_execution_record=True`` 时写入：单场完整世界态 + 可选合并的 ``llm_model_traces``。 |
| **纯文本复盘稿（可选）** | 与上同 stem 的 `<experiment_tag>.execution.transcript.txt` | 同上，人类可读稿。 |
| **按角色合一档案（可选）** | `<execution_stem>_<firm_a>.agent_episode.json` 等 | 仅 ``write_execution_record=True`` 时：每角色 inbox 子集 + 日志子集 + 该角色 LLM 行。 |
| **逐次 LLM 原始 trace（默认主产物）** | `<model_trace_stem>_<展示名或id>.jsonl` | **每次** `agenerate` 等一行 JSON：`full_rendered_prompt`、`raw_model_content`、`parsed`、`input_values` 等；按 `input_values["agent"]` 分文件。 |
| **无 agent 桶** | `<stem>_no_agent.jsonl` | 某次生成未带 `agent` 字段的 LLM 调用，单独成文件。 |
| **终局 LLM 评测** | `<stem>_terminal_evaluator.jsonl` | `EpisodeLLMEvaluator` 的 `agenerate` 调用（与参与者 trace 同字段）。 |

无安装时用模块方式（需将 `social_env` 加入 `PYTHONPATH`）可参考该文件末尾的 `python -m sotopia.cli.benchmark.negotiation_batch` 说明。

**从存储加载题库场景（与 `benchmark_v2_data_models` 生成的 `EnvironmentProfile` 对齐）**：先用
`scripts/generate_long_term_negotiation_scenarios.py` 写入 `~/.sotopia/data/`，再在执行时传
`--scenario-manifest ~/.sotopia/data/long_term_negotiation_manifest.json` 和/或多次
`--scenario-env-pk <pk>`。此时每条 episode 会从 profile 的 ``game_metadata.timeline`` 构造
`NegotiationTimelineParams`，参与者人数由各场景的 ``game_metadata.num_participants``（或 ``quartet`` 推断）决定；CLI 的 ``--quartet`` 对场景路径仅作提示，可用 ``--num-participants`` 覆盖。

若需 **用大模型写好 scenario / agent_goals 文本**：运行 `scripts/generate_long_term_negotiation_llm.py`
（manifest 默认为 ``~/.sotopia/data/long_term_negotiation_llm_manifest.json``），命令行同上把 manifest 换成该路径即可。

### 测试动作协议（与场景类型关系）

当前 LLM 造数默认覆盖三类场景（`business_coopetition`、`wet_market_competition`、`resource_scheduling_management`），但**测试时动作协议不按场景切换**，而是由运行时 phase 决定（`NegotiationWorldController.observation_for_*`）。

结论：

- 三类场景在测试时使用**同一套动作空间/JSON 协议**；
- 场景类型主要影响：`scenario` 文本、`game_metadata`、提示语语义与目标压力；
- 真正允许的动作由当前 phase 的 `available_actions` 与 action_instruction 决定。

#### 按 phase 的 `available_actions`

| phase | `available_actions` |
|------|----------------------|
| `SCHEDULE_INVITE` | `["speak", "action", "none"]` |
| `SCHEDULE_RESPONSE` | `["speak", "action", "none"]` |
| `SESSION` | `["speak", "non-verbal communication", "action", "none", "leave"]` |
| 无 active session 的兜底 | `["none"]` |

#### `action_type="action"` 时可提交的 JSON（核心）

通用规则（所有 `action` JSON 共享）：

- `negotiation_op`：操作类型（必填，字符串），决定本次动作会被哪条处理逻辑消费。
- 参与者字段（如 `requester`、`proposed_participants`）统一使用**个人名**（不是 `firm_*`）。
- 布尔字段（如 `accept`）必须是 JSON 布尔值 `true/false`，不要写字符串 `"true"`。
- 未在当前 phase 的白名单内的 `negotiation_op` 会被判定为无效动作。

**1) 邀约阶段 `SCHEDULE_INVITE`**

- `{"negotiation_op":"session_request","proposed_participants":[...],"purpose":"..."}`
- `{"negotiation_op":"sched_pass"}`

参数说明：

- `session_request`
  - `negotiation_op`：固定为 `"session_request"`（必填）。
  - `proposed_participants`：拟邀约参与者名单（必填，字符串数组，建议至少 1 人）。
  - `purpose`：本次会话目的（必填，字符串；建议写清议题，如“讨论交付排期与违约条款”）。
- `sched_pass`
  - `negotiation_op`：固定为 `"sched_pass"`（必填）。
  - 含义：本轮不发起邀约，直接让出调度动作。

**2) 应答阶段 `SCHEDULE_RESPONSE`**

- `{"negotiation_op":"session_response","requester":"<个人名>","accept":true|false}`
- `{"negotiation_op":"session_response_batch","responses":[{"requester":"<个人名>","accept":...}, ...]}`
- `{"negotiation_op":"sched_pass"}`

参数说明：

- `session_response`
  - `negotiation_op`：固定为 `"session_response"`（必填）。
  - `requester`：邀约发起人个人名（必填，字符串）。
  - `accept`：是否接受该邀约（必填，布尔）。
- `session_response_batch`
  - `negotiation_op`：固定为 `"session_response_batch"`（必填）。
  - `responses`：批量应答列表（必填，数组）。
  - `responses[i].requester`：该条邀约的发起人（必填，字符串）。
  - `responses[i].accept`：是否接受该条邀约（必填，布尔）。
- `sched_pass`
  - `negotiation_op`：固定为 `"sched_pass"`（必填）。
  - 含义：本轮不对邀约作出实质应答（通常不推荐，除非策略上需要拖延）。

**3) 会话阶段 `SESSION`（formal / control）**

- `formal`：
  - `propose_contract`
  - `accept`
  - `reject_contract`
  - `amend_contract`
  - `request_financing_review`
  - `request_regulatory_review`
  - `contract_share`
  - `sign`
  - `finance_commit` / `finance_decline`
  - `regulatory_approve` / `regulatory_block`
- 全局终止：
  - `{"negotiation_op":"terminate_negotiation"}`
- 会话控制：
  - `{"negotiation_op":"session_control","verb":"leave"}`
  - `{"negotiation_op":"session_control","verb":"terminate_session"}`

参数说明（核心字段）：

- `propose_contract`
  - `negotiation_op`：固定 `"propose_contract"`（必填）。
  - 常见附加字段（按环境提示）：
    - `counterparty`：对手方（字符串）；
    - `terms`：条款对象（如价格、数量、交付日、违约责任）；
    - `rationale`：提案理由（字符串）。
- `amend_contract`
  - `negotiation_op`：固定 `"amend_contract"`（必填）。
  - 常见附加字段：`delta_terms`（要修改的条款）、`reason`（修改原因）。
- `accept` / `reject_contract` / `sign`
  - `negotiation_op`：分别固定为 `"accept"` / `"reject_contract"` / `"sign"`（必填）。
  - 含义：接受草案、拒绝草案、签署合同。
- `request_financing_review` / `finance_commit` / `finance_decline`
  - 与融资方相关的评审与承诺/拒绝动作；`negotiation_op` 固定为对应值（必填）。
- `request_regulatory_review` / `regulatory_approve` / `regulatory_block`
  - 与监管方相关的评审与批准/阻断动作；`negotiation_op` 固定为对应值（必填）。
- `contract_share`
  - `negotiation_op`：固定 `"contract_share"`（必填）。
  - 含义：在会话中广播或转发当前合同草案给相关方。
- `terminate_negotiation`
  - `negotiation_op`：固定 `"terminate_negotiation"`（必填）。
  - 含义：终止整场谈判进程（全局终止）。
- `session_control`
  - `negotiation_op`：固定 `"session_control"`（必填）。
  - `verb`：控制指令（必填），当前支持：
    - `"leave"`：离开当前会话；
    - `"terminate_session"`：结束当前会话（不一定终止整场谈判）。

最小可用示例（可直接参考）：

- 发起邀约：`{"negotiation_op":"session_request","proposed_participants":["Riley Carter","Jordan Hayes"],"purpose":"讨论供货价格与交付节点"}`
- 单条应答：`{"negotiation_op":"session_response","requester":"Riley Carter","accept":true}`
- 批量应答：`{"negotiation_op":"session_response_batch","responses":[{"requester":"Riley Carter","accept":true},{"requester":"Jordan Hayes","accept":false}]}`
- 会话离开：`{"negotiation_op":"session_control","verb":"leave"}`

> 备注：JSON 里的参与者字符串应使用当回合 Environment 提示中的**个人名**；控制器会做人名到 canonical role 的归一化。

**批量造数与规模/要求（两脚本）**

- 手写规则场景 ``generate_long_term_negotiation_scenarios.py``：``--modes``（bilat / tri / quartet / firms3 / firms4）、``--timeline-labels`` 筛选时间轴预设、``--replicates`` 扩大每种组合份数、``--requirements`` 写入 manifest 的 ``generation_spec``。
- LLM 场景 ``generate_long_term_negotiation_llm.py``：``--n`` 与 ``--concurrency`` 控制条数与并发；``--modes`` 按顺序对每条 LLM 结果轮转人数模式（含 firms3 / firms4）；``--timeline-labels``、``--requirements`` 同上写入 manifest。
  - **AgentProfile 默认对所有公司角色走 LLM**（``firm_a`` / ``firm_b`` / ``firm_c`` / ``firm_d``）；``investor`` / ``regulator`` 为静态人设模板。导出 JSON 含 ``profile_source``。``--agent-profiles-all-llm`` 让六角色均 LLM；``--legacy-agent-profiles`` 全手写。

**按人数 / 公司数精确指定生成条数（``--mode-counts``）**

两个脚本现在都支持 ``--mode-counts MODE=N[,MODE=N...]``，按 token（人数 / 公司数）精确指定生成条数：

| token | 人数 N | lineup | 含义 |
|-------|--------|--------|------|
| ``bilat`` | 2 | with_institutional | 2 公司，无机构 |
| ``tri`` | 3 | with_institutional | 2 公司 + investor |
| ``quartet`` | 4 | with_institutional | 2 公司 + investor + regulator |
| ``firms3`` | 3 | firms_only | 3 公司互谈 |
| ``firms4`` | 4 | firms_only | 4 公司互谈 |

```bash
# LLM 路径：8 条 firms3 + 12 条 firms4 + 5 条 bilat + 3 条 quartet（共 28 条 LLM 场景）。
# 传入 --mode-counts 后，--n / --modes 不再决定路由，仅 --concurrency / --timeline-labels 生效。
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_llm.py \
    --mode-counts firms3=8,firms4=12,bilat=5,quartet=3 \
    --timeline-labels D6,D8 --concurrency 4 \
    --tag ltr_llm_mix \
    --agent-profile-out long_term_negotiation_llm_agent_profiles.mix.json

# 手写路径：3 条 firms3 + 4 条 firms4 + 2 条 bilat（每个 mode 在 D6/D8 上轮转，共 9 条）。
# 传入后 --modes / --replicates 被忽略；剩余参数（--timeline-labels / --requirements / --tag）仍生效。
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_scenarios.py \
    --tag ltr_mix_v1 \
    --mode-counts firms3=3,firms4=4,bilat=2 \
    --timeline-labels D6,D8
```

manifest 中会写入 ``generation_spec.mode_counts_spec``（原始字符串）与 ``generation_spec.mode_counts_resolved``（``{mode: count}`` 解析结果），便于实验记录与回放。

### `generate_long_term_negotiation_llm.py`：数量参数（`--n` / `--modes` / `--mode-counts`）

- **方式 A：`--n` + `--modes`（与 README 中 `--n 12`、`--modes bilat,quartet,firms3,firms4` 一致）**
  - **`--n`**：一共调用多少次 `agenerate_env_profile`，即生成多少条 `EnvironmentProfile`。
  - **`--modes`**：逗号分隔且**保序轮转**；第 `i` 条结果使用的模式为 `modes[i % len(modes)]`。
  - **`--timeline-labels`**：筛出若干时间轴预设（如 `D6,D8`）后，每条环境在**过滤后的 presets 列表上轮转**选取 `NegotiationTimelineParams`（与脚本内 `variant_i % len(presets)` 一致）。
  - 若只想改「总条数」：改 **`--n`** 即可。若希望四种模式**各占固定条数**，仅靠轮转无法精确指定（除非把 `--n` 凑成模式数的倍数并接受均匀分配）。

- **方式 B：`--mode-counts MODE=N[,MODE=N...]`（精确按模式定条数）**
  - 总条数 = 各 `N` 之和；展开为「逐条 mode 列表」后依次造场景。
  - 传入后 **`--n` 与 `--modes` 不再参与路由**（脚本会打印告警并忽略 `--n`）。
  - 合法 `MODE` 与 `--modes` 相同：`bilat` / `tri` / `quartet` / `firms3` / `firms4`。

### 1.1 三家及以上「公司」（``firms_only`` lineup）

谈判世界支持 **2 ~ 4 个 canonical 角色**，由 ``EnvironmentProfile.game_metadata`` 中两个字段描述：

- ``lineup ∈ {"with_institutional", "firms_only"}``：决定按哪一种顺序取角色。
  - ``with_institutional``（默认 / 历史兼容）：``firm_a → firm_b → investor → regulator`` 前缀。
  - ``firms_only``：``firm_a → firm_b → firm_c → firm_d`` 前缀，**不含**机构位；融资 / 监管路径自然成为 no-op。
- ``num_participants ∈ {2, 3, 4}``：取 lineup 顺序的前 N 个角色作为 roster。

合同主体（``c.parties``）= ``PRINCIPAL_PARTY_ROLES ∩ session.participants``：

- ``PRINCIPAL_PARTY_ROLES = {firm_a, firm_b, firm_c, firm_d}``。
- 双方 lineup 的 session 仍只有 firm_a/firm_b → 主体退化为这两家；
- 3 公司 firms_only 的 session（firm_a/firm_b/firm_c）→ 主体 = 这三家，全部 accept + sign 才能 ``success``。

CLI 模式（``--modes`` token）一览：

| token | lineup | N | roster |
|-------|--------|---|--------|
| ``bilat`` | with_institutional | 2 | firm_a, firm_b |
| ``tri`` | with_institutional | 3 | firm_a, firm_b, investor |
| ``quartet`` | with_institutional | 4 | firm_a, firm_b, investor, regulator |
| ``firms3`` | firms_only | 3 | firm_a, firm_b, firm_c |
| ``firms4`` | firms_only | 4 | firm_a, firm_b, firm_c, firm_d |

**完整流水线（用大模型造数 → 跑评测）**

下面以 ``firms3`` / ``firms4`` 为主、混入双方与四方 lineup，端到端走一遍。LLM 路径需要 ``social_env/.env`` 里的 ``OPENAI_API_KEY``（``litellm`` / OpenAI 兼容；亦可用第三方网关，详见末尾"踩过的坑"）。

> 验证状态（提交一致性）：本节命令在 5 种 lineup（bilat / tri / quartet / firms3 / firms4）上 round-trip 通过，``compute_negotiation_final_state_metrics`` 同时落盘；规则冒烟 ``tests/test_long_term_negotiation_smoke.py`` 通过。

```bash
# 1. 进 conda 环境 + 仓库根
cd social_env

# 2. 用大模型批量造场景 + 公司侧 LLM 人设；机构位（如混入 tri/quartet 时）保持静态人设。
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_llm.py \
    --model gpt-4o-mini \
    --agent-profile-model gpt-4o-mini \
    --n 12 --concurrency 4 \
    --modes bilat,quartet,firms3,firms4 \
    --timeline-labels D6,D8 \
    --requirements "validate firm_a..firm_d expansion" \
    --tag ltr_multi_firm_llm_v1 \
    --agent-profile-out long_term_negotiation_llm_agent_profiles.multi_firm.json

SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_llm.py \
  --model gpt-4o-mini \
  --agent-profile-model gpt-4o-mini \
  --mode-counts bilat=3,quartet=3,firms3=3,firms4=3 \
  --concurrency 4 \
  --timeline-labels D6,D8 \
  --requirements "validate firm_a..firm_d expansion" \
  --tag ltr_multi_firm_llm_v1 \
  --agent-profile-out long_term_negotiation_llm_agent_profiles.multi_firm.json
# 输出：~/.sotopia/data/long_term_negotiation_llm_manifest.json
#       ~/.sotopia/data/long_term_negotiation_llm_agent_profiles.multi_firm.json
#       ~/.sotopia/data/{AgentProfile,EnvironmentProfile,...}/...
```

**`generate_long_term_negotiation_llm.py` 执行流程与涉及文件**

以下均在仓库根 **`social_env`** 下、`SOTOPIA_STORAGE_BACKEND=local` 时使用本地目录 **`~/.sotopia/data/`**。

1. **`scripts/generate_long_term_negotiation_llm.py`**
   - `main()` 解析 CLI 后 `asyncio.run(main_async(...))`。
   - **`main_async()`** 根据 `parse_mode_counts` 或 **`--n` + `modes_cycle_from_arg("--modes")`** 决定要生成几条、每条对应哪种 `mode`。
   - **`_load_handwritten_generator()`** 动态加载同目录 **`scripts/generate_long_term_negotiation_scenarios.py`** 为模块 `ltr`，复用其中的落库、`EnvAgentComboStorage`、`persist_scenario_v2`、事件脚本等逻辑。
   - 若带 **`--clean`**：调用 **`ltr.wipe_local_data`** 清空本地数据（慎用）。
   - **Agent 侧**：默认走 **`long_term_negotiation/llm_agent_profile_gen.py`** 中的 **`agenerate_negotiation_agent_profiles`**（及 **`agent_profile_to_jsonable`**），生成并保存 `AgentProfile`，并写出 **`--agent-profile-out`** 指定的 JSON；若 **`--legacy-agent-profiles`** 则改为手写常量画像。
   - **`ltr.pairwise_strangers`**、**`ltr.save_negotiation_agent_profiles_v2`**：关系与 `AgentProfileV2`。
   - **`ltr.negotiation_event_scripts`** 并逐个 **`save()`**：写入 **`EventScript`** 等。
   - **场景侧**：对每个 inspiration 并发执行 **`generate_one_llm_profile`** → **`sotopia/generation_utils/generate.py`** 的 **`agenerate_env_profile`**（LiteLLM），得到候选 **`EnvironmentProfile`**。
   - 对每条成功的 env：**`scenario_loader.build_negotiation_game_metadata_bundle`** 合并 `game_metadata`（时间轴、lineup、`num_participants` 等与手写脚本同源）；按需 **`firms3_goal_padding` / `firms4_goal_padding` / `tri_goal_padding` / `quartet_goal_padding`**；**`env_llm.save()`**；**`ltr.save_combo`**；**`ltr.persist_scenario_v2`**。
   - **`ltr.save_environment_list_for_combos`** 汇总环境列表。
   - 最后将本次运行的元数据写入 **`~/.sotopia/data/<--manifest-name>`**（默认 `long_term_negotiation_llm_manifest.json`）。

2. **依赖模块（Concept）**

   | 模块 / 脚本 | 作用 |
   |-------------|------|
   | `long_term_negotiation/llm_agent_profile_gen.py` | LLM 生成谈判用 `AgentProfile` 与导出 JSON |
   | `generation_utils/generate.py` | `agenerate_env_profile` 生成自然语言 `scenario` / `agent_goals` 等 |
   | `long_term_negotiation/scenario_loader.py` | `build_negotiation_game_metadata_bundle` 对齐谈判 `game_metadata` |
   | `scripts/generate_long_term_negotiation_scenarios.py`（作为 `ltr`） | 与手写造数共用写库、V2、combo、事件锚点等 |

3. **典型落盘（`~/.sotopia/data/`）**

   | 路径（类型） | 内容 |
   |--------------|------|
   | `AgentProfile/`、`AgentProfileV2/` | 角色卡 |
   | `EnvironmentProfile/` | 每条 LLM 场景 |
   | `EnvAgentComboStorage/` 等 | 场景与角色组合 |
   | `EventScript/`、`RelationshipProfile/` 等 | `ltr` 流程中的事件与关系 |
   | `--agent-profile-out` 所指文件 | 本次导出的 agent 画像 JSON（含 `profile_source`） |
   | `--manifest-name` 所指文件 | 供 `negotiation-batch --scenario-manifest` 使用的 manifest |

#### `long_term_negotiation_llm_manifest.json` 与 `long_term_negotiation_llm_agent_profiles.*.json` 的关系

- **`long_term_negotiation_llm_manifest.json`（场景索引）**
  - 记录本次生成出的环境集合（`environment_profile pk`、`codename`、`mode`、`lineup`、`num_participants` 等）。
  - 是批评测入口文件：`negotiation-batch --scenario-manifest` **直接读取它**来展开 episode 任务。
- **`long_term_negotiation_llm_agent_profiles.<name>.json`（画像绑定明细）**
  - 记录每个环境对应的角色画像映射（`AgentProfile pk`/`AgentProfileV2 pk`、`profile_source` 等）。
  - 主要用于审计/复现“当时每个场景绑定了哪套 agent 画像”，不是 CLI 必填参数。
- **两者如何关联**
  - 通过 `codename` + 同次生成 `tag` 对齐：manifest 给出“跑哪些 env”，agent-profile 文件给出“这些 env 绑定了谁”。

#### 测试 / 评测时哪些文件会被加载，何时加载

1. **运行 `generate_long_term_negotiation_llm.py` 时（数据构造阶段）**
   - 写出 `long_term_negotiation_llm_manifest.json`（最后一步）。
   - 写出 `--agent-profile-out` 指定的 agent-profile 明细 JSON（最后一步）。
2. **运行 `negotiation-batch --scenario-manifest ...` 时（评测启动阶段）**
   - `scenario_loader.environment_pks_from_manifest(...)` 在 CLI 参数解析后立刻读取 **manifest**，得到本批 `env_pk` 列表。
3. **每个 episode 实际执行前（任务展开阶段）**
   - `load_negotiation_scenario_from_environment_profile_pk(env_pk)` 按 manifest 中的 pk 去 `EnvironmentProfile/` 加载场景元数据（`timeline`、`lineup`、`num_participants` 等）。
4. **agent-profile 明细 JSON 在评测阶段**
   - 默认不自动加载（CLI 不依赖它）；仅在你做人设审计、回放或自定义脚本做“env->agent pk”核对时读取。

**一句话**：先在本包 **`llm_agent_profile_gen`** 中造并保存 Agent，再对每条 inspiration 调 **`agenerate_env_profile`** 造 Environment，用 **`scenario_loader`** 接上谈判 **`game_metadata`**，最后经 **`generate_long_term_negotiation_scenarios.py`** 提供的 **`ltr`** 辅助函数把 combo / V2 / 事件等写入本地存储，并写 manifest。

```bash
# 3. 用刚才造的 manifest 批量跑评测，写汇总 JSON + 文本 log（含中间状态最后一帧的 final_state_score）
mkdir -p logs runs
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model gpt-5-mini \
    --evaluator-model gpt-5-mini \
    --batch-size 8 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --print-logs \
    --execution-trace-dir runs/execution_traces \
    --output runs/ \
    --tag ltr_multi_firm_llm_v1
# 输出：runs/negotiation_eval_<tag>_<时间戳>.json（含 aggregate_means 与 rows）
#       logs/negotiation_batch_YYYYMMDD_HHMMSS.log
#       runs/execution_traces/<模型>/<时间戳>/<tag>_<runid>_<seq>_<角色或terminal_evaluator>.jsonl

# 4. （可选）省钱版：跳过终局 LLM 评测，只跑环境 + agents
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model gpt-4o-mini \
    --batch-size 3 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --skip-llm-scoring \
    --output runs/ \
    --tag ltr_multi_firm_llm_v1
```

不需要 LLM 造场景时，第 2 步可改用规则手写脚本（**离线、无需 API key**）：

```bash
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_scenarios.py \
    --clean \
    --tag ltr_multi_firm_v1 \
    --modes bilat,tri,quartet,firms3,firms4 \
    --timeline-labels D6,D8 \
    --replicates 2 \
    --requirements "validate firm_c/firm_d expansion across with_institutional + firms_only lineups"
# 输出：~/.sotopia/data/long_term_negotiation_manifest.json + 6 条 AgentProfile（含 firm_c/firm_d）
#       共 20 条 EnvironmentProfile（5 modes × 2 timelines × replicates 2）
```

之后第 3 步把 ``--scenario-manifest`` 换成 ``~/.sotopia/data/long_term_negotiation_manifest.json`` 即可。

**踩过的坑（数据合成相关）**

1. ``OpenAIException - Invalid schema for response_format 'LLMNegotiationAgentDraft': 'additionalProperties' is required to be supplied and to be false``
   - 触发：第三方 OpenAI 兼容网关对 ``json_schema`` 做了 strict 校验；Pydantic v2 默认不在带 ``Field(default=...)`` 的 root object 上加 ``additionalProperties: false``，``format_bad_output`` 修复路径会把整份 schema 以 strict 下传，于是 400。
   - 解决：``LLMNegotiationAgentDraft`` 加 ``OPENAI_DISABLE_STRICT_JSON_SCHEMA: ClassVar[bool] = True``，``_build_json_schema_response_format`` 检测到该标记会下传 ``strict=False``。新 Pydantic 模型如果遇到同样错误，照搬该 ClassVar 即可。
2. ``firms3`` / ``firms4`` 的 ``EnvironmentProfile.game_metadata`` 必含 ``lineup="firms_only"``；老的 manifest（仅有 ``quartet`` / ``num_participants``）缺省 ``lineup="with_institutional"``，加载到 firm_c/firm_d 角色会报 unknown role。重新跑一次 ``--clean`` 数据生成即可。

**单条 episode 程序化（任意 N + 任意 lineup）**

```python
from sotopia.settings.long_term_negotiation.llm_evaluation import (
    run_llm_negotiation_episode_evaluation,
)
import asyncio

# 3 公司 firms_only：roster = (firm_a, firm_b, firm_c)
model_dict = {
    "env": "gpt-4o-mini",
    "agent1": "gpt-4o-mini",
    "agent2": "gpt-4o-mini",
    "agent3": "gpt-4o-mini",
}
result = asyncio.run(
    run_llm_negotiation_episode_evaluation(
        model_dict,
        num_participants=3,
        lineup="firms_only",
        run_terminal_llm_eval=False,
    )
)
print(result.terminal, result.rule_metrics["negotiation_final_state_score"])
```

### 2. 仅代码批量（不写 CLI）

```python
from sotopia.settings.long_term_negotiation.batch_evaluation import (
    run_long_term_negotiation_eval_batch,
)

rows = run_long_term_negotiation_eval_batch(
    agent_models=["gpt-4o-mini"],
    evaluator_model="gpt-4o-mini",
    repeats_per_model=1,
    batch_size=3,
    run_terminal_llm_eval=True,
)
```

### 3. 单 episode 程序化

```python
from sotopia.settings.long_term_negotiation.llm_evaluation import (
    run_llm_negotiation_episode_evaluation,
)
import asyncio

model_dict = {
    "env": "gpt-4o-mini",      # 终局评测模型
    "agent1": "gpt-4o-mini",
    "agent2": "gpt-4o-mini",
}

result = asyncio.run(
    run_llm_negotiation_episode_evaluation(
        model_dict,
        quartet=False,
        run_terminal_llm_eval=True,
    )
)
# result.terminal, result.rule_metrics, result.llm_aggregate
```

同步封装：`evaluate_long_term_negotiation_llm_sync`（同模块）。

### 4. `run_llm_negotiation_episode_evaluation` 内部顺序（概念）

1. `default_negotiation_roster` → 按 ``num_participants``（2/3/4）+ ``lineup``（``with_institutional`` /
   ``firms_only``）取对应顺序前缀，确定 N 名参与者：
   - ``with_institutional`` 取 ``SESSION_SPEAKER_ROLE_ORDER``（firm_a, firm_b, investor, regulator）；
   - ``firms_only`` 取 ``SESSION_FIRMS_ONLY_ROLE_ORDER``（firm_a, firm_b, firm_c, firm_d）。
   场景 manifest 中已有的 ``lineup`` / ``num_participants`` 优先；CLI 再覆盖。
2. `build_llm_negotiation_agents` → `build_negotiation_social_llm_agents`。  
3. 构造 `LongTermNegotiationEnv`，`await env.run_episode_async(...)`。  
4. `compute_negotiation_rule_metrics`（**自动合并 final-state 指标**：见下文）。  
5. 若开启终局 LLM：`format_negotiation_episode_for_llm_eval` → `EpisodeLLMEvaluator` → `unweighted_aggregate_evaluate`。

### 4.1 中间状态最后一帧的评价指标

`controller.state_snapshots` 在每个 end-of-day 写一帧浅快照；`run_episode_async` 结束时再写一帧 `label="after_terminal"` 的“最终中间状态”，保证就算中途成交（mid-day terminal）也至少有一帧。`compute_negotiation_final_state_metrics(env)` 取 **最后一帧**与 `default_agent_resources_bundle()` 做 delta，输出（已并入 `rule_metrics`）：

* `negotiation_final_state_n_snapshots` / `_day_closed`
* `negotiation_final_state_total_cash` / `_total_cash_delta` / `_min_cash`
* `negotiation_final_state_n_solvent` / `_solvency_ratio`
* **`negotiation_final_state_score ∈ [0, 1]`** —— 模型评价指标之一，权重：terminal=success ×0.3，primary contract factor ×0.2（`signed`=1.0，`accepted`=0.75，`amended`=0.5，`proposed`=0.25，`rejected/failed`=0），solvency_ratio ×0.15，total_cash_delta ≥ 0 ×0.1，predefined_rule_score ×0.25。
* 4 个分项 `negotiation_final_state_score_component_*` 便于排错。

CLI（`negotiation-batch`）的 `episode_done` 行也单独打印 `final_state_score=…`，无需翻汇总 JSON。

### 4.2 不同场景的规则计算方法（写入版）

目前默认三类数据场景：`business_coopetition`、`wet_market_competition`、`resource_scheduling_management`。  
**规则口径是“统一主公式 + 场景参数化”**，不是三套互斥打分器。

1) **统一主公式（所有场景共用）**

- 先算 `primary_factor`（按主合同状态映射）；
- 再算 `predefined_rule_score`（来自 `predefined_outcome_rule` 的利润率归一化分）；
- 最终：
  - `negotiation_final_state_score = 0.3*terminal + 0.2*primary + 0.15*solvency + 0.1*liquidity + 0.25*predefined_rule`

2) **场景差异注入点（按 scene_type 变化）**

- `scenario_loader._infer_environment_context` 根据场景语义生成：
  - `scene_type`
  - `physical_social_parameters`（如 `labor_supply_tightness`、`bid_spread_index` 等）
  - `agent_perception_cues`
- `scenario_loader._collect_scenario_bound_news_threads` 按 `scene_type` 绑定不同新闻线索集合（例如湿货市场、外包、竞标等不同线索族）；
- `predefined_outcome_rule.news_signal` 与 `margin_formula` 共同影响：
  - `realized_margin`
  - `predefined_rule_total_profit`
  - `predefined_rule_score`

3) **三类场景的建议解释口径（用于报表）**

- `business_coopetition`：重点看 `primary_contract` + `predefined_rule_score`（合作与竞争并存时是否达成可结算合同）；
- `wet_market_competition`：重点看 `solvency_ratio` + `liquidity_preserved`（高频交易/现金流压力下的存活性）；
- `resource_scheduling_management`：重点看 `primary_contract` + `message/action_log` 完整性（排班与容量协调是否形成可执行安排）。

4) **资源协商调度场景的专用计算口径（resource_scheduling_management）**

- 调度目标：在每轮对话后，对有限资源 `R={r_k}` 生成 `allocation(i,k)`；
- 三阶段规则：
  1. **Demand Scoring**（需求强度）  
     - `D_{i,k} = base_demand_{i,k} * (1 + α*urgency_i + β*aggressiveness_i)`  
     - `effective_demand_{i,k} = D_{i,k} * (1 + trust_bonus_i)`
  2. **Priority Computation**（优先级）  
     - `P_{i,k} = D_{i,k} * (1 + cooperation_i) * (1 + reputation_i)`  
     - `P'_{i,k} = P_{i,k} + Σ_j trust(j,i)*influence_j - overlap_penalty_{i,k}`
  3. **Allocation**（分配）  
     - 推荐 softmax：`allocation_{i,k} = r_k * exp(P'_{i,k}) / Σ_j exp(P'_{j,k})`  
     - 或受限优化：`maximize Σ_i P'_{i,k}*allocation_{i,k}` 且 `Σ_i allocation_{i,k} ≤ r_k`
- 冲突与兜底：
  - 当 `Σ_i demand_{i,k} > r_k` 启动竞争与让步；
  - 生存资源（现金/基本供给）优先于一般生产资源；
  - 允许对话驱动的再协商（价格调整、未来承诺、互信交换）。
- 对话驱动更新链：
  - `dialogue -> belief_update -> urgency_shift -> demand/priority_shift -> allocation`
  - 分配后更新 `cash/inventory/trust/reputation` 并进入下一轮。

> 实务建议：横向比较不同场景时，先比较 `negotiation_final_state_score`，再按场景主目标看分项（`*_component_*`）与 `negotiation_predefined_rule_*` 明细，避免只看单一总分。

## 规则 agent / Dummy 对照

不参与 LLM API 的快速路径：使用 `long_term_negotiation.build_rule_dummy_agents` 等（见 `settings` 顶层 `__init__.py` 导出），环境与控制器仍可走 `LongTermNegotiationEnv`，但不会经过 `llm_evaluation` 的评测管线。

可以，下面给你一份**完整可对照代码/日志**的评测分数清单。

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

如果你愿意，我可以下一步给你一份“可直接跑在 `runs/negotiation_eval_<tag>_<时间戳>.json` 上”的统计口径模板（均值/方差/成功率/分位数/按场景分组）。
