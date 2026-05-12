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
| `batch_evaluation.py` | **批量**异步调度、并发限流、JSONL 友好记录：`run_long_term_negotiation_eval_batch`；支持 ``scenario_environment_pks``、``negotiation_run_config``。 |
| `negotiation_run_config.py` | JSON 运行配置：``load_negotiation_run_config`` / ``build_negotiation_agents_from_run_config``（选用谈判 LLM Agent 变体与 ``memory.backend``：plain / summarizing）。 |
| `eval_logging.py` | CLI / 程序化侧统一配置：`configure_negotiation_cli_logging`、`episode_*_line`、`sotopia.negotiation.batch` logger。 |
| `scenario_loader.py` | ``build_negotiation_game_metadata_bundle``；从库里还原 ``NegotiationTimelineParams``。 |
| `external_events.py` | 外部事件规格与 runner。 |
| `agent_state_variables.py` | 心理/状态变量与 prompt 附加段。 |
| 其余 | 调度解析、会话 roster 分类等支撑模块。 |

各模块内仍有更细的 docstring；评测主链路的「文件 → 函数 → 顺序」以 `llm_evaluation.py`、`batch_evaluation.py` 及 CLI `sotopia/cli/benchmark/negotiation_batch.py` 的模块说明为准。

## 运行 LLM 评测 / 生成 JSONL：调用顺序

### 1. 命令行（推荐）

- 安装可执行入口：`pip install -e .`（在 `social_env` 根目录），主命令名为 **`sotopia`**（见 `pyproject.toml` 的 `[project.scripts]`）。
- 子命令 **`negotiation-batch`** 在 Typer 中注册；若本机显示为根级选项，以 `sotopia --help` 为准。
- 实现文件：`sotopia/cli/benchmark/negotiation_batch.py`，入口函数 **`negotiation_batch`**。

顺序概括：

1. `negotiation_batch` 解析参数；无场景 pk 时构造默认 `NegotiationTimelineParams`，有场景 pk 时单场参数由存储层提供。参与者数量默认：无场景时由 `--quartet`（2 或 4）或显式 `--num-participants`；有场景时读 `game_metadata.num_participants`（缺省由 `quartet` 推断），CLI 的 `--num-participants` 可覆盖每场。
2. 调用 `sotopia.settings.long_term_negotiation.batch_evaluation.run_long_term_negotiation_eval_batch`。
3. 内部 `run_long_term_negotiation_eval_batch_async` 对每个 `(agent_model, repeat)` 限流并发，单任务调用 `run_llm_negotiation_episode_evaluation`。
4. 汇总为 dict 列表，可选 `-o/--output` 追加写入 **JSONL**。

**``--run-config``（选用 Agent / 记忆后端）**

- 传入指向 JSON 的路径，例如 ``--run-config sotopia/settings/long_term_negotiation/run_config_examples/summarizing_memory.json``。
- 字段语义见 ``negotiation_run_config.py`` 模块 docstring；示例 JSON 在同目录 ``run_config_examples/`` 下。
- 不传时与历史行为一致（plain 滑动窗口记忆）。传入时，每条 JSONL 记录会多一个 ``negotiation_run_config`` 字段便于复现实验。

**日志与可读输出（与 JSONL 区分）**

- **JSONL（-o）**：机器可读的一条 episode 一整行 JSON；适合后续聚合脚本，不叫「运行时 log」。
- **stderr 进度条**：批量跑时用 `tqdm` 显示完成进度与速率，不写进 `-o`，也不写入 `--log-file`。
- **文本 log（可读）**：`--print-logs` 用 Rich 在控制台打出 INFO；`--log-file PATH` 向该路径 **追加** UTF-8 纯文本行（格式 `时间 | LEVEL | logger名 | message`），每条 episode 有 `episode_start` / `episode_done`（及失败时的 traceback）。配置入口：`long_term_negotiation.eval_logging.configure_negotiation_cli_logging`。
  - 若不传 `--log-file`，CLI 会自动落盘到 `logs/negotiation_batch_YYYYMMDD_HHMMSS.log`（文件名显式带运行时间）。
  - 启动时会打印 `[negotiation-batch] run_started_at=... log_file=...` 便于追踪本次运行日志文件。
  - 第三方库（如 LiteLLM）在 INFO 仍可能刷屏，可按需调高其日志级别。
- **程序化**：若直接从 Python 调用 `run_long_term_negotiation_eval_batch*`，同样需要自行对上述函数调用一次以保持与 CLI 行为一致；传入 `negotiation_run_config=load_negotiation_run_config(Path("..."))` 与 CLI `--run-config` 等价；中间模型输出可传 `model_trace_dir=` / `execution_trace_dir=`；单次评测可传 `run_llm_negotiation_episode_evaluation(..., model_trace_dir=..., execution_trace_dir=..., model_trace_tag=..., execution_trace_tag=..., negotiation_run_config=...)`。

无安装时用模块方式（需将 `social_env` 加入 `PYTHONPATH`）可参考该文件末尾的 `python -m sotopia.cli.benchmark.negotiation_batch` 说明。

**从存储加载题库场景（与 `benchmark_v2_data_models` 生成的 `EnvironmentProfile` 对齐）**：先用
`scripts/generate_long_term_negotiation_scenarios.py` 写入 `~/.sotopia/data/`，再在执行时传
`--scenario-manifest ~/.sotopia/data/long_term_negotiation_manifest.json` 和/或多次
`--scenario-env-pk <pk>`。此时每条 episode 会从 profile 的 ``game_metadata.timeline`` 构造
`NegotiationTimelineParams`，参与者人数由各场景的 ``game_metadata.num_participants``（或 ``quartet`` 推断）决定；CLI 的 ``--quartet`` 对场景路径仅作提示，可用 ``--num-participants`` 覆盖。

若需 **用大模型写好 scenario / agent_goals 文本**：运行 `scripts/generate_long_term_negotiation_llm.py`
（manifest 默认为 ``~/.sotopia/data/long_term_negotiation_llm_manifest.json``），命令行同上把 manifest 换成该路径即可。

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
# 3. 用刚才造的 manifest 批量跑评测，写 JSONL + 文本 log（含中间状态最后一帧的 final_state_score）
mkdir -p logs runs
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model gpt-5-mini \
    --evaluator-model gpt-5-mini \
    --batch-size 8 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --print-logs \
    --execution-trace-dir runs/execution_traces \
    --output runs/ltr_multi_firm_eval.jsonl \
    --tag ltr_multi_firm_llm_v1
# 输出：runs/ltr_multi_firm_eval.jsonl    每行一条 episode（含 rule_metrics / llm_aggregate）
#       logs/negotiation_batch_YYYYMMDD_HHMMSS.log  每条 episode 的 episode_done 行（带 final_state_score）
#       runs/execution_traces/<tag>_<runid>_<seq>.execution.json
#       runs/execution_traces/<tag>_<runid>_<seq>.execution.transcript.txt  （同 stem，纯文本全量交互）

# 4. （可选）省钱版：跳过终局 LLM 评测，只跑环境 + agents
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model gpt-4o-mini \
    --batch-size 3 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --skip-llm-scoring \
    --output runs/ltr_multi_firm_eval.norubric.jsonl \
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
* **`negotiation_final_state_score ∈ [0, 1]`** —— 模型评价指标之一，权重：terminal=success ×0.4，primary contract phase ×0.3（`signed`=1.0，`accepted`=0.75，`amended`=0.5，`proposed`=0.25），solvency_ratio ×0.2，total_cash_delta ≥ 0 ×0.1。
* 4 个分项 `negotiation_final_state_score_component_*` 便于排错。

CLI（`negotiation-batch`）的 `episode_done` 行也单独打印 `final_state_score=…`，无需翻 JSONL。

## 规则 agent / Dummy 对照

不参与 LLM API 的快速路径：使用 `long_term_negotiation.build_rule_dummy_agents` 等（见 `settings` 顶层 `__init__.py` 导出），环境与控制器仍可走 `LongTermNegotiationEnv`，但不会经过 `llm_evaluation` 的评测管线。
