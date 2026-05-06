# Sotopia 项目说明（中文）

> 本文件位置：`sotopia/PROJECT_OVERVIEW_zh.md`

## 0. 项目简介

**Sotopia**（ICLR 2024 Spotlight）是一个 **开放式社交学习 / 评测环境**，用来在 *社会场景* 中模拟和评估语言智能体（LLM agents）的社交智能。

核心使用方式：
- 给定一个「社交任务剧本」（`EnvironmentProfile` + `AgentProfile`），把两个/多个 LLM Agent 放到同一个 `ParallelSotopiaEnv` 里轮流"说话/做事"，
- 由一个 LLM 评估器（`EpisodeLLMEvaluator`）按 7 个维度打分（goal、believability、relationship、knowledge、secret、social_rules、financial_and_material_benefits），
- 全部交互写入 `EpisodeLog`，可以做 benchmark 排行榜。

仓库里另有 `sotopia-rl/` 子仓库，是 Sotopia-RL（基于 Sotopia 的 RL 训练管线，含 SFT / RM / GRPO），独立于核心 Sotopia 框架。

---

## 1. 每个文件夹的作用

### 1.1 顶层结构

| 路径 | 作用 | 与核心模拟的关系 |
| --- | --- | --- |
| `sotopia/` | **Python 主包**：环境、Agent、评测器、生成、采样、数据库、CLI、API 等核心代码。 | 核心 |
| `sotopia_conf/` | `gin-config` 配置文件 + 解析工具，给 `examples/experiment_eval.py` 之类的批量实验脚本注入参数。 | 核心（配置） |
| `examples/` | 顶层使用示例：批量评测、benchmark、生成场景、修复 episode、自定义维度等。 | 核心入口 |
| `examples/experimental/` | 多种"实验性"高级玩法（spyfall、werewolves、undercover、negotiation_arena、interview_openhands、group_discussion 等基于 `aact` 节点的多 Agent 设置），需要 Redis。 | 实验性扩展 |
| `scripts/` | Shell 脚本封装：评测、迁移 redis-to-local、运行交互/脚本/全流程实验。 | 实验入口 |
| `tests/` | **pytest 单元/集成测试套件**。 | 单元测试 |
| `sotopia-chat/` | 独立的 chat 服务（`chat_server.py`、`fastapi_server.py`），用于在线人机/多人交互演示。 | 周边服务（与单元测试无关） |
| `ui/` | Streamlit 前端，浏览数据库 / episode / 标注。 | 周边服务 |
| `docs/` | 项目官网（Next.js + Nextra），不是 Python 文档。 | 文档站 |
| `notebooks/` | Jupyter 教程 + 数据序列化/统计 notebook。 | 演示/教程 |
| `stubs/` | 第三方库的类型 stub（`absl`, `redis_om`, `pettingzoo` 等）给 mypy 用。 | 静态类型，无运行时作用 |
| `figs/` | 项目封面图。 | 文档资源 |
| `sotopia-rl/` | **独立子项目**：Sotopia-RL（社交智能 RL 训练）。 | 独立训练管线 |
| `.devcontainer/` | VSCode/Codespace devcontainer（docker compose）。 | 开发环境 |
| `.github/` | CI/CD（`tests.yml`）、issue 模板。 | 仓库元数据 |
| `pyproject.toml`, `uv.lock` | 包定义（uv/pip）。 | 构建/依赖 |
| `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`, `.pre-commit-config.yaml`, `.gitignore` | 标准开源项目元数据。 | — |

### 1.2 `sotopia/`（核心 Python 包）

| 路径 | 作用 |
| --- | --- |
| `sotopia/agents/` | Agent 实现：`base_agent.py`、`llm_agent.py`（LLMAgent / HumanAgent / ScriptWritingAgent / Agents 容器）、`redis_agent.py`、`generate_agent_background.py`。 |
| `sotopia/envs/` | 模拟环境：`parallel.py`（`ParallelSotopiaEnv`，PettingZoo 风格的并行多智能体环境，是核心）、`evaluators.py`（`RuleBasedTerminatedEvaluator`、`EpisodeLLMEvaluator`、`EvaluationForAgents`、`unweighted_aggregate_evaluate`）、`social_game.py`（`SocialGame`、`SocialDeductionGame` 抽象）。 |
| `sotopia/server.py` | **整个交互流程的入口**：`run_async_server` / `run_sync_server` / `arun_one_episode`，负责 reset、轮转 step、调用评估器、写入 `EpisodeLog`。 |
| `sotopia/messages/` | 消息/动作/观察的数据类：`AgentAction`, `Observation`, `ScriptBackground`, `ScriptEnvironmentResponse` 等。 |
| `sotopia/samplers/` | 采样器：`UniformSampler`, `ConstraintBasedSampler`, `BaseSampler`，从数据库里组合 (env, agents) 三元组。 |
| `sotopia/database/` | 持久化层：基于 `redis-om` 的 JsonModel（`AgentProfile`、`EnvironmentProfile`、`EpisodeLog`、`EnvAgentComboStorage`、`AnnotationForEpisode`、`Annotator`、`RelationshipProfile`、`CustomEvaluationDimension*`、`SotopiaDimensions`），还有 `storage_backend.py`（**支持 Redis 和本地 JSON 两种后端**，由 `SOTOPIA_STORAGE_BACKEND` 决定）、`serialization.py`（CSV / JSONL ↔ DB）、`evaluation_dimensions.py`（评估维度构建器）。 |
| `sotopia/generation_utils/` | LLM 生成工具：`generate.py`（`agenerate`、`agenerate_env_profile`、`agenerate_script` 等基于 `litellm.acompletion` 的异步生成）、`output_parsers.py`。 |
| `sotopia/renderers/` | 把消息渲染成给 LLM 的 prompt（`xml_renderer.py` 等）。 |
| `sotopia/cli/` | `sotopia` CLI 入口（用 `typer`）：`install/`（交互式数据库/数据集安装向导）、`benchmark/benchmark.py`（**官方 benchmark 实现**）。 |
| `sotopia/api/` | FastAPI 服务（`fastapi_server.py`、`websocket_utils.py`），把模拟暴露为 HTTP/WS。 |
| `sotopia/experimental/` | 基于 [`aact`](https://github.com/ProKil/aact) 节点架构的实验版 Agent/Env/Moderator（`agents/moderator.py`、`agents/redis_agent.py`、`envs/templates/`、`envs/utility_nodes/`）。供 `examples/experimental/*` 使用。 |
| `sotopia/logging.py`, `sotopia/utils.py`, `sotopia/__init__.py`, `sotopia/py.typed` | 工具与包元数据。 |

### 1.3 `examples/`

| 文件/目录 | 作用 |
| --- | --- |
| `minimalist_demo.py` | **最简单的一次性 demo**：跑一个 episode。 |
| `experiment_eval.py` | **批量自我对弈评测主入口**，配 `sotopia_conf/*.gin` 使用，由 `scripts/run_interaction.sh`、`scripts/evaluate_finetuned_*.sh` 调起。 |
| `benchmark_evaluator.py` | 评估"评估器本身"——把机器评分与人类标注做相关性分析（pearson/spearman/MSE）。 |
| `evaluate_existing_episode.py` | 对已存在的 episode 重新跑评估器。 |
| `experiment_eval.py` + `fix_missing_episodes*.py` | 批量补跑/修复缺失的 episode（按 tag 检索）。 |
| `generate_scenarios.py`, `generate_specific_envs.py` | **数据合成**：基于已有数据集（mutual_friend、craigslist_bargains）合成新的 `EnvironmentProfile`。 |
| `generate_script.py` | 生成 Script 形式的对话（无交互，纯一次生成）。 |
| `use_custom_dimensions.py` | 演示自定义评估维度。 |
| `fast_api_example.py` | 调用 FastAPI 服务的客户端示例。 |
| `generation_api/custom_model.py` | 自定义本地模型生成接入示例。 |
| `experimental/` 子目录 | 每个子目录是一个实验性"游戏/场景"：`spyfall`, `undercover`, `werewolves`, `negotiation_arena`, `interview_openhands`, `group_discussion_agents`, `multi_agents_private_dm`, `tick_and_echo_agents`, `realtime`, `websocket`, `nodes`, `sotopia_original_replica`。每个目录基本是一个 `aact` 节点 toml + Python 入口。 |

### 1.4 `scripts/`

| 脚本 | 作用 |
| --- | --- |
| `run_all.sh` | 串联跑「正常」+「omniscient」+「script」三种模式实验。 |
| `run_interaction.sh` | 调 `examples/experiment_eval.py` 跑一次双 Agent 自我对弈。 |
| `run_script_full.sh` | 跑 script 模式（`run_async_server_in_batch_script.gin`）。 |
| `evaluate_finetuned_full.sh`, `evaluate_finetuned_MF.sh` | 微调模型的批量评测。 |
| `display_benchmark_results.sh` | 展示 benchmark 结果。 |
| `fix_missing_episodes_with_tag.sh` | 按 tag 补跑缺失 episode。 |
| `modal/` | 在 [Modal](https://modal.com/) 上跑实验。 |
| `README.md` | **数据库迁移说明**（Redis dump → 本地 JSON）；本目录下的 `migrate_redis_to_local.sh` / `start_redis_with_dump.sh` / `export_redis_to_local.py` / `stop_redis.sh` 等迁移脚本只在 README 中描述，未必都真实存在于此目录。 |

### 1.5 `sotopia_conf/`（实验配置）

`gin-config` 文件，给 `examples/experiment_eval.py` 等脚本提供默认参数：

- `run_async_server_in_batch.gin` — 普通批量交互。
- `run_async_server_in_batch_script.gin` — Script 模式。
- `rerun_missing_episodes_in_batch.gin`, `rerun_missing_episodes_with_tag.gin` — 补跑。
- `server.py`, `gin_utils.py` — 解析逻辑。
- `generation_utils_conf/`, `server_conf/` — 子配置。

### 1.6 `tests/`（pytest 测试套件）

| 子目录/文件 | 测的是什么 |
| --- | --- |
| `tests/conftest.py` | **核心**：①把 `SOTOPIA_STORAGE_BACKEND` 默认设为 `local`；②`autouse` 的 `mock_llm_calls` fixture 拦截 `litellm.acompletion`，按 schema 自动生成假 LLM 响应（除非 test 加 `@pytest.mark.real_llm`）；③`mock_llm_response` fixture 让单测指定具体回答。 |
| `tests/tests.sh` | 在 devcontainer 里 `uv run pytest tests/experimental` 的快捷脚本。 |
| `tests/api/test_fastapi.py` | 测 `sotopia/api/fastapi_server.py`。 |
| `tests/cli/test_install.py` | 测 `sotopia install` CLI。 |
| `tests/database/test_database.py`, `test_evaluation_dimensions.py`, `test_local_storage.py`, `test_serialization.py`, `test_storage_backend.py` | 数据库 / 本地存储 / 序列化 / 自定义维度。 |
| `tests/envs/test_parallel.py`, `test_evaluators.py`, `test_get_bio.py`, `test_background.json` | `ParallelSotopiaEnv` 和评估器；`test_background.json` 是测试夹具数据。 |
| `tests/experimental/test_agent.py` | 测 `sotopia.experimental` 的 aact agent。 |
| `tests/generation_utils/test_generation.py` | 测 `agenerate*` 系列。 |
| `tests/integration/test_benchmark.py` | **集成**：跑 `sotopia.cli.benchmark` 的端到端流程。 |
| `tests/renderers/test_xml_renderer.py` | XML prompt 渲染。 |
| `tests/sampler/test_sampler.py` | `UniformSampler` / `ConstraintBasedSampler`。 |

### 1.7 `sotopia-rl/`（独立 RL 子项目）

| 路径 | 作用 |
| --- | --- |
| `sotopia-rl/sotopia_rl/` | RL 训练核心：`sft_trainer.py`、`rm_trainer.py`、`ppo_trainer.py`、`grpo_trainer.py`、`data.py`。 |
| `sotopia-rl/scripts/` | 训练/推理脚本：`train_sft.{py,sh}`、`train_rm.{py,sh}`、`train_grpo.{py,sh}`、`inference_*.py`、`accelerate_config_*.yaml`、`annotate/`、`data_process/`、`evaluate/`。 |
| `sotopia-rl/data/` | RL 训练用的小样本 JSON。 |
| `sotopia-rl/evals/` | RL 后的自动评测：`experiment_eval.py`、`self_play.sh`、`*_serving.sh`、`sotopia_conf/`、`logs/`、Jinja chat 模板等。 |
| `sotopia-rl/serves/` | Django web 服务（用于人工标注/服务）。 |
| `sotopia-rl/annotator/` | 通过 Google Forms 做人工标注的工具。 |
| `sotopia-rl/prompter/` | Prompt 工具。 |
| `sotopia-rl/tests/`, `sotopia-rl/stubs/`, `sotopia-rl/assets/` | 测试 / 类型 stub / 图片资源。 |

---

## 2. 想"测试"应该从什么命令开始

这里"测试"分两种含义，按需选用：

### 2.1 跑 pytest 单元/集成测试（验证代码本身）

最简单：

```bash
cd /mnt/userdata/yphao/FC/game_MAS/social_env

uv sync --all-extras
uv run --extra test --extra api pytest tests
```

特点：
- `tests/conftest.py` 自动 mock 掉 `litellm.acompletion`，**默认不调真实 LLM**；想跑真实 LLM 调用要 `pytest -m real_llm` 并配 `OPENAI_API_KEY`。
- 默认 `SOTOPIA_STORAGE_BACKEND=local`，**不需要 Redis**。
- `pyproject.toml [tool.pytest.ini_options]` 已设 `testpaths = ["tests"]`、`python_files = "test_*.py"`。
- 想跑某个子集，例如：
  - `uv run pytest tests/database` — 数据库测试
  - `uv run pytest tests/envs/test_parallel.py` — 并行环境
  - `uv run pytest tests/integration` — 端到端 benchmark
  - `bash tests/tests.sh` — 在 devcontainer 里跑 `tests/experimental`

### 2.2 跑社交对弈实验（让 LLM Agent 在 Sotopia 里互相"被测"）

#### 2.2.1 一键 demo（最快）

```bash
cd /mnt/userdata/yphao/FC/game_MAS/sotopia
echo 'OPENAI_API_KEY=sk-xxx' > .env
echo 'SOTOPIA_STORAGE_BACKEND=local' >> .env

uv sync --all-extras
uv run sotopia install                     # 交互式拉取数据集到 ~/.sotopia/data
uv run --env-file .env python examples/minimalist_demo.py
```

`minimalist_demo.py` 内部就是 `run_async_server(model_dict={...gpt-4o-mini...}, sampler=UniformSampler())`，会随机抽一个场景、跑一个 episode。

#### 2.2.2 官方 benchmark（推荐"正经测试"用）

```bash
# 跑某个 model 在 hard tasks 上 vs 默认 partner（llama-3-70b-chat）的成绩
uv run sotopia benchmark \
    --models gpt-4o-mini \
    --partner-model together_ai/meta-llama/Llama-3-70b-chat-hf \
    --evaluator-model gpt-4o \
    --task hard \
    --batch-size 10 \
    --push-to-db          # 写入数据库
```

实现见 `sotopia/cli/benchmark/benchmark.py`：
- 读 `EnvironmentList["01HAK34YPB1H1RWXQDASDKHSNS"]`（hard 子集），
- 调 `_list_all_env_agent_combo_not_in_db` 组合 (env, agents)，
- 调 `run_async_benchmark_in_batch` → `run_async_server`，
- 用 `SotopiaDimensions` 做 `EpisodeLLMEvaluator`，
- `display_in_table` / `save_to_jsonl` 输出。

#### 2.2.3 批量自我对弈（论文复现入口）

```bash
cd scripts
./run_interaction.sh gpt-4o-mini False True my_tag
# 等价于 examples/experiment_eval.py + sotopia_conf/run_async_server_in_batch.gin
```

### 2.3 数据合成 / 评测内容（"被测样本"在哪里）

- **被测的"题目"**（场景 + Agent + 目标）= `EnvironmentProfile` × `AgentProfile` 的组合，存放在数据库里：
  - Redis 模式：通过 `uv run sotopia install` 从 HuggingFace（`cmu-lti/sotopia` 等，列表见 `sotopia/cli/install/published_datasets.json`）拉到 Redis；
  - Local 模式：拉到 `~/.sotopia/data/{AgentProfile,EnvironmentProfile,EnvAgentComboStorage,EnvironmentList,Annotator,EpisodeLog,...}/*.json`；
  - 也可以从 Redis dump 迁移过来：`scripts/README.md` 说明的 `migrate_redis_to_local.sh` / `export_redis_to_local.py`。
- **数据合成（生成新场景）**：
  - `examples/generate_scenarios.py` —— 主入口，按 `mutual_friend` / `craigslist_bargains` 生成新 `EnvironmentProfile`；
  - `examples/generate_specific_envs.py` —— 具体的合成函数；
  - `examples/use_custom_dimensions.py` —— 自定义评估维度示例；
  - 底层调用 `sotopia/generation_utils/generate.py` 中的 `agenerate_env_profile` / `agenerate_script` 等。
- **被测维度**（"测什么"）：
  - 默认 7 维 `SotopiaDimensions`，定义在 `sotopia/database/evaluation_dimensions.py`；
  - 范围：goal[0,10]、believability[0,10]、knowledge[0,10]、relationship[-5,5]、financial_and_material_benefits[-5,5]、social_rules[-10,0]、secret[-10,0]（详见 `sotopia/cli/benchmark/benchmark.py:dimension_range_mapping`）。
- **测试结果**：写入 `EpisodeLog`（每条对话历史 + 每个 agent 的多维 reward）。

---

## 3. 测试流程的设计在哪里 / 哪些文件与测试无关

### 3.1 "对弈/评测"的核心流程文件（按调用顺序）

1. **入口（实验 runner）**
   - `examples/experiment_eval.py`（批量）
   - `examples/minimalist_demo.py`（单条 demo）
   - `sotopia/cli/benchmark/benchmark.py`（标准 benchmark）
2. **场景采样**
   - `sotopia/samplers/{base_sampler,uniform_sampler,constraint_based_sampler}.py` —— 从 DB 里抽 env+agent。
3. **环境构建**
   - `sotopia/envs/parallel.py` (`ParallelSotopiaEnv`) —— 维护 background、turn、observation、消息池。
   - `sotopia/envs/social_game.py` —— 通用社交游戏抽象。
4. **Agent 行为**
   - `sotopia/agents/llm_agent.py`（`LLMAgent.aact()`）
   - `sotopia/messages/message_classes.py`（`AgentAction`、`Observation`）
   - `sotopia/renderers/xml_renderer.py`（把 history 渲染成 prompt）
   - `sotopia/generation_utils/generate.py`（调 `litellm.acompletion`）
5. **回合驱动 + 评估**
   - `sotopia/server.py` (`run_async_server` / `arun_one_episode`) —— 主循环。
   - `sotopia/envs/evaluators.py`：
     - `RuleBasedTerminatedEvaluator` —— 终止判定（max_turn_number、max_stale_turn）；
     - `EpisodeLLMEvaluator` + `EvaluationForAgents[SotopiaDimensions]` —— 终末多维打分；
     - `unweighted_aggregate_evaluate` —— 聚合。
6. **存盘 / 取数**
   - `sotopia/database/logs.py` (`EpisodeLog`)
   - `sotopia/database/storage_backend.py`（Redis / Local JSON 切换）
   - `sotopia/database/serialization.py`（导出/导入）
7. **配置**
   - `sotopia_conf/run_async_server_in_batch{,_script}.gin`
   - `sotopia_conf/{generation_utils_conf,server_conf}/`
8. **pytest 自有流程**
   - `tests/conftest.py`（mock LLM、强制 local 后端）
   - `pyproject.toml` 的 `[tool.pytest.ini_options]`
   - `tests/integration/test_benchmark.py`（最贴近完整测试链路）

### 3.2 与"测试/评测"基本无关，可以忽略的文件

| 路径 | 为什么无关 |
| --- | --- |
| `docs/` 整个目录 | 项目主页 Next.js 站点，纯静态文档，与 Python 无关。 |
| `figs/` | 仅一张 README 封面图。 |
| `notebooks/` | 数据可视化教程；不是测试也不是 CI 的一部分。 |
| `stubs/` | 仅供 `mypy` 静态检查；运行/测试不会执行。 |
| `ui/` | Streamlit 前端，浏览数据库；不影响模拟与评测。 |
| `sotopia-chat/` | 独立的人机/多人 chat 服务（FastAPI/Streamlit 演示），不参与 benchmark。 |
| `.devcontainer/`、`.github/`、`.pre-commit-config.yaml`、`.gitignore`、`LICENSE`、`CODE_OF_CONDUCT.md`、`uv.lock` | 仓库基础设施 / 元数据。 |
| `sotopia/api/` 中除非你要测 HTTP/WS 服务，否则与命令行 benchmark 无关。 | API 层。 |
| `sotopia/cli/install/`、`scripts/migrate_*` | 一次性安装/数据迁移工具，跑过一次即可。 |
| `sotopia-rl/` 整个子项目 | 是另一篇论文（Sotopia-RL）的训练代码（SFT/RM/GRPO + Django 服务 + 标注），**与核心 Sotopia benchmark 不耦合**；只有它内部的 `evals/experiment_eval.py` 是用 Sotopia 评测训练后模型，可以视为"下游使用方"。 |
| `examples/experimental/` | 各种实验性多 Agent 游戏（werewolves, spyfall…），是 demo / 研究脚本，不是必走的测试链路；依赖 `aact` + Redis。 |
| `examples/fast_api_example.py`、`examples/generation_api/custom_model.py` | 仅是接入示例。 |
| `examples/fix_missing_episodes*.py`, `scripts/fix_missing_episodes_with_tag.sh` | 实验运维脚本（补跑漏 episode），非首次测试需要。 |
| `examples/generate_script.py` | 生成 script 模式数据，非主测试链路。 |
| `sotopia/experimental/` 与 `examples/experimental/` 相伴 | 仅在跑 aact 实验时用到。 |

> **快速判断方法**：只要某文件不在「§3.1 的核心流程文件 + 配置 + tests/」集合里，基本都是"周边/工具/文档"，与跑测试无关。

---

## 4. TL;DR 速查

- **想跑代码层面的 pytest**：
  ```bash
  uv sync --all-extras && uv run --extra test --extra api pytest tests
  ```
  默认 mock LLM、用 local 存储，无需 Redis、无需 API key。
- **想跑一个最小 LLM 模拟 demo**：
  ```bash
  uv run sotopia install
  uv run --env-file .env python examples/minimalist_demo.py
  ```
- **想跑官方多模型 benchmark**：
  ```bash
  uv run sotopia benchmark --models <m> --partner-model <p> --evaluator-model gpt-4o --task hard
  ```
  代码全在 `sotopia/cli/benchmark/benchmark.py`。
- **想合成自己的测试题**：用 `examples/generate_scenarios.py` + `sotopia/generation_utils/generate.py`。
- **流程总枢纽**：`sotopia/server.py` → `sotopia/envs/parallel.py` → `sotopia/envs/evaluators.py` → `sotopia/database/logs.py`。




# 运行说明

# python examples/minimalist_demo.py

下面给你一张完整的「调用时序表」，从你按下 `python examples/minimalist_demo.py` 那一刻开始，到 demo 跑完为止。每一行包含：**调用谁 / 在哪个文件 / 它做什么**。能省略的细节（warning、错误处理）已省。

---

## 阶段 0：进程启动 + import 副作用（在 `asyncio.run` 之前）

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 0.1 | `import sotopia.samplers` | `examples/minimalist_demo.py:11` | 触发 sotopia 包加载 |
| 0.2 | `sotopia/database/__init__.py` 顶层执行 | `sotopia/database/__init__.py:122-156` | 读 `os.environ["SOTOPIA_STORAGE_BACKEND"]`；你设的是 `local`，所以走 `LocalJSONBackend`，打印 `Using local JSON storage backend at /home/yphao/.sotopia/data` |
| 0.3 | `patch_model_for_local_storage(AgentProfile/EnvironmentProfile/...)` | `sotopia/database/persistent_profile.py:219-222` + `base_models.py:111-242` | 给 `AgentProfile.save()/.get()/.all()/.find()` 等方法打补丁，改成读写 `~/.sotopia/data/{ClassName}/{pk}.json` |
| 0.4 | `from sotopia.server import run_async_server` | `examples/minimalist_demo.py:12` | 导入主循环入口 |
| 0.5 | `logging.basicConfig(...)` + `RichHandler` | `examples/minimalist_demo.py:19-24` | 配置漂亮日志输出 |

> 这之后 demo 已经能"知道"本地数据库在哪，但还没读任何文件。

---

## 阶段 1：进入 `run_async_server`（事件循环开始）

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 1.1 | `asyncio.run(run_async_server(...))` | `examples/minimalist_demo.py:27-36` | 把 `run_async_server` 协程交给 asyncio 事件循环 |
| 1.2 | `run_async_server(model_dict, sampler, ...)` | `sotopia/server.py:274-382` | 总调度器；本次 `model_dict={"env","agent1","agent2"}` 全是 gpt-4o-mini，sampler 是 `UniformSampler` |
| 1.3 | `get_agent_class("gpt-4o-mini")` → `LLMAgent` | `sotopia/server.py:307-317` | 既不是 "human" 也不是 "redis"，所以两个 agent 都用 `LLMAgent`（位于 `sotopia/agents/llm_agent.py`） |
| 1.4 | 构造 `env_params` | `sotopia/server.py:325-337` | 给 ParallelSotopiaEnv 准备的参数。**关键两个 evaluator**：<br>• `RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2)`（`sotopia/envs/evaluators.py`）<br>• `EpisodeLLMEvaluator(model="gpt-4o-mini", EvaluationForAgents[SotopiaDimensions])` |
| 1.5 | `sampler.sample(...)` 返回 `env_agent_combo_iter` | `sotopia/server.py:346-356` | 真正的采样调用——见阶段 2 |

---

## 阶段 2：`UniformSampler.sample` —— 抽出"题目和角色"

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 2.1 | `EnvironmentProfile.all()` | `sotopia/samplers/uniform_sampler.py:53` → patched 版 `base_models.py:205-209` → `LocalJSONBackend.all()` (`storage_backend.py`) | **扫描 `~/.sotopia/data/EnvironmentProfile/*.json`** 把每个 JSON 反序列化为 `EnvironmentProfile` 对象。命中你 seed 进去那 1 条（咖啡店场景） |
| 2.2 | `AgentProfile.all()` | `uniform_sampler.py:59` 同链路 | 扫描 `~/.sotopia/data/AgentProfile/*.json` 拿到 Alex Johnson + Sam Kim 这 2 条 |
| 2.3 | `random.choice(env_candidates)` | `uniform_sampler.py:65` | 只有 1 条 env，所以选中那一条 |
| 2.4 | `ParallelSotopiaEnv(env_profile=...)` 构造 | `uniform_sampler.py:69`，类定义在 `sotopia/envs/parallel.py:216` 起 | 创建并行环境对象（继承自 PettingZoo `ParallelEnv`），保存场景、agent_goals、evaluator 列表等 |
| 2.5 | `random.sample(agent_candidates, 2)` + 创建 2 个 `LLMAgent` | `uniform_sampler.py:75-87` | 给两个 LLMAgent 实例分别绑定一个 AgentProfile 和 model_name="gpt-4o-mini" |
| 2.6 | `agent.goal = env.profile.agent_goals[i]` | `uniform_sampler.py:88-90` | 把每个 agent 的"社交目标"塞进它自己（来自 `EnvironmentProfile.agent_goals`） |
| 2.7 | `yield env, agents` | `uniform_sampler.py:92` | 返回一组 `(env, [agent1, agent2])` |

> 控制台你会看到这一行就是阶段 2 完成的标志：
> ```
> INFO - sotopia.samplers.uniform_sampler - Creating ParallelSotopiaEnv with 2 agents
> ```

---

## 阶段 3：`arun_one_episode` —— 单局对弈准备

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 3.1 | `arun_one_episode(env, agent_list, ...)` | `sotopia/server.py:118 起，被 357-368 行调用并 gather` | 运行单条 episode 的协程 |
| 3.2 | `Agents({a.agent_name: a for a in agent_list})` | `server.py:134` | 把 agent 列表包成名字 → 实例的字典 |
| 3.3 | `env.reset(agents=agents, omniscient=False)` | `server.py:139`，定义在 `sotopia/envs/parallel.py:216` | **关键 reset**：根据 EnvironmentProfile 生成 `ScriptBackground`（场景描述、两位的角色背景、目标），并塞给每个 agent 的 inbox。返回每个 agent 看到的初始 `Observation` |
| 3.4 | `agents.reset()` | `server.py:140` | 清空每个 LLMAgent 的内部状态（goal、inbox） |
| 3.5 | 把初始 observation 写进 `messages` 列表 | `server.py:145-151` | 这就是日志里 turn 0 的内容，第一条 `yield messages` |
| 3.6 | 给每个 agent 设 `goal` | `server.py:154-155` | 再赋一次（防止 reset 清掉了） |

---

## 阶段 4：主对弈循环（`while not done:`）

每一轮（最多 20 轮，见 `RuleBasedTerminatedEvaluator(max_turn_number=20)`）做：

### 4A. 两个 agent **并行**生成动作

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 4A.1 | `await asyncio.gather(*[agents[name].aact(obs) ...])` | `server.py:162-167` | 两个 agent 同时进入 LLM 调用 |
| 4A.2 | `LLMAgent.aact(obs)` | `sotopia/agents/llm_agent.py:64-103` | 单个 agent 的"思考一回合" |
| 4A.3 | `self.recv_message("Environment", obs)` | `llm_agent.py:65`（实现在 `base_agent.py`） | 把刚到的 observation 塞进自己的 inbox |
| 4A.4 | （首回合才走）`agenerate_goal(...)` | `llm_agent.py:68-73` → `sotopia/generation_utils/generate.py` | 用 LLM 推断/澄清自身 goal（首回合只调一次） |
| 4A.5 | `agenerate_action(model_name, history, turn_number, action_types, agent, goal, structured_output=True, ...)` | `llm_agent.py:90-102` → `sotopia/generation_utils/generate.py` | **真正打 LLM 的地方**。内部 → `agenerate(...)` → `litellm.acompletion(model="gpt-4o-mini", base_url=os.environ["OPENAI_BASE_URL"], api_key=...)` 走你 `.env` 里设的 `https://api.v3.cm/v1` |
| 4A.6 | LLM 返回 JSON，解析为 `AgentAction(action_type=speak/non-verbal/leave/action/none, argument=..., to=[...])` | `output_parsers.py` | 结构化输出 |
| 4A.7 | `AgentAction.model_validate(..., context={"agent_names":...,"sender":...})` | `server.py:182-186` | 校验 `to=` 收件人合法。失败则发回错误信息让 agent 重生成一次（4A.5 的重试） |

### 4B. 把动作交给环境推进一步

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 4B.1 | `await env.astep(agent_messages)` | `server.py:217`，定义在 `sotopia/envs/parallel.py:496-566` | 环境 turn++、记录所有 actions、跑 evaluator、生成下一轮 observation |
| 4B.2 | `self._process_incoming_actions(actions)` | `parallel.py:508` | 把 dict[name→Action] 整理成内部消息序列 |
| 4B.3 | `await self._run_evaluators(self.evaluators)` | `parallel.py:510` | 跑 `RuleBasedTerminatedEvaluator`：判定是否到达 `max_turn=20` 或连续 `max_stale_turn=2` 没新内容 → 决定本回合是否 `terminated` |
| 4B.4 | （仅 terminated 时）`await self._run_evaluators(self.terminal_evaluators)` | `parallel.py:513` | 跑 `EpisodeLLMEvaluator`：**再次调 LLM**（同一 model）按 `SotopiaDimensions` 的 7 个维度（goal/believability/relationship/knowledge/secret/social_rules/financial）打分，结构化输出 |
| 4B.5 | 计算下一轮 `action_mask`（round-robin 让两个 agent 轮流发言）+ 构造每个 agent 的 `Observation` | `parallel.py:522-555` | 决定下一回合谁能说话；`available_actions` 为 `["none"]` 表示这回合该轮空 |
| 4B.6 | 返回 `(obs, rewards, terminated, _, info)` | `parallel.py:557-566` | 给主循环 |

### 4C. 主循环把这一轮记录到 `messages` 并判 `done`

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 4C.1 | `messages.append([(env_name, agent_name, environment_messages[name]) ...])` | `server.py:218-223` | 写日志 |
| 4C.2 | `rewards.append(...) ; reasons.append(...)` | `server.py:225-228` | 累计 |
| 4C.3 | `done = all(terminated.values())` | `server.py:229` | 所有 agent 都 terminated 才退出循环 |

---

## 阶段 5：Episode 结束 → 写入 `EpisodeLog`

| # | 调用 | 位置 | 作用 |
| --- | --- | --- | --- |
| 5.1 | 构造 `EpisodeLog(...)` | `server.py:230-241` | 收集这一局所有信息：`environment` / `agents` / `models` / 整段 `messages`（每条转 natural language）/ `reasoning`（来自 LLM 评估器的 comments）/ `rewards`（每个 agent 的 complete_rating） |
| 5.2 | （`push_to_db=True` 时才执行）`epilog.save()` | `server.py:253-264`（patched 版 `base_models.py:128-144`） | 把整局存到 `~/.sotopia/data/EpisodeLog/{uuid}.json`<br>**注意**：`run_async_server(... push_to_db=False)` 默认是 False，所以 demo 默认**不写盘**，只在内存里跑完 |
| 5.3 | `flatten_listed_messages(last_messages)` | `server.py:271` | 把嵌套 messages 拍平返回 |
| 5.4 | 回到 `run_async_server` 的 `asyncio.gather(*episode_futures)` | `server.py:370-374` | 单局完成 |
| 5.5 | `return batch_results` | `server.py:382` | 整个 demo 退出 |

---

## 整张时序图（一图速查）

```
python examples/minimalist_demo.py
│
├─ import sotopia.samplers / .server          ← 触发 sotopia/database/__init__.py
│       └─ Using local JSON storage backend at ~/.sotopia/data
│
├─ asyncio.run( run_async_server(...) )       ← server.py:274
│       │
│       ├─ get_agent_class → LLMAgent         ← server.py:307
│       │
│       ├─ env_params = {evaluators=[Rule, EpisodeLLM]}   ← server.py:325
│       │
│       ├─ sampler.sample()                   ← uniform_sampler.py:18
│       │       ├─ EnvironmentProfile.all()   ← 读 ~/.sotopia/data/EnvironmentProfile/*.json
│       │       ├─ AgentProfile.all()         ← 读 ~/.sotopia/data/AgentProfile/*.json
│       │       ├─ ParallelSotopiaEnv(env_profile=...)
│       │       └─ LLMAgent × 2  + agent.goal=...
│       │
│       └─ arun_one_episode(env, agents)      ← server.py:118
│               │
│               ├─ env.reset()                 ← parallel.py:216  → ScriptBackground 写 inbox
│               │
│               └─ while not done:             ← server.py:158-229  (≤20 turns)
│                       │
│                       ├─ asyncio.gather( LLMAgent.aact(obs) × 2 )    ← llm_agent.py:64
│                       │       └─ agenerate_action(..., model="gpt-4o-mini")
│                       │             └─ litellm.acompletion()         ← generate.py:91
│                       │                   └─ HTTPS POST  https://api.v3.cm/v1/chat/completions
│                       │
│                       ├─ env.astep(actions)                          ← parallel.py:496
│                       │       ├─ RuleBasedTerminatedEvaluator        ← evaluators.py
│                       │       └─ (terminated 时) EpisodeLLMEvaluator  ← evaluators.py
│                       │             └─ 再调一次 LLM 打 7 维分数
│                       │
│                       └─ messages.append(...) ; done = all(terminated)
│
└─ EpisodeLog(messages, rewards, reasoning, ...)   ← server.py:230
        └─ (默认不 save，因为 push_to_db=False)
```

---

## 哪些 LLM 请求会被发出去（计费层面）

每局会有这么多次 `litellm.acompletion`：

| 请求 | 次数 | 何时 |
| --- | --- | --- |
| `agenerate_goal` | 2 | 仅首回合，每个 agent 一次 |
| `agenerate_action` | 2 × N | 每回合两个 agent 各一次（其中实际"该 agent 说话"的那个会真出 speak/non-verbal，另一个只能输出 `none`） |
| `EpisodeLLMEvaluator` 评分 | 1 | episode 结束时（结构化 7 维 JSON） |

`gpt-4o-mini` 一局对话 token 大约 5k-15k，按 v3.cm 的报价应该远不到 1 美分。

---

## TL;DR

```
minimalist_demo.py
  └─ run_async_server                  (sotopia/server.py)
        ├─ UniformSampler.sample       (sotopia/samplers/uniform_sampler.py) ── 读本地 JSON
        └─ arun_one_episode            (sotopia/server.py)
              ├─ env.reset             (sotopia/envs/parallel.py)            ── 生成场景 + 角色
              └─ loop:
                    ├─ LLMAgent.aact   (sotopia/agents/llm_agent.py)
                    │     └─ agenerate_action → litellm.acompletion (sotopia/generation_utils/generate.py)
                    │           └─ POST https://api.v3.cm/v1/chat/completions
                    └─ env.astep       (sotopia/envs/parallel.py)
                          └─ Rule + EpisodeLLM 评估器  (sotopia/envs/evaluators.py)
```

整个数据/控制流：**LocalJSON 后端 → UniformSampler 抽场景+角色 → 主循环交替 LLM 决策与环境步进 → 终局 7 维 LLM 打分 → 内存返回**（默认不写盘）。

如果想看真实跑出来的轨迹（每条 LLM 请求和回答），加这两行环境变量再跑：

```bash
export LITELLM_LOG=DEBUG
export PYTHONUNBUFFERED=1
SOTOPIA_STORAGE_BACKEND=local python examples/minimalist_demo.py
```