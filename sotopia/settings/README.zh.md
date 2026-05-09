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
| `negotiation_metrics.py` | 规则向标量指标 `compute_negotiation_rule_metrics`。 |
| `dummy_agents.py` | 规则策略 dummy，用于无 LLM 冒烟或对照实验。 |
| `negotiation_llm_agent.py` | `NegotiationSocialLLMAgent` 与 `build_negotiation_social_llm_agents`（结构化 JSON 动作）。 |
| `llm_evaluation.py` | **单次** LLM episode + 可选终局 `EpisodeLLMEvaluator`：`run_llm_negotiation_episode_evaluation`。 |
| `batch_evaluation.py` | **批量**异步调度、并发限流、JSONL 友好记录：`run_long_term_negotiation_eval_batch`；支持 ``scenario_environment_pks``。 |
| `scenario_loader.py` | 从 ``EnvironmentProfile.game_metadata`` 还原 ``NegotiationTimelineParams`` / quartet / strict 标志。 |
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

1. `negotiation_batch` 解析参数；无场景 pk 时构造默认 `NegotiationTimelineParams`，有场景 pk 时单场参数由存储层提供。
2. 调用 `sotopia.settings.long_term_negotiation.batch_evaluation.run_long_term_negotiation_eval_batch`。
3. 内部 `run_long_term_negotiation_eval_batch_async` 对每个 `(agent_model, repeat)` 限流并发，单任务调用 `run_llm_negotiation_episode_evaluation`。
4. 汇总为 dict 列表，可选 `-o/--output` 追加写入 **JSONL**。

无安装时用模块方式（需将 `social_env` 加入 `PYTHONPATH`）可参考该文件末尾的 `python -m sotopia.cli.benchmark.negotiation_batch` 说明。

**从存储加载题库场景（与 `benchmark_v2_data_models` 生成的 `EnvironmentProfile` 对齐）**：先用
`scripts/generate_long_term_negotiation_scenarios.py` 写入 `~/.sotopia/data/`，再在执行时传
`--scenario-manifest ~/.sotopia/data/long_term_negotiation_manifest.json` 和/或多次
`--scenario-env-pk <pk>`。此时每条 episode 会从 profile 的 ``game_metadata.timeline`` 构造
`NegotiationTimelineParams`，roster 宽窄由各场景的 ``quartet`` 字段决定；CLI 的 ``--quartet`` 失效。

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

1. `default_negotiation_roster` → 确定参与者顺序。  
2. `build_llm_negotiation_agents` → `build_negotiation_social_llm_agents`。  
3. 构造 `LongTermNegotiationEnv`，`await env.run_episode_async(...)`。  
4. `compute_negotiation_rule_metrics`。  
5. 若开启终局 LLM：`format_negotiation_episode_for_llm_eval` → `EpisodeLLMEvaluator` → `unweighted_aggregate_evaluate`。

## 规则 agent / Dummy 对照

不参与 LLM API 的快速路径：使用 `long_term_negotiation.build_rule_dummy_agents` 等（见 `settings` 顶层 `__init__.py` 导出），环境与控制器仍可走 `LongTermNegotiationEnv`，但不会经过 `llm_evaluation` 的评测管线。
