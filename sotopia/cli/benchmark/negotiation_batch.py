"""长期谈判 **批量 LLM 评测** CLI（语义对齐 ``benchmark.benchmark``：多模型 × 并发 batch，但不改 ``benchmark.py``）。

================================================================================
运行评测 / 生成 JSONL：应调用哪些代码（文件 → 函数）与调用顺序
================================================================================

**入口（本文件）**

1. Typer 子命令 ``negotiation-batch`` → 函数 `negotiation_batch`（同上模块）。
   作用：解析命令行、构造 ``NegotiationTimelineParams``，委托批量评测，可选把每条
   episode 的记录追加写入 JSONL。

**下游（必读顺序）**

2. ``sotopia.settings.long_term_negotiation.batch_evaluation.run_long_term_negotiation_eval_batch``
   （文件 ``batch_evaluation.py``）
   作用：同步壳，内部 ``asyncio.run(... run_long_term_negotiation_eval_batch_async ...)``。

3. ``run_long_term_negotiation_eval_batch_async``
   （同文件 ``batch_evaluation.py``）
   作用：为每个 (agent_model × repeats) 建协程，用 ``asyncio_gather_bounded`` 限流并发，
   单任务内调用单次评测。

    对每个任务：`uniform_negotiation_model_dict` 拼 ``model_dict`` →
    await ``run_llm_negotiation_episode_evaluation`` → `build_eval_record` 转成可 JSON
    的 dict → ``negotiation_eval_record_to_jsonable`` 兜底序列化类型。

**单次评测核心（环境与 LLM）**

4. ``sotopia.settings.long_term_negotiation.llm_evaluation.run_llm_negotiation_episode_evaluation``
   （文件 ``llm_evaluation.py``）
   作用：单次 episode：``build_llm_negotiation_agents`` → ``LongTermNegotiationEnv``
   （``env.py``）→ ``await env.run_episode_async`` → ``compute_negotiation_rule_metrics``；
   若 ``run_terminal_llm_eval=True``：``format_negotiation_episode_for_llm_eval`` →
   ``EpisodeLLMEvaluator`` → ``unweighted_aggregate_evaluate``。

**程序化调用（不写 JSONL）**

- 单次：直接 import ``run_llm_negotiation_episode_evaluation``（或同步封装
   ``evaluate_long_term_negotiation_llm_sync``），见 ``llm_evaluation.py`` 模块注释。
- 批量：直接 import ``run_long_term_negotiation_eval_batch`` / ``*_async``，见 ``batch_evaluation.py``。

控制台可执行入口见 ``python -m sotopia.cli.benchmark.negotiation_batch`` → `main()` → `app()`。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from ..app import app


@app.command("negotiation-batch")
def negotiation_batch(
    agent_models: Annotated[
        list[str] | None,
        typer.Option(
            "--agent-model",
            "-m",
            help="参与者 LLM（可重复指定多个，与主 benchmark 的 --models 类似；默认 gpt-4o-mini）",
        ),
    ] = None,
    evaluator_model: Annotated[
        str,
        typer.Option(
            "--evaluator-model",
            "-e",
            help="终局 LLM 评测模型（写入 model_dict['env']）",
        ),
    ] = "gpt-4o-mini",
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="并发上限（等价于主 benchmark 中的 batch_size）",
        ),
    ] = 3,
    repeats: Annotated[
        int,
        typer.Option("--repeats", "-r", help="每个 agent 模型重复跑的 episode 次数"),
    ] = 1,
    quartet: Annotated[
        bool,
        typer.Option("--quartet", help="四方 strict_design_v1 roster"),
    ] = False,
    skip_llm_scoring: Annotated[
        bool,
        typer.Option("--skip-llm-scoring", help="跳过 EpisodeLLMEvaluator，仅跑环境+智能体"),
    ] = False,
    max_macro_steps: Annotated[int, typer.Option(help="单次 episode 宏观步上限")] = 3500,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="结果追加写入 JSONL（每行一条）"),
    ] = None,
    tag: Annotated[
        str,
        typer.Option(help="实验 tag 前缀（写入每条 record 的 experiment_tag 组成部分）"),
    ] = "",
    print_logs: Annotated[bool, typer.Option(help="Rich 控制台日志")] = False,
    scenario_env_pk: Annotated[
        list[str] | None,
        typer.Option(
            "--scenario-env-pk",
            help="从存储加载的 EnvironmentProfile.pk（可重复该选项）",
        ),
    ] = None,
    scenario_manifest: Annotated[
        Path | None,
        typer.Option("--scenario-manifest", help="manifest JSON（如 ~/.sotopia/data/long_term_negotiation_manifest.json）"),
    ] = None,
) -> None:
    """并行跑多组长期谈判 episode + 可选终局评测，输出 JSON 可聚合记录。

    评测主链路参见本模块顶层 docstring；此处仅负责 Typer IO 并调用
    ``run_long_term_negotiation_eval_batch``。
    """
    from sotopia.settings import NegotiationTimelineParams
    from sotopia.settings.long_term_negotiation.batch_evaluation import (
        run_long_term_negotiation_eval_batch,
    )
    from sotopia.settings.long_term_negotiation.scenario_loader import (
        environment_pks_from_manifest,
    )

    models = agent_models if agent_models is not None else ["gpt-4o-mini"]
    log_level = logging.INFO if print_logs else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()] if print_logs else [logging.NullHandler()],
    )
    tag_base = tag.strip() if tag.strip() else "negotiation_eval_batch"

    scenario_pks: list[str] = []
    scenario_pks.extend(scenario_env_pk or [])
    if scenario_manifest is not None:
        scenario_pks.extend(environment_pks_from_manifest(scenario_manifest))
    scenario_pks = list(dict.fromkeys(scenario_pks))

    if scenario_pks and quartet:
        typer.echo(
            typer.style(
                "[negotiation-batch] --quartet ignored: roster comes from each scenario's EnvironmentProfile.",
                fg=typer.colors.YELLOW,
            ),
            err=True,
        )

    params: NegotiationTimelineParams | None
    if scenario_pks:
        params = None
    else:
        params = NegotiationTimelineParams(
            D=8,
            s_max_per_day=2,
            max_session_rounds=32,
            max_total_turns_per_session=64,
        )

    typer.echo(
        typer.style(
            f"negotiation-batch: agent_models={models}, evaluator={evaluator_model}, "
            f"batch_size={batch_size}, repeats={repeats}, quartet={quartet}, "
            f"scenarios={len(scenario_pks)} env pk(s)",
            fg=typer.colors.CYAN,
        )
    )
    try:
        rows = run_long_term_negotiation_eval_batch(
            agent_models=models,
            evaluator_model=evaluator_model,
            quartet=False if scenario_pks else quartet,
            repeats_per_model=repeats,
            batch_size=batch_size,
            params=params,
            scenario_environment_pks=scenario_pks or None,
            max_macro_steps=max_macro_steps,
            run_terminal_llm_eval=not skip_llm_scoring,
            experiment_tag_base=tag_base,
        )
    except Exception as exc:  # pragma: no cover
        typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1) from exc

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(typer.style(f"Appended {len(rows)} lines to {output}", fg=typer.colors.GREEN))

    successes = sum(1 for r in rows if r.get("terminal") == "success")
    typer.echo(f"Done. episodes={len(rows)}, terminal_success_count={successes}")


def main() -> None:
    """``python -m sotopia.cli.benchmark.negotiation_batch`` 时的入口。"""
    app()


if __name__ == "__main__":
    main()
