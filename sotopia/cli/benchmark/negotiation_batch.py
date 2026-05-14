"""长期谈判 **批量 LLM 评测** CLI（语义对齐 ``benchmark.benchmark``：多模型 × 并发 batch，但不改 ``benchmark.py``）。

================================================================================
运行评测 / 生成汇总 JSON：应调用哪些代码（文件 → 函数）与调用顺序
================================================================================

**入口（本文件）**

1. Typer 子命令 ``negotiation-batch`` → 函数 `negotiation_batch`（同上模块）。
   作用：解析命令行、构造 ``NegotiationTimelineParams``，委托批量评测，可选把整次 run 写入
   带时间戳的汇总 JSON（``aggregate_means`` + ``rows``）。可选 ``--run-config`` 指向 JSON，选用谈判 Agent 变体与
   记忆后端（见 ``negotiation_run_config.load_negotiation_run_config``）。

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

**程序化调用（不写每场 JSONL 时）**

- 单次：直接 import ``run_llm_negotiation_episode_evaluation``（或同步封装
   ``evaluate_long_term_negotiation_llm_sync``），见 ``llm_evaluation.py`` 模块注释。
- 批量：直接 import ``run_long_term_negotiation_eval_batch`` / ``*_async``，见 ``batch_evaluation.py``。

**控制台 / 诊断日志**

- 结构化评测结果：**-o/--output** 一次 run 一个 JSON（``aggregate_means`` + ``rows``），不是文本 log。
- 进度：``batch_evaluation.asyncio_gather_bounded`` 的 ``tqdm`` 写在 **stderr**。
- Episode 单行摘要：`sotopia.negotiation.batch` logger（``episode_start`` / ``episode_done``），需在 CLI 打开
  **--print-logs**（Rich 控制台）和/或 **--log-file**（UTF-8 纯文本追加）。实现：
  ``sotopia.settings.long_term_negotiation.eval_logging.configure_negotiation_cli_logging``。

控制台可执行入口见 ``python -m sotopia.cli.benchmark.negotiation_batch`` → `main()` → `app()`。
若传入 ``--artifact-root``，则 **每场 JSONL（model trace）与默认 negotiation 文本日志**
共用该根目录，并在其下按 ``{测试模型名}/{时间戳}/`` 嵌套（与分别传 ``--execution-trace-dir``
``--model-trace-dir`` 且指向同一根目录等价；单模型时落在同一叶子文件夹）。

若只传 ``--execution-trace-dir`` / ``--model-trace-dir``，默认同样在各自根下建
``{测试模型名}/{时间戳}/``（``--trace-flat`` 可关闭）。每场在该目录下仅追加 ``{tag}_<名字>.jsonl``
（参与者 + ``terminal_evaluator`` 等）；``*.execution.json`` 仅在单次 API ``write_execution_record=True`` 时写入，本 CLI 不传该开关。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from ..app import app

def _fmt(v: Any, precision: int = 4) -> str:
    """Safe float formatting."""
    if isinstance(v, (int, float)):
        return f"{v:.{precision}f}"
    return str(v)


def _print_evaluation_summary(rows: list[dict[str, Any]], aggregate_means: dict[str, Any]) -> None:
    """打印批量评测结果的重要汇总信息。"""
    n = len(rows)
    successes = sum(1 for r in rows if str(r.get("terminal") or "") == "success")
    timeouts = sum(1 for r in rows if str(r.get("terminal") or "") == "timeout")
    failures = sum(1 for r in rows if str(r.get("terminal") or "") == "failure")
    others = n - successes - timeouts - failures
    rm = aggregate_means.get("rule_metrics_mean") or {}

    # ── Header ──
    typer.echo("")
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_BLACK))
    typer.echo(typer.style("  EVALUATION RESULTS", fg=typer.colors.BRIGHT_CYAN, bold=True))
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_BLACK))

    # ── Terminal Status ──
    typer.echo(typer.style("\n── Terminal Status ──", fg=typer.colors.YELLOW, bold=True))
    typer.echo(f"  Episodes:           {n}")
    typer.echo(f"  Success:            {successes} ({_fmt(100 * successes / n if n else 0, 1)}%)")
    typer.echo(f"  Timeout:            {timeouts} ({_fmt(100 * timeouts / n if n else 0, 1)}%)")
    typer.echo(f"  Failure:            {failures} ({_fmt(100 * failures / n if n else 0, 1)}%)")
    if others:
        typer.echo(f"  Other:              {others}")

    # ── Final State Score ──
    fs = rm.get("negotiation_final_state_score")
    typer.echo(typer.style("\n── Final State Score (mean) ──", fg=typer.colors.YELLOW, bold=True))
    typer.echo(f"  Overall Score:      {_fmt(fs) if fs is not None else 'n/a'}")

    components = [
        ("terminal_success", "Terminal Success"),
        ("primary_contract", "Primary Contract"),
        ("solvency", "Solvency"),
        ("liquidity_preserved", "Liquidity Preserved"),
        ("predefined_rule", "Predefined Rule"),
        ("scheduling_effectiveness", "Scheduling Effectiveness"),
    ]
    for key, label in components:
        val = rm.get(f"negotiation_final_state_score_component_{key}")
        if val is not None:
            typer.echo(f"    {label:<28s} {_fmt(val)}")

    # ── Key Rule Metrics ──
    typer.echo(typer.style("\n── Rule Metrics (mean) ──", fg=typer.colors.YELLOW, bold=True))
    rule_keys = [
        ("negotiation_macro_steps_used", "Macro Steps"),
        ("negotiation_n_session_log", "Sessions"),
        ("negotiation_n_action_log", "Actions"),
        ("negotiation_n_message_log", "Messages"),
        ("negotiation_participant_mean_cash", "Mean Cash"),
        ("negotiation_participant_min_cash", "Min Cash"),
        ("negotiation_primary_contract_phase", "Contract Phase (0-4)"),
        ("negotiation_final_state_total_cash", "Final Total Cash"),
        ("negotiation_final_state_total_cash_delta", "Cash Delta"),
        ("negotiation_final_state_solvency_ratio", "Solvency Ratio"),
    ]
    for key, label in rule_keys:
        val = rm.get(key)
        if val is not None:
            typer.echo(f"  {label:<28s} {_fmt(val)}")

    # ── Per-agent Profit/Loss ──
    profit_keys = [k for k in rm if "individual_profit" in k or "company_profit" in k]
    if profit_keys:
        typer.echo(typer.style("\n── Profit / Loss (mean) ──", fg=typer.colors.YELLOW, bold=True))
        for k in sorted(profit_keys):
            typer.echo(f"  {k:<40s} {_fmt(rm[k])}")

    # ── Predefined Rule Details ──
    rule_detail_keys = [
        ("negotiation_predefined_rule_score", "Predef Rule Score"),
        ("negotiation_predefined_rule_realized_margin", "Realized Margin"),
        ("negotiation_predefined_rule_realized_price", "Realized Price"),
        ("negotiation_predefined_rule_reference_price", "Reference Price"),
        ("negotiation_predefined_rule_buyer_savings_ratio", "Buyer Savings Ratio"),
        ("negotiation_predefined_rule_total_profit", "Total Profit"),
        ("negotiation_predefined_rule_contract_value", "Contract Value"),
    ]
    shown = False
    for key, label in rule_detail_keys:
        val = rm.get(key)
        if val is not None:
            if not shown:
                typer.echo(typer.style("\n── Predefined Rule Details (mean) ──", fg=typer.colors.YELLOW, bold=True))
                shown = True
            typer.echo(f"  {label:<28s} {_fmt(val)}")

    # ── LLM Evaluation Scores ──
    llm_overall = aggregate_means.get("llm_overall_mean")
    llm_dims = aggregate_means.get("llm_dimension_scores_mean")
    if llm_overall or llm_dims:
        typer.echo(typer.style("\n── LLM Evaluation (mean) ──", fg=typer.colors.YELLOW, bold=True))
        if llm_overall and isinstance(llm_overall, dict):
            for k, v in llm_overall.items():
                typer.echo(f"  {k:<28s} {_fmt(v)}")
        if llm_dims and isinstance(llm_dims, dict):
            for agent_key, dims in llm_dims.items():
                if isinstance(dims, dict):
                    typer.echo(f"  [{agent_key}]")
                    for dk, dv in dims.items():
                        typer.echo(f"    {dk:<26s} {_fmt(dv)}")

    typer.echo("")
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_BLACK))
    typer.echo(typer.style(f"  Done. {n} episodes | {successes} success | "
                           f"mean score={_fmt(fs) if fs is not None else 'n/a'}",
                           fg=typer.colors.GREEN, bold=True))
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_BLACK))
    typer.echo("")


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
        typer.Option(
            "--quartet",
            help="无场景时：4 名参与者 + strict_design_v1；有场景时仍写入元数据提示，人数以场景/ --num-participants 为准",
        ),
    ] = False,
    num_participants: Annotated[
        int | None,
        typer.Option(
            "--num-participants",
            help="交互的 canonical 角色数 2–4（与 types.SESSION_SPEAKER_ROLE_ORDER 前缀一致）；"
            "默认无场景时由 --quartet 决定，有场景时读 game_metadata；指定则覆盖场景",
        ),
    ] = None,
    skip_llm_scoring: Annotated[
        bool,
        typer.Option("--skip-llm-scoring", help="跳过 EpisodeLLMEvaluator，仅跑环境+智能体"),
    ] = False,
    max_macro_steps: Annotated[int, typer.Option(help="单次 episode 宏观步上限")] = 3500,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "结果写入独立 JSON 文件（每次 run 单独文件，自动附加时间戳；"
                "使用可读缩进格式，不与历史 run 混写）"
            ),
        ),
    ] = None,
    tag: Annotated[
        str,
        typer.Option(help="实验 tag 前缀（写入每条 record 的 experiment_tag 组成部分）"),
    ] = "",
    print_logs: Annotated[bool, typer.Option(help="Rich 控制台日志（INFO；含每条 episode 起止单行摘要）")] = False,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            help="追加写入 UTF-8 纯文本 log（episode 摘要等与 --print-logs 同源 logger；无 ANSI）",
        ),
    ] = None,
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
    run_config: Annotated[
        Path | None,
        typer.Option(
            "--run-config",
            help=(
                "JSON：谈判 Agent / 记忆后端等（见 sotopia.settings.long_term_negotiation.negotiation_run_config；"
                "示例见同目录 run_config_examples/*.json）"
            ),
        ),
    ] = None,
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help=(
                "统一产物根目录：每场 JSONL（model trace）与默认 negotiation_batch.log "
                "写入其下 {测试模型名}/{时间戳}/（单模型时同一文件夹；与 --execution-trace-dir 等二选一优先本项）"
            ),
        ),
    ] = None,
    execution_trace_dir: Annotated[
        Path | None,
        typer.Option(
            "--execution-trace-dir",
            help=(
                "与 --model-trace-dir 二选一或同时传：用于每场 JSONL（{tag}_<名字>.jsonl）的根目录；"
                "若未传 --model-trace-dir 则仅此目录接收 trace。"
                "默认嵌套 {根}/{测试模型名}/{时间戳}/（--trace-flat 可关）；--artifact-root 优先时本项忽略。"
                "*.execution.json 等仅在 API write_execution_record=True 时写入，CLI 批量默认不写。"
            ),
        ),
    ] = None,
    model_trace_dir: Annotated[
        Path | None,
        typer.Option(
            "--model-trace-dir",
            help="每场 LLM trace JSONL 根目录；默认 {根}/{测试模型名}/{时间戳}/（见 --trace-flat）；若已设 --artifact-root 则忽略",
        ),
    ] = None,
    trace_flat: Annotated[
        bool,
        typer.Option(
            "--trace-flat",
            help="不做 模型名/时间戳 子目录，直接把文件写入 --execution-trace-dir / --model-trace-dir 根路径",
        ),
    ] = False,
) -> None:
    """并行跑多组长期谈判 episode + 可选终局评测，输出 JSON 可聚合记录。

    评测主链路参见本模块顶层 docstring；此处仅负责 Typer IO 并调用
    ``run_long_term_negotiation_eval_batch``。
    """
    from sotopia.settings import NegotiationTimelineParams
    from sotopia.settings.long_term_negotiation.batch_evaluation import (
        aggregate_negotiation_eval_run_means,
        run_long_term_negotiation_eval_batch,
    )
    from sotopia.settings.long_term_negotiation.eval_logging import (
        configure_negotiation_cli_logging,
        negotiation_batch_log_under_artifact_root,
        negotiation_default_batch_log_file,
    )
    from sotopia.settings.long_term_negotiation.negotiation_run_config import (
        load_negotiation_run_config,
    )
    from sotopia.settings.long_term_negotiation.scenario_loader import (
        environment_pks_from_manifest,
    )

    models = agent_models if agent_models is not None else ["gpt-4o-mini"]
    run_started_dt = datetime.now()
    run_started_at = run_started_dt.strftime("%Y-%m-%d %H:%M:%S")
    ts = run_started_dt.strftime("%Y%m%d_%H%M%S")

    eff_exec: Path | None = execution_trace_dir
    eff_model: Path | None = model_trace_dir
    if artifact_root is not None:
        if execution_trace_dir is not None or model_trace_dir is not None:
            typer.echo(
                typer.style(
                    "[negotiation-batch] --artifact-root set: ignoring --execution-trace-dir / --model-trace-dir.",
                    fg=typer.colors.YELLOW,
                ),
                err=True,
            )
        eff_exec = artifact_root
        eff_model = artifact_root

    effective_log_file: Path
    if log_file is None:
        if artifact_root is not None:
            effective_log_file = negotiation_batch_log_under_artifact_root(artifact_root, models, ts)
        else:
            effective_log_file = negotiation_default_batch_log_file(models, ts)
    else:
        lf = Path(log_file)
        if str(lf).endswith("/") or (lf.suffix == "" and not lf.name.lower().endswith(".log")):
            effective_log_file = lf / f"negotiation_batch_{ts}.log"
        else:
            effective_log_file = lf

    configure_negotiation_cli_logging(verbose_console=print_logs, log_file=effective_log_file)
    tag_base = tag.strip() if tag.strip() else "negotiation_eval_batch"
    typer.echo(
        typer.style(
            f"[negotiation-batch] run_started_at={run_started_at} log_file={effective_log_file}",
            fg=typer.colors.CYAN,
        )
    )

    negotiation_run_cfg: dict[str, Any] | None = None
    if run_config is not None:
        negotiation_run_cfg = load_negotiation_run_config(run_config)
        mem = negotiation_run_cfg.get("memory") if isinstance(negotiation_run_cfg.get("memory"), dict) else {}
        typer.echo(
            typer.style(
                f"[negotiation-batch] run-config={run_config} agent={negotiation_run_cfg.get('negotiation_agent')} "
                f"memory.backend={mem.get('backend', 'plain')}",
                fg=typer.colors.CYAN,
            )
        )

    if num_participants is not None and not (2 <= num_participants <= 4):
        typer.echo(typer.style("--num-participants must be between 2 and 4", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)

    scenario_pks: list[str] = []
    scenario_pks.extend(scenario_env_pk or [])
    if scenario_manifest is not None:
        scenario_pks.extend(environment_pks_from_manifest(scenario_manifest))
    scenario_pks = list(dict.fromkeys(scenario_pks))

    if scenario_pks and quartet:
        typer.echo(
            typer.style(
                "[negotiation-batch] --quartet ignored for sizing: roster size comes from "
                "each scenario's game_metadata (use --num-participants to override).",
                fg=typer.colors.YELLOW,
            ),
            err=True,
        )
    if scenario_pks and num_participants is not None:
        typer.echo(
            typer.style(
                "[negotiation-batch] --num-participants overrides stored num_participants for every scenario job.",
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
            f"num_participants={num_participants}, scenarios={len(scenario_pks)} env pk(s)",
            fg=typer.colors.CYAN,
        )
    )
    try:
        rows = run_long_term_negotiation_eval_batch(
            agent_models=models,
            evaluator_model=evaluator_model,
            quartet=False if scenario_pks else quartet,
            num_participants=num_participants,
            repeats_per_model=repeats,
            batch_size=batch_size,
            params=params,
            scenario_environment_pks=scenario_pks or None,
            max_macro_steps=max_macro_steps,
            run_terminal_llm_eval=not skip_llm_scoring,
            experiment_tag_base=tag_base,
            negotiation_run_config=negotiation_run_cfg,
            execution_trace_dir=eff_exec,
            model_trace_dir=eff_model,
            nest_trace_dirs_by_model_time=not trace_flat,
            run_timestamp=ts,
        )
        aggregate_means = aggregate_negotiation_eval_run_means(rows)
    except Exception as exc:  # pragma: no cover
        typer.echo(typer.style(str(exc), fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1) from exc

    if output is not None:
        out_base = Path(output)
        # 约定：--output 可以给目录或文件名。两者都写成“本次 run 独立文件”。
        # - 目录（或无后缀）：<dir>/negotiation_eval_<tag>_<ts>.json
        # - 文件：<parent>/<stem>_<ts>.json
        if out_base.exists() and out_base.is_dir():
            out_dir = out_base
            out_file = out_dir / f"negotiation_eval_{tag_base}_{ts}.json"
        elif out_base.suffix == "":
            out_dir = out_base
            out_file = out_dir / f"negotiation_eval_{tag_base}_{ts}.json"
        else:
            out_dir = out_base.parent
            out_file = out_dir / f"{out_base.stem}_{ts}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_started_at": run_started_at,
            "run_timestamp": ts,
            "tag": tag_base,
            "agent_models": models,
            "evaluator_model": evaluator_model,
            "aggregate_means": aggregate_means,
            "rows": rows,
        }
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(typer.style(f"Saved {len(rows)} records to {out_file}", fg=typer.colors.GREEN))

    successes = sum(1 for r in rows if r.get("terminal") == "success")
    rm = aggregate_means.get("rule_metrics_mean") or {}
    _print_evaluation_summary(rows, aggregate_means)


def main() -> None:
    """``python -m sotopia.cli.benchmark.negotiation_batch`` 时的入口。"""
    app()


if __name__ == "__main__":
    main()
