#!/usr/bin/env python
"""长期谈判批量评测 — V2 扩展版入口。

绕过 CLI（sotopia.cli.benchmark.negotiation_batch），直接调用
``run_long_term_negotiation_eval_batch_async_multi_extended``，
使用 ExtendedLongTermNegotiationEnv + V2 deterministic payouts。

用法::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/run_long_term_negotiation_eval_batch_v2.py \\
        --agent-model "custom/deepseek-v4-flash@https://api.deepseek.com" \\
        --evaluator-model "custom/deepseek-v4-flash@https://api.deepseek.com" \\
        --batch-size 8 --repeats 1 \\
        --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \\
        --run-config sotopia/settings/long_term_negotiation/run_config_examples/summarizing_memory.json \\
        --execution-trace-dir runs/execution_traces \\
        --output runs/ \\
        --tag ltr_multi_firm_llm_v1

与原始 CLI 的参数对齐，自动处理时间戳、日志、汇总 JSON。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")


def _fmt(v: Any, precision: int = 4) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.{precision}f}"
    return str(v)


def print_evaluation_summary(
    rows: list[dict[str, Any]], aggregate_means: dict[str, Any]
) -> None:
    """打印批量评测汇总（与 CLI 格式一致）。"""
    n = len(rows)
    successes = sum(1 for r in rows if str(r.get("terminal") or "") == "success")
    timeouts = sum(1 for r in rows if str(r.get("terminal") or "") == "timeout")
    failures = sum(1 for r in rows if str(r.get("terminal") or "") == "failure")
    others = n - successes - timeouts - failures
    rm = aggregate_means.get("rule_metrics_mean") or {}

    print("")
    print("=" * 64)
    print("  EVALUATION RESULTS (V2 Extended)")
    print("=" * 64)

    print("\n── Terminal Status ──")
    print(f"  Episodes:           {n}")
    print(f"  Success:            {successes} ({_fmt(100 * successes / n if n else 0, 1)}%)")
    print(f"  Timeout:            {timeouts} ({_fmt(100 * timeouts / n if n else 0, 1)}%)")
    print(f"  Failure:            {failures} ({_fmt(100 * failures / n if n else 0, 1)}%)")
    if others:
        print(f"  Other:              {others}")

    fs = rm.get("negotiation_final_state_score")
    print("\n── Final State Score (mean) ──")
    print(f"  Overall Score:      {_fmt(fs) if fs is not None else 'n/a'}")

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
            print(f"    {label:<28s} {_fmt(val)}")

    print("\n── Rule Metrics (mean) ──")
    rule_keys = [
        ("negotiation_macro_steps_used", "Macro Steps"),
        ("negotiation_n_session_log", "Sessions"),
        ("negotiation_n_action_log", "Actions"),
        ("negotiation_n_message_log", "Messages"),
        ("negotiation_participant_mean_cash", "Mean Cash"),
        ("negotiation_participant_min_cash", "Min Cash"),
        ("negotiation_final_state_total_cash", "Final Total Cash"),
        ("negotiation_final_state_total_cash_delta", "Cash Delta"),
        ("negotiation_final_state_solvency_ratio", "Solvency Ratio"),
    ]
    for key, label in rule_keys:
        val = rm.get(key)
        if val is not None:
            print(f"  {label:<28s} {_fmt(val)}")

    profit_keys = [k for k in rm if "individual_profit" in k or "company_profit" in k]
    if profit_keys:
        print("\n── Profit / Loss (mean) ──")
        for k in sorted(profit_keys):
            print(f"  {k:<40s} {_fmt(rm[k])}")

    v2_keys = [
        "negotiation_predefined_rule_payout_mode_v2_cooperation",
        "negotiation_predefined_rule_payout_mode_v2_buy_sell",
        "negotiation_predefined_rule_cost_price",
        "negotiation_predefined_rule_seller_earnings_per_unit",
    ]
    shown = False
    for key in v2_keys:
        val = rm.get(key)
        if val is not None and val != 0.0:
            if not shown:
                print("\n── V2-Specific Metrics (mean) ──")
                shown = True
            print(f"  {key:<50s} {_fmt(val)}")

    print("")
    print("=" * 64)
    print(f"  Done. {n} episodes | {successes} success | "
          f"mean score={_fmt(fs) if fs is not None else 'n/a'}")
    print("=" * 64)
    print("")


async def main_async(args: argparse.Namespace) -> int:
    from sotopia.settings.long_term_negotiation.batch_evaluation import (
        aggregate_negotiation_eval_run_means,
    )
    from sotopia.settings.long_term_negotiation.batch_evaluation_multi import (
        run_long_term_negotiation_eval_batch_async_multi_extended,
    )
    from sotopia.settings.long_term_negotiation.negotiation_run_config import (
        load_negotiation_run_config,
    )
    from sotopia.settings.long_term_negotiation.scenario_loader import (
        environment_pks_from_manifest,
    )

    models = args.agent_models if args.agent_models else ["gpt-4o-mini"]
    run_started_dt = datetime.now()
    run_started_at = run_started_dt.strftime("%Y-%m-%d %H:%M:%S")
    ts = run_started_dt.strftime("%Y%m%d_%H%M%S")
    tag_base = args.tag.strip() if args.tag.strip() else "negotiation_eval_batch_v2"

    # Resolve scenario PKs
    scenario_pks: list[str] = []
    if args.scenario_env_pk:
        scenario_pks.extend(args.scenario_env_pk)
    if args.scenario_manifest:
        manifest_path = Path(args.scenario_manifest).expanduser()
        if manifest_path.exists():
            scenario_pks.extend(environment_pks_from_manifest(manifest_path))
    scenario_pks = list(dict.fromkeys(scenario_pks))

    # Negotiation run config
    negotiation_run_cfg: dict[str, Any] | None = None
    if args.run_config:
        negotiation_run_cfg = load_negotiation_run_config(Path(args.run_config))
        mem = negotiation_run_cfg.get("memory") if isinstance(negotiation_run_cfg.get("memory"), dict) else {}
        print(
            f"[v2-batch] run-config={args.run_config} "
            f"agent={negotiation_run_cfg.get('negotiation_agent')} "
            f"memory.backend={mem.get('backend', 'plain')}"
        )

    # Trace dirs
    exec_dir: str | None = None
    if args.execution_trace_dir:
        exec_dir = str(Path(args.execution_trace_dir).resolve())

    print(
        f"[v2-batch] run_started_at={run_started_at}\n"
        f"  agent_models={models}, evaluator={args.evaluator_model}\n"
        f"  batch_size={args.batch_size}, repeats={args.repeats}\n"
        f"  scenarios={len(scenario_pks)} env pk(s)\n"
        f"  extended_env=True, v2_deterministic_payouts=True"
    )

    try:
        rows = await run_long_term_negotiation_eval_batch_async_multi_extended(
            agent_models=models,
            evaluator_model=args.evaluator_model,
            repeats_per_model=args.repeats,
            batch_size=args.batch_size,
            scenario_environment_pks=scenario_pks or None,
            max_macro_steps=args.max_macro_steps,
            run_terminal_llm_eval=not args.skip_llm_scoring,
            experiment_tag_base=tag_base,
            negotiation_run_config=negotiation_run_cfg,
            execution_trace_dir=exec_dir,
            model_trace_dir=exec_dir,
            nest_trace_dirs_by_model_time=not args.trace_flat,
            run_timestamp=ts,
        )
        aggregate_means = aggregate_negotiation_eval_run_means(rows)
    except Exception as exc:
        print(f"[v2-batch] ERROR: {exc}", file=sys.stderr)
        raise

    # Output JSON
    if args.output:
        out = Path(args.output)
        if out.is_dir():
            out_file = out / f"negotiation_eval_{tag_base}_{ts}.json"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out_file = out.parent / f"{out.stem}_{ts}.json"
        payload = {
            "run_started_at": run_started_at,
            "run_timestamp": ts,
            "tag": tag_base,
            "agent_models": models,
            "evaluator_model": args.evaluator_model,
            "extended_eval": True,
            "v2_deterministic_payouts": True,
            "aggregate_means": aggregate_means,
            "rows": rows,
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[v2-batch] Saved {len(rows)} records to {out_file}")

    print_evaluation_summary(rows, aggregate_means)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="V2 extended negotiation batch evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--agent-model", "-m", action="append", dest="agent_models",
        help="参与者 LLM（可重复指定多个）",
    )
    ap.add_argument(
        "--evaluator-model", "-e", default="gpt-4o-mini",
        help="终局 LLM 评测模型",
    )
    ap.add_argument(
        "--batch-size", "-b", type=int, default=3,
        help="并发上限",
    )
    ap.add_argument(
        "--repeats", "-r", type=int, default=1,
        help="每个 agent 模型重复跑的 episode 次数",
    )
    ap.add_argument(
        "--skip-llm-scoring", action="store_true",
        help="跳过 EpisodeLLMEvaluator",
    )
    ap.add_argument(
        "--max-macro-steps", type=int, default=3500,
        help="单次 episode 宏观步上限",
    )
    ap.add_argument(
        "--output", "-o", default=None,
        help="汇总 JSON 输出目录（自动附加时间戳文件名）",
    )
    ap.add_argument("--tag", default="", help="实验 tag 前缀")
    ap.add_argument(
        "--scenario-env-pk", action="append", dest="scenario_env_pk",
        help="从存储加载的 EnvironmentProfile.pk（可重复）",
    )
    ap.add_argument(
        "--scenario-manifest", default=None,
        help="manifest JSON 路径（如 ~/.sotopia/data/long_term_negotiation_llm_manifest.json）",
    )
    ap.add_argument(
        "--run-config", default=None,
        help="JSON 谈判 Agent / 记忆后端配置",
    )
    ap.add_argument(
        "--execution-trace-dir", default=None,
        help="每场 JSONL trace 根目录",
    )
    ap.add_argument(
        "--trace-flat", action="store_true",
        help="不做 模型名/时间戳 子目录",
    )
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
