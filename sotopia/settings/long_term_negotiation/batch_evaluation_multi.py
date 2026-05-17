"""长期谈判批量评测的多智能体扩展。

通过 **继承** ``ScriptEnvironmentResponse`` 增加 ``p3_rate`` / ``p4_rate`` 字段，
并提供对应的聚合函数与 record 构建函数，使 LLM 终局评分覆盖 **全部 N 名参与者**
（而不仅仅是前 2 名）。

用法示例（替换 CLI 中的原始调用）::

    from sotopia.settings.long_term_negotiation.batch_evaluation_multi import (
        run_long_term_negotiation_eval_batch_async_multi,
    )
    rows = await run_long_term_negotiation_eval_batch_async_multi(
        agent_models=[...],
        evaluator_model=...,
        ...
    )

设计原则：
- 不修改原始文件（``message_classes.py`` / ``evaluators.py`` / ``batch_evaluation.py`` / ``llm_evaluation.py``）。
- ``MultiAgentScriptEnvironmentResponse`` **是** ``ScriptEnvironmentResponse``（继承），可直接用于所有
  接受 ``ScriptEnvironmentResponse`` 的上下游。
- ``_mean_llm_dimension_scores`` / ``_print_evaluation_summary`` 等下游消费者已原生支持任意
  agent key，因此只需打通数据携带链路即可。
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any

from pydantic import Field, validate_call

from sotopia.messages.message_classes import ScriptEnvironmentResponse

from .batch_evaluation import (
    _finite_number,
    _mean_llm_dimension_scores,
    _mean_llm_overall_from_aggregate,
    _mean_numeric_fields_across_rows,
    build_eval_record,
    negotiation_eval_record_to_jsonable,
)
from .llm_evaluation import LongTermNegotiationEvalResult


# ============================================================
# _reduce 内联（避免 evaluators 模块的 gin 依赖）
# ============================================================
@validate_call
def _reduce(
    responses_per_reducer: list[tuple[tuple[str, float | int | bool], str]],
) -> tuple[dict[str, float | int | bool], str]:
    """与 ``sotopia.envs.evaluators._reduce`` 等价，内联以避免导入链依赖 ``gin``。"""
    responses_dict: dict[str, list[float | int | bool]] = defaultdict(list)
    comments_dict: dict[str, str] = defaultdict(str)
    reduced_dict: dict[str, float | int | bool] = {}
    for response, reasoning in responses_per_reducer:
        responses_dict[response[0]].append(response[1])
        comments_dict[response[0]] += reasoning
    scores: list[float | int] = []
    for k, v in responses_dict.items():
        if k == "terminated":
            assert all(isinstance(x, bool) for x in v)
            reduced_dict[k] = any(v)
        else:
            assert all(isinstance(x, (float, int)) for x in v)
            reduced_dict[k] = sum(v) / len(v)
            scores.append(reduced_dict[k])
    if len(scores) and "overall_score" not in responses_dict:
        scores = [x for x in scores if x is not None]
        reduced_dict["overall_score"] = sum(scores) / len(scores)
    comments = "\n".join([f"{k}: {v}" for k, v in comments_dict.items()])
    return reduced_dict, comments


# ============================================================
# 1. 数据模型：继承 ScriptEnvironmentResponse
# ============================================================
class MultiAgentScriptEnvironmentResponse(ScriptEnvironmentResponse):
    """扩展 ``ScriptEnvironmentResponse``，支持 3–4 名参与者的 LLM 评分。

    通过 **继承** 保留原始字段 ``p1_rate`` / ``p2_rate`` / ``terminated`` / ``comments``，
    仅新增 ``p3_rate`` / ``p4_rate``（默认 ``None``，向后兼容）。
    """

    p3_rate: float | tuple[float, dict[str, float]] | None = Field(
        default=None,
        description="rating of participant 3, on the scale of 1 to 10",
    )
    p4_rate: float | tuple[float, dict[str, float]] | None = Field(
        default=None,
        description="rating of participant 4, on the scale of 1 to 10",
    )


# ============================================================
# 2. 多智能体聚合函数
# ============================================================
# agent_N → p{N}_rate 映射（最多 4 人，与设计文档 2..4 一致）
_AGENT_RATE_FIELD_MAP: dict[str, str] = {
    "agent_1": "p1_rate",
    "agent_2": "p2_rate",
    "agent_3": "p3_rate",
    "agent_4": "p4_rate",
}


def _build_p_rate(
    agent_responses: dict[str, tuple[dict[str, float | int | bool], str]],
    agent_key: str,
) -> tuple[float, dict[str, float]] | None:
    entry = agent_responses.get(agent_key)
    if entry is None or entry == ({}, ""):
        return None
    scores, _reasoning = entry
    if "overall_score" not in scores:
        return None
    dims = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
    return (float(scores["overall_score"]), dims)


@validate_call
def unweighted_aggregate_evaluate_multi(
    responses: list[tuple[str, tuple[tuple[str, int | float | bool], str]]],
) -> MultiAgentScriptEnvironmentResponse:
    """与 ``unweighted_aggregate_evaluate`` 功能等价，但 **动态填充全部 N 名智能体的评分**。

    原始函数仅写入 ``p1_rate`` / ``p2_rate``（为兼容 ``ScriptEnvironmentResponse`` 的两字段结构）。
    本函数通过 ``MultiAgentScriptEnvironmentResponse`` 的额外字段 ``p3_rate`` / ``p4_rate``
    携带第 3、4 名智能体的评分。
    """
    responses_dict: dict[str, list[tuple[tuple[str, int | float | bool], str]]] = (
        defaultdict(list)
    )
    for response in responses:
        assert response[0] == "environment" or response[0].startswith("agent")
        responses_dict[response[0]].append(response[1])

    environment_responses: tuple[dict[str, float | int | bool], str] = ({}, "")
    agent_responses: dict[str, tuple[dict[str, float | int | bool], str]] = {}

    for k, v in responses_dict.items():
        if k == "environment":
            environment_responses = _reduce(v)
        else:
            agent_responses[k] = _reduce(v)

    # 构建所有智能体的 comments
    agent_comments = ""
    for agent_key, (_, comment) in sorted(agent_responses.items()):
        if comment:
            agent_name = agent_key.replace("_", " ").title()
            agent_comments += f"{agent_name} comments:\n{comment}\n"

    comments = (
        f"Environment comments: {environment_responses[1]}\n"
        if environment_responses[1]
        else ""
    ) + agent_comments

    terminated: bool = (
        environment_responses[0].get("terminated", False)
        if isinstance(environment_responses[0], dict)
        else False
    )

    # 动态填充所有 p{N}_rate：遍历已知的 agent_N → p{N}_rate 映射
    rate_kwargs: dict[str, Any] = {}
    for agent_key, field_name in _AGENT_RATE_FIELD_MAP.items():
        rate_kwargs[field_name] = _build_p_rate(agent_responses, agent_key)

    return MultiAgentScriptEnvironmentResponse(
        terminated=terminated,
        comments=comments,
        **rate_kwargs,
    )


# ============================================================
# 3. 扩展 build_eval_record — 动态提取全部 agent 的 LLM 维度分
# ============================================================
def build_eval_record_multi(
    *,
    experiment_tag: str,
    seq: int,
    agent_model: str,
    evaluator_model: str,
    quartet: bool,
    num_participants: int,
    result: LongTermNegotiationEvalResult,
    environment_profile_pk: str | None = None,
    scenario_codename: str | None = None,
    negotiation_run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """与 ``build_eval_record`` 功能等价，但 **动态提取全部 p{N}_rate**。

    原始函数硬编码只遍历 ``(1, "p1_rate"), (2, "p2_rate")``。
    本函数先调用原始 ``build_eval_record``，再检测 ``llm_aggregate`` 中额外
    的 ``p3_rate`` / ``p4_rate`` 并补入 ``llm_dimension_scores``。
    """
    row = build_eval_record(
        experiment_tag=experiment_tag,
        seq=seq,
        agent_model=agent_model,
        evaluator_model=evaluator_model,
        quartet=quartet,
        num_participants=num_participants,
        result=result,
        environment_profile_pk=environment_profile_pk,
        scenario_codename=scenario_codename,
        negotiation_run_config=negotiation_run_config,
    )

    # 检测 llm_aggregate 中是否有额外的 p{N}_rate（p3/p4），补入 llm_dimension_scores
    llm_dump = row.get("llm_aggregate")
    if isinstance(llm_dump, dict):
        llm_dim_scores: dict[str, dict[str, Any]] = row.setdefault("llm_dimension_scores", {})
        for idx, rate_key in ((3, "p3_rate"), (4, "p4_rate")):
            rate_val = llm_dump.get(rate_key)
            if (
                isinstance(rate_val, (tuple, list))
                and len(rate_val) >= 2
                and isinstance(rate_val[1], dict)
            ):
                dims = dict(rate_val[1])
                dims.setdefault("overall_score", rate_val[0])
                llm_dim_scores[f"agent{idx}"] = dims

    return row


# ============================================================
# 4. 补丁工具：临时替换聚合函数
# ============================================================
@contextlib.contextmanager
def _patch_aggregator():
    """在 ``llm_evaluation`` 模块中临时将 ``unweighted_aggregate_evaluate`` 替换为多智能体版本。

    不影响原始文件；退出上下文后自动恢复。
    """
    from . import llm_evaluation as _llm_eval_mod

    _original = _llm_eval_mod.unweighted_aggregate_evaluate
    _llm_eval_mod.unweighted_aggregate_evaluate = unweighted_aggregate_evaluate_multi  # type: ignore[assignment]
    try:
        yield
    finally:
        _llm_eval_mod.unweighted_aggregate_evaluate = _original  # type: ignore[assignment]


# ============================================================
# 5. 扩展 episode 评测
# ============================================================
async def run_llm_negotiation_episode_evaluation_multi(
    model_dict: dict[str, str],
    **kwargs: Any,
) -> LongTermNegotiationEvalResult:
    """与 ``run_llm_negotiation_episode_evaluation`` 功能等价，但终局 LLM 评分使用
    ``unweighted_aggregate_evaluate_multi``，使返回的 ``llm_aggregate`` 携带全部 N 名智能体的评分
    （``MultiAgentScriptEnvironmentResponse`` 实例）。

    通过 ``_patch_aggregator`` 上下文管理器在调用期间临时替换聚合函数，不需要修改原始文件。
    """
    from .llm_evaluation import run_llm_negotiation_episode_evaluation

    with _patch_aggregator():
        return await run_llm_negotiation_episode_evaluation(model_dict, **kwargs)


def _post_process_rows_for_multi_agent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """后处理：对已完成的 rows 补充 agent3 / agent4 的 llm_dimension_scores。

    原始 ``build_eval_record`` 只提取 p1_rate / p2_rate；本函数从 ``llm_aggregate``
    中检测 p3_rate / p4_rate 并补入 ``llm_dimension_scores``。
    """
    for row in rows:
        llm_dump = row.get("llm_aggregate")
        if not isinstance(llm_dump, dict):
            continue
        llm_dim_scores: dict[str, dict[str, Any]] = row.setdefault("llm_dimension_scores", {})
        for idx, rate_key in ((3, "p3_rate"), (4, "p4_rate")):
            rate_val = llm_dump.get(rate_key)
            if (
                isinstance(rate_val, (tuple, list))
                and len(rate_val) >= 2
                and isinstance(rate_val[1], dict)
            ):
                dims = dict(rate_val[1])
                dims.setdefault("overall_score", rate_val[0])
                llm_dim_scores[f"agent{idx}"] = dims
    return rows


# ============================================================
# 6. 扩展批量调度 — 通过补丁 + 后处理，完全复用原始逻辑
# ============================================================
async def run_long_term_negotiation_eval_batch_async_multi(
    *,
    agent_models: list[str],
    evaluator_model: str,
    quartet: bool = False,
    repeats_per_model: int = 1,
    batch_size: int = 3,
    params: Any = None,
    scenario_environment_pks: list[str] | None = None,
    max_macro_steps: int = 4000,
    run_terminal_llm_eval: bool = True,
    experiment_tag_base: str = "negotiation_eval_batch",
    run_id: str | None = None,
    history_max_action_log: int | None = 500,
    num_participants: int | None = None,
    model_trace_dir: str | None = None,
    execution_trace_dir: str | None = None,
    nest_trace_dirs_by_model_time: bool = False,
    run_timestamp: str | None = None,
    negotiation_run_config: dict[str, Any] | None = None,
    write_execution_record: bool = False,
) -> list[dict[str, Any]]:
    """与 ``run_long_term_negotiation_eval_batch_async`` 功能等价，但每条 episode 的终局 LLM
    评分通过 ``unweighted_aggregate_evaluate_multi`` 携带全部 N 名智能体评分，并后处理
    补全 ``llm_dimension_scores``。

    下游 ``aggregate_negotiation_eval_run_means`` 与 ``_print_evaluation_summary``
    已原生按动态 agent key 工作，无需额外修改。
    """
    from .batch_evaluation import run_long_term_negotiation_eval_batch_async

    with _patch_aggregator():
        rows = await run_long_term_negotiation_eval_batch_async(
            agent_models=agent_models,
            evaluator_model=evaluator_model,
            quartet=quartet,
            repeats_per_model=repeats_per_model,
            batch_size=batch_size,
            params=params,
            scenario_environment_pks=scenario_environment_pks,
            max_macro_steps=max_macro_steps,
            run_terminal_llm_eval=run_terminal_llm_eval,
            experiment_tag_base=experiment_tag_base,
            run_id=run_id,
            history_max_action_log=history_max_action_log,
            num_participants=num_participants,
            model_trace_dir=model_trace_dir,
            execution_trace_dir=execution_trace_dir,
            nest_trace_dirs_by_model_time=nest_trace_dirs_by_model_time,
            run_timestamp=run_timestamp,
            negotiation_run_config=negotiation_run_config,
            write_execution_record=write_execution_record,
        )

    return _post_process_rows_for_multi_agent(rows)


def run_long_term_negotiation_eval_batch_multi(
    *,
    agent_models: list[str],
    evaluator_model: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """同步入口：内部 ``asyncio.run``。"""
    import asyncio

    return asyncio.run(
        run_long_term_negotiation_eval_batch_async_multi(
            agent_models=agent_models,
            evaluator_model=evaluator_model,
            **kwargs,
        )
    )


# ============================================================
# 7. 按模型类别分组统计
# ============================================================
def _model_key_for_display(agent_model: str, max_len: int = 55) -> str:
    """从完整模型名中提取用于展示的短标签。"""
    s = str(agent_model or "?")
    # 去 custom/ 前缀和 @base_url 后缀
    if s.startswith("custom/"):
        s = s[len("custom/"):]
    if "@" in s:
        s = s.split("@")[0]
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


def aggregate_negotiation_eval_run_means_by_model(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """在 ``aggregate_negotiation_eval_run_means`` 基础上增加 **按 agent_model 分组** 的统计。

    返回结构::

        {
            "overall": { ... },          # 全量聚合（与原函数一致）
            "by_model": {
                "<model_label>": {       # 每个模型的独立聚合
                    "model_display": str,
                    "n_episodes": int,
                    "terminal_success_rate": float,
                    "rule_metrics_mean": {...},
                    "llm_dimension_scores_mean": {...},
                    "llm_overall_mean": {...},
                },
                ...
            },
        }

    下游 ``_print_evaluation_summary`` 仍可直接消费 ``overall`` 子 dict（向后兼容）。
    """
    from collections import defaultdict

    overall = {
        "n_episodes": len(rows),
        "terminal_success_rate": (
            sum(1 for r in rows if str(r.get("terminal") or "") == "success") / len(rows)
            if rows else 0.0
        ),
        "rule_metrics_mean": _mean_numeric_fields_across_rows(rows, field="rule_metrics"),
    }
    dim_means = _mean_llm_dimension_scores(rows)
    if dim_means:
        overall["llm_dimension_scores_mean"] = dim_means
    ov = _mean_llm_overall_from_aggregate(rows)
    if ov:
        overall["llm_overall_mean"] = ov

    # 按 agent_model 分组
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("agent_model") or "?")].append(r)

    by_model: dict[str, Any] = {}
    for model_name in sorted(groups):
        subset = groups[model_name]
        n = len(subset)
        succ = sum(1 for r in subset if str(r.get("terminal") or "") == "success")
        entry: dict[str, Any] = {
            "model_display": _model_key_for_display(model_name),
            "model_full": model_name,
            "n_episodes": n,
            "terminal_success_rate": (succ / n) if n else 0.0,
            "rule_metrics_mean": _mean_numeric_fields_across_rows(subset, field="rule_metrics"),
        }
        sub_dim = _mean_llm_dimension_scores(subset)
        if sub_dim:
            entry["llm_dimension_scores_mean"] = sub_dim
        sub_ov = _mean_llm_overall_from_aggregate(subset)
        if sub_ov:
            entry["llm_overall_mean"] = sub_ov
        by_model[model_name] = entry

    return {"overall": overall, "by_model": by_model}


# ============================================================
# 8. 按模型类别打印汇总
# ============================================================
def print_evaluation_summary_multi(
    rows: list[dict[str, Any]],
    aggregate_means: dict[str, Any],
) -> None:
    """与 ``_print_evaluation_summary`` 功能等价，但 **额外按 agent_model 分段展示**。

    当存在多个模型时，先打印整体汇总，再逐模型打印各自的统计。
    """
    import typer

    overall = aggregate_means.get("overall", aggregate_means)
    by_model = aggregate_means.get("by_model", {})

    # ── 先复用原有的整体汇总格式 ──
    _print_via_overall(rows, overall)

    # ── 按模型分段 ──
    if not by_model or len(by_model) <= 1:
        return

    typer.echo("")
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_MAGENTA))
    typer.echo(typer.style("  PER-MODEL BREAKDOWN", fg=typer.colors.BRIGHT_MAGENTA, bold=True))
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_MAGENTA))

    for model_name, stats in by_model.items():
        display = stats.get("model_display", model_name)
        n = stats.get("n_episodes", 0)

        typer.echo("")
        typer.echo(typer.style(f"── {display} ──", fg=typer.colors.YELLOW, bold=True))
        typer.echo(f"  Episodes:           {n}")
        s_rate = stats.get("terminal_success_rate", 0)
        typer.echo(f"  Success Rate:       {_fmt(s_rate * 100 if n else 0, 1)}%")

        rm = stats.get("rule_metrics_mean") or {}
        fs = rm.get("negotiation_final_state_score")
        if fs is not None:
            typer.echo(f"  Final State Score:  {_fmt(fs)}")

        llm_overall = stats.get("llm_overall_mean")
        llm_dims = stats.get("llm_dimension_scores_mean")
        if llm_overall and isinstance(llm_overall, dict):
            typer.echo("  [Overall LLM]")
            for k, v in llm_overall.items():
                typer.echo(f"    {k:<26s} {_fmt(v)}")
        if llm_dims and isinstance(llm_dims, dict):
            for agent_key, dims in llm_dims.items():
                if isinstance(dims, dict):
                    typer.echo(f"  [{agent_key}]")
                    for dk, dv in dims.items():
                        typer.echo(f"    {dk:<26s} {_fmt(dv)}")

    typer.echo("")
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_MAGENTA))


def _print_via_overall(rows: list[dict[str, Any]], overall: dict[str, Any]) -> None:
    """使用与 ``_print_evaluation_summary`` 相同的格式打印整体汇总。"""
    import typer

    n = overall.get("n_episodes", len(rows))
    successes = sum(1 for r in rows if str(r.get("terminal") or "") == "success")
    timeouts = sum(1 for r in rows if str(r.get("terminal") or "") == "timeout")
    failures = sum(1 for r in rows if str(r.get("terminal") or "") == "failure")
    others = n - successes - timeouts - failures
    rm = overall.get("rule_metrics_mean") or {}

    typer.echo("")
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_BLACK))
    typer.echo(typer.style("  EVALUATION RESULTS", fg=typer.colors.BRIGHT_CYAN, bold=True))
    typer.echo(typer.style("=" * 64, fg=typer.colors.BRIGHT_BLACK))

    typer.echo(typer.style("\n── Terminal Status ──", fg=typer.colors.YELLOW, bold=True))
    typer.echo(f"  Episodes:           {n}")
    typer.echo(f"  Success:            {successes} ({_fmt(100 * successes / n if n else 0, 1)}%)")
    typer.echo(f"  Timeout:            {timeouts} ({_fmt(100 * timeouts / n if n else 0, 1)}%)")
    typer.echo(f"  Failure:            {failures} ({_fmt(100 * failures / n if n else 0, 1)}%)")
    if others:
        typer.echo(f"  Other:              {others}")

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

    llm_overall = overall.get("llm_overall_mean")
    llm_dims = overall.get("llm_dimension_scores_mean")
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


def _fmt(v: Any, precision: int = 4) -> str:
    """格式化数字为右对齐字符串。"""
    if isinstance(v, float):
        return f"{v:{precision}.{precision}f}"
    return str(v)


# ============================================================
# 9. JSONL 轨迹 → 缩进 JSON（便于人工阅读）
# ============================================================
def convert_jsonl_traces_to_indented_json(
    trace_dir: str,
    *,
    suffix: str = ".indented.json",
    delete_original: bool = False,
) -> list[str]:
    """将目录下所有 ``*.jsonl`` 文件转换为带缩进的 JSON 文件。

    每个 JSONL 文件被解析为 JSON 数组并写入 ``{stem}{suffix}`` 文件。
    若 ``delete_original=True`` 则在成功转换后删除原始 JSONL。

    返回已转换的文件路径列表。
    """
    import json
    from pathlib import Path

    base = Path(trace_dir)
    if not base.is_dir():
        raise ValueError(f"not a directory: {trace_dir}")

    converted: list[str] = []
    for jsonl_path in sorted(base.rglob("*.jsonl")):
        records: list[Any] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        except Exception as exc:
            print(f"[WARN] Failed to parse {jsonl_path}: {exc}")
            continue

        if not records:
            continue

        out_path = jsonl_path.with_suffix(suffix)
        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        converted.append(str(out_path))

        if delete_original:
            jsonl_path.unlink()

    return converted


def indent_jsonl_in_place(trace_dir: str) -> list[str]:
    """将目录下所有 ``*.jsonl`` 文件 **原地** 改写为缩进格式。

    注意：改写后的文件不再是严格的 JSONL（每行一条 JSON），
    而是每条记录以 ``indent=2`` 的 JSON 格式保存，记录之间仍以换行分隔。
    适合不需要流式解析的场景。
    """
    import json
    from pathlib import Path

    base = Path(trace_dir)
    if not base.is_dir():
        raise ValueError(f"not a directory: {trace_dir}")

    converted: list[str] = []
    for jsonl_path in sorted(base.rglob("*.jsonl")):
        records: list[Any] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        except Exception as exc:
            print(f"[WARN] Failed to parse {jsonl_path}: {exc}")
            continue

        if not records:
            continue

        # 每条记录独立缩进，之间空行分隔
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(records):
                if i > 0:
                    f.write("\n")
                f.write(json.dumps(rec, ensure_ascii=False, indent=2, default=str))
                f.write("\n")
        converted.append(str(jsonl_path))

    return converted


# ============================================================
# 10. 一键式：运行 + 按模型统计 + 保存 + 打印
# ============================================================
async def run_multi_and_save(
    *,
    agent_models: list[str],
    evaluator_model: str,
    output: str | None = None,
    tag: str = "ltr_multi_firm",
    execution_trace_dir: str | None = None,
    indent_traces: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """端到端：运行多智能体批量评测，按模型分组统计，保存结果并打印汇总。

    参数
    ----
    output : str | None
        输出 JSON 文件路径（目录或文件名）。若为 None 则只打印不保存。
    indent_traces : bool
        是否在运行结束后将 execution_trace_dir 下的 JSONL 转为缩进格式。
    """
    import json
    import time as _time
    from pathlib import Path

    started_at = _time.strftime("%Y-%m-%d %H:%M:%S")
    ts = _time.strftime("%Y%m%d_%H%M%S")

    rows = await run_long_term_negotiation_eval_batch_async_multi(
        agent_models=agent_models,
        evaluator_model=evaluator_model,
        execution_trace_dir=execution_trace_dir,
        run_timestamp=ts,
        **kwargs,
    )

    aggregate_means = aggregate_negotiation_eval_run_means_by_model(rows)

    # 保存
    if output is not None:
        out_base = Path(output)
        if out_base.exists() and out_base.is_dir():
            out_dir = out_base
            out_file = out_dir / f"negotiation_eval_{tag}_{ts}.json"
        elif out_base.suffix == "":
            out_dir = out_base
            out_file = out_dir / f"negotiation_eval_{tag}_{ts}.json"
        else:
            out_dir = out_base.parent
            out_file = out_dir / f"{out_base.stem}_{ts}.json"
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_started_at": started_at,
            "run_timestamp": ts,
            "tag": tag,
            "agent_models": agent_models,
            "evaluator_model": evaluator_model,
            "aggregate_means": aggregate_means,
            "rows": rows,
        }
        out_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        import typer
        typer.echo(typer.style(f"Saved {len(rows)} records to {out_file}", fg=typer.colors.GREEN))

    # 缩进 JSONL 轨迹
    if indent_traces and execution_trace_dir:
        converted = indent_jsonl_in_place(execution_trace_dir)
        if converted:
            import typer
            typer.echo(
                typer.style(
                    f"Indented {len(converted)} JSONL trace files in {execution_trace_dir}",
                    fg=typer.colors.GREEN,
                )
            )

    # 打印
    print_evaluation_summary_multi(rows, aggregate_means)

    return {"rows": rows, "aggregate_means": aggregate_means}


def run_multi_and_save_sync(
    *,
    agent_models: list[str],
    evaluator_model: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """同步入口。"""
    import asyncio
    return asyncio.run(
        run_multi_and_save(
            agent_models=agent_models,
            evaluator_model=evaluator_model,
            **kwargs,
        )
    )


# ============================================================
# 8. V2 扩展批量评测 — 使用 ExtendedLongTermNegotiationEnv + V2 指标
# ============================================================
@contextlib.contextmanager
def _patch_for_extended_eval():
    """临时替换 batch_evaluation 中的 episode 评测入口为 V2 扩展版。"""
    from . import batch_evaluation as _be

    _original_eval = _be.run_llm_negotiation_episode_evaluation
    from .extended_llm_evaluation import run_extended_llm_negotiation_episode_evaluation

    _be.run_llm_negotiation_episode_evaluation = run_extended_llm_negotiation_episode_evaluation  # type: ignore[assignment]
    try:
        yield
    finally:
        _be.run_llm_negotiation_episode_evaluation = _original_eval  # type: ignore[assignment]


async def run_long_term_negotiation_eval_batch_async_multi_extended(
    *,
    agent_models: list[str],
    evaluator_model: str,
    quartet: bool = False,
    repeats_per_model: int = 1,
    batch_size: int = 3,
    params: Any = None,
    scenario_environment_pks: list[str] | None = None,
    max_macro_steps: int = 4000,
    run_terminal_llm_eval: bool = True,
    experiment_tag_base: str = "negotiation_eval_batch",
    run_id: str | None = None,
    history_max_action_log: int | None = 500,
    num_participants: int | None = None,
    model_trace_dir: str | None = None,
    execution_trace_dir: str | None = None,
    nest_trace_dirs_by_model_time: bool = False,
    run_timestamp: str | None = None,
    negotiation_run_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """V2 扩展版批量评测：使用 ExtendedLongTermNegotiationEnv + V2 deterministic payouts。

    与 ``run_long_term_negotiation_eval_batch_async_multi`` 功能等价，
    但每条 episode 走 V2 合同结算（cooperation: predetermined_payouts; buy_sell: price_difference）。
    """
    with _patch_aggregator(), _patch_for_extended_eval():
        rows = await run_long_term_negotiation_eval_batch_async_multi(
            agent_models=agent_models,
            evaluator_model=evaluator_model,
            quartet=quartet,
            repeats_per_model=repeats_per_model,
            batch_size=batch_size,
            params=params,
            scenario_environment_pks=scenario_environment_pks,
            max_macro_steps=max_macro_steps,
            run_terminal_llm_eval=run_terminal_llm_eval,
            experiment_tag_base=experiment_tag_base,
            run_id=run_id,
            history_max_action_log=history_max_action_log,
            num_participants=num_participants,
            model_trace_dir=model_trace_dir,
            execution_trace_dir=execution_trace_dir,
            nest_trace_dirs_by_model_time=nest_trace_dirs_by_model_time,
            run_timestamp=run_timestamp,
            negotiation_run_config=negotiation_run_config,
        )
    return rows


def run_long_term_negotiation_eval_batch_multi_extended(
    *,
    agent_models: list[str],
    evaluator_model: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """同步入口。"""
    import asyncio

    return asyncio.run(
        run_long_term_negotiation_eval_batch_async_multi_extended(
            agent_models=agent_models,
            evaluator_model=evaluator_model,
            **kwargs,
        )
    )


__all__ = [
    "MultiAgentScriptEnvironmentResponse",
    "aggregate_negotiation_eval_run_means_by_model",
    "build_eval_record_multi",
    "convert_jsonl_traces_to_indented_json",
    "indent_jsonl_in_place",
    "print_evaluation_summary_multi",
    "run_llm_negotiation_episode_evaluation_multi",
    "run_long_term_negotiation_eval_batch_async_multi",
    "run_long_term_negotiation_eval_batch_multi",
    "run_long_term_negotiation_eval_batch_async_multi_extended",
    "run_long_term_negotiation_eval_batch_multi_extended",
    "run_multi_and_save",
    "run_multi_and_save_sync",
    "unweighted_aggregate_evaluate_multi",
]
