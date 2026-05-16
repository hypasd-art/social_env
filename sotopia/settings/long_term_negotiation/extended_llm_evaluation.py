"""合同经济学确定性化 V2 — 扩展运行时入口。

扩展 ``run_llm_negotiation_episode_evaluation``：
- 支持从 game_metadata.initial_resources_by_role 加载场景专属初始资金
- 使用 ExtendedLongTermNegotiationEnv
- 使用 V2 指标计算
"""

from __future__ import annotations

import asyncio
from contextvars import Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sotopia.database import EnvironmentProfile, EnvAgentComboStorage, AgentProfile
from sotopia.database.base_models import LLMEvalBaseModel
from sotopia.messages import ScriptEnvironmentResponse
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    unweighted_aggregate_evaluate,
)

from .extended_env import ExtendedLongTermNegotiationEnv
from .extended_negotiation_metrics import (
    _is_v2_rule,
    compute_v2_rule_payout_metrics,
)
from .llm_evaluation import (
    LongTermNegotiationEvalResult,
    _agent_profile_v2_for_agent,
    _build_role_addons_from_env_binding,
    _initial_resources_for_roster_from_env,
    build_llm_negotiation_agents,
    default_negotiation_roster,
    format_negotiation_episode_for_llm_eval,
)
from .negotiation_metrics import (
    build_rule_evaluation_state_record,
    compute_negotiation_rule_metrics,
)
from .roles import default_agent_resources_bundle
from .scenario_loader import (
    DIALOGUE_STYLE_EVAL_RUBRIC_EN,
    goal_addon_for_deal_closure_pressure,
    load_negotiation_scenario_from_environment_profile_pk,
)
from .types import (
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    NegotiationTimelineParams,
    SUPPORTED_NEGOTIATION_LINEUPS,
)


def _resolve_initial_resources(
    *,
    environment_profile_pk: str | None,
    roster: tuple[str, ...],
    game_metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]] | None:
    """按优先级解析初始资金。

    1. game_metadata["initial_resources_by_role"] — 场景合成时指定（最高优先级）
    2. AgentProfileV2.initial_resources — 数据库已有字段
    3. default_agent_resources_bundle() — 硬编码兜底（返回 None，由 Env 默认处理）
    """
    gm = game_metadata or {}
    ir_from_gm = gm.get("initial_resources_by_role")
    if isinstance(ir_from_gm, dict):
        # Validate and convert
        result: dict[str, dict[str, float]] = {}
        for role in roster:
            role_res = ir_from_gm.get(role)
            if isinstance(role_res, dict):
                result[role] = {
                    str(k): float(v)
                    for k, v in role_res.items()
                    if isinstance(v, (int, float))
                }
        if result:
            return result

    # Fall back to AgentProfileV2 initialization
    return _initial_resources_for_roster_from_env(environment_profile_pk, roster)


def _compute_v2_rule_metrics(
    env: Any,
    *,
    predefined_outcome_rule: dict[str, Any] | None = None,
) -> dict[str, float]:
    """V2 兼容的 rule metrics 计算。"""
    ctrl = env.ctrl
    st = env.system_state
    term = getattr(ctrl, "terminal", None) or ""
    out: dict[str, float] = {}
    out["negotiation_terminal_is_success"] = 1.0 if term == "success" else 0.0
    out["negotiation_terminal_is_timeout"] = 1.0 if term == "timeout" else 0.0
    out["negotiation_terminal_is_failure"] = 1.0 if term == "failure" else 0.0
    out["negotiation_terminal_is_max_steps_cap"] = (
        1.0 if term == "max_steps" or term == "" else 0.0
    )
    macro = float(getattr(env, "last_episode_macro_steps", 0) or 0)
    out["negotiation_macro_steps_used"] = macro
    out["negotiation_n_session_log"] = float(len(getattr(ctrl, "session_log", []) or []))
    out["negotiation_n_action_log"] = float(len(getattr(ctrl, "action_log", []) or []))
    out["negotiation_n_message_log"] = float(len(getattr(ctrl, "message_log", []) or []))
    vh = getattr(ctrl, "visible_history", {}) or {}
    out["negotiation_visible_history_total_lines"] = float(
        sum(len(v) for v in vh.values())
    )

    pcs = getattr(ctrl, "primary_contract_id", None)
    if pcs:
        c = getattr(ctrl, "contracts", {}).get(pcs)
        if c is not None:
            stmap = {
                "proposed": 1.0,
                "amended": 2.0,
                "accepted": 3.0,
                "signed": 4.0,
                "rejected": -1.0,
            }
            raw = getattr(c, "status", "") or ""
            out["negotiation_primary_contract_phase"] = float(stmap.get(str(raw), 0.0))

    cash_list = [
        float(st.agent_resources.get(a, {}).get("cash", 0.0)) for a in st.agent_keys
    ]
    if cash_list:
        out["negotiation_participant_mean_cash"] = float(sum(cash_list) / len(cash_list))
        out["negotiation_participant_min_cash"] = float(min(cash_list))

    # Use V2-aware final state metrics
    from .negotiation_metrics import (
        _clip,
        _final_intermediate_snapshot,
        _scene_score_weights,
        _scheduling_effectiveness_factor,
        primary_contract_status_factor,
    )

    ctrl2 = env.ctrl
    fs_out: dict[str, float] = {}
    snap = _final_intermediate_snapshot(ctrl2)
    fs_out["negotiation_final_state_n_snapshots"] = float(
        len(getattr(ctrl2, "state_snapshots", []) or [])
    )
    if snap is None:
        fs_out["negotiation_final_state_score"] = 0.0
        out.update(fs_out)
        return out

    fs_out["negotiation_final_state_day_closed"] = float(
        int(snap.get("day_closed") or snap.get("day") or 0)
    )

    initial_resources = default_agent_resources_bundle()
    final_resources_raw = snap.get("agent_resources") or {}
    final_resources: dict[str, dict[str, float]] = {
        str(role): {
            k: float(v) for k, v in (vals or {}).items() if isinstance(v, (int, float))
        }
        for role, vals in final_resources_raw.items()
    }

    cash_final: list[float] = []
    cash_initial: list[float] = []
    for role, vals in final_resources.items():
        c_final = float(vals.get("cash", 0.0))
        c_init = float(initial_resources.get(role, {}).get("cash", 0.0))
        cash_final.append(c_final)
        cash_initial.append(c_init)

    if cash_final:
        fs_out["negotiation_final_state_total_cash"] = float(sum(cash_final))
        fs_out["negotiation_final_state_total_cash_delta"] = (
            float(sum(cash_final)) - float(sum(cash_initial))
        )
        fs_out["negotiation_final_state_min_cash"] = float(min(cash_final))
        fs_out["negotiation_final_state_n_solvent"] = float(
            sum(1 for c in cash_final if c > 0.0)
        )
        fs_out["negotiation_final_state_solvency_ratio"] = (
            fs_out["negotiation_final_state_n_solvent"] / float(len(cash_final))
        )
    else:
        fs_out["negotiation_final_state_total_cash"] = 0.0
        fs_out["negotiation_final_state_total_cash_delta"] = 0.0
        fs_out["negotiation_final_state_min_cash"] = 0.0
        fs_out["negotiation_final_state_n_solvent"] = 0.0
        fs_out["negotiation_final_state_solvency_ratio"] = 0.0

    success_factor = 1.0 if term == "success" else 0.0

    primary_factor = 0.0
    pcs2 = getattr(ctrl2, "primary_contract_id", None)
    if pcs2:
        c2 = getattr(ctrl2, "contracts", {}).get(pcs2)
        if c2 is not None:
            status2 = str(getattr(c2, "status", "") or "").lower()
            primary_factor = primary_contract_status_factor(status2)

    solvency_factor = float(fs_out["negotiation_final_state_solvency_ratio"])
    liquidity_factor = (
        1.0 if fs_out["negotiation_final_state_total_cash_delta"] >= 0.0 else 0.0
    )
    scheduling_factor = _scheduling_effectiveness_factor(ctrl2)
    fs_out["negotiation_scheduling_effectiveness_factor"] = scheduling_factor
    weights = _scene_score_weights(predefined_outcome_rule)

    # V2-aware payout metrics
    rule_out, rule_factor = compute_v2_rule_payout_metrics(
        env=env,
        primary_factor=primary_factor,
        predefined_outcome_rule=predefined_outcome_rule,
    )
    fs_out.update(rule_out)

    score = (
        weights["terminal_success"] * success_factor
        + weights["primary_contract"] * primary_factor
        + weights["solvency"] * solvency_factor
        + weights["liquidity_preserved"] * liquidity_factor
        + weights["scheduling_effectiveness"] * scheduling_factor
        + weights["predefined_rule"] * rule_factor
    )
    fs_out["negotiation_final_state_score"] = float(max(0.0, min(1.0, score)))
    fs_out["negotiation_final_state_score_component_terminal_success"] = (
        weights["terminal_success"] * success_factor
    )
    fs_out["negotiation_final_state_score_component_primary_contract"] = (
        weights["primary_contract"] * primary_factor
    )
    fs_out["negotiation_final_state_score_component_solvency"] = (
        weights["solvency"] * solvency_factor
    )
    fs_out["negotiation_final_state_score_component_liquidity_preserved"] = (
        weights["liquidity_preserved"] * liquidity_factor
    )
    fs_out["negotiation_final_state_score_component_predefined_rule"] = (
        weights["predefined_rule"] * rule_factor
    )
    fs_out["negotiation_final_state_score_component_scheduling_effectiveness"] = (
        weights["scheduling_effectiveness"] * scheduling_factor
    )
    out.update(fs_out)
    return out


async def run_extended_llm_negotiation_episode_evaluation(
    model_dict: dict[str, str],
    *,
    quartet: bool = False,
    num_participants: int | None = None,
    lineup: str | None = None,
    params: NegotiationTimelineParams | None = None,
    environment_profile_pk: str | None = None,
    max_macro_steps: int = 4000,
    run_terminal_llm_eval: bool = True,
    evaluation_dimension_model: type[LLMEvalBaseModel] = None,  # type: ignore[assignment]
    history_max_action_log: int | None = 500,
    model_trace_dir: Path | str | None = None,
    model_trace_tag: str | None = None,
    execution_trace_dir: Path | str | None = None,
    execution_trace_tag: str | None = None,
    negotiation_run_config: dict[str, Any] | None = None,
    write_execution_record: bool = False,
) -> LongTermNegotiationEvalResult:
    """V2 扩展版 episode 评测入口。

    与原始 ``run_llm_negotiation_episode_evaluation`` 的差异：
    - 使用 ExtendedLongTermNegotiationEnv
    - 支持 game_metadata.initial_resources_by_role
    - 使用 V2 指标计算
    """
    from sotopia.database import SotopiaDimensions

    if evaluation_dimension_model is None:
        evaluation_dimension_model = SotopiaDimensions

    if "env" not in model_dict:
        raise KeyError("model_dict must contain key 'env' for the evaluator / scoring model.")

    trace_token: Token | None = None
    trace_stem: str | None = None
    _jsonl_dir = model_trace_dir if model_trace_dir is not None else execution_trace_dir
    _jsonl_tag = model_trace_tag or execution_trace_tag or "negotiation_episode"
    if _jsonl_dir is not None:
        from .model_trace import begin_episode_trace, safe_trace_filename

        trace_path = Path(_jsonl_dir).resolve() / safe_trace_filename(_jsonl_tag)
        trace_stem = trace_path.stem
        trace_token = begin_episode_trace(trace_path)

    n_from_scen: int | None = None
    lineup_from_scen: str | None = None
    predefined_rule: dict[str, Any] | None = None
    gm: dict[str, Any] = {}
    if environment_profile_pk:
        scen = load_negotiation_scenario_from_environment_profile_pk(environment_profile_pk)
        env_profile = EnvironmentProfile.get(environment_profile_pk)
        gm = env_profile.game_metadata if isinstance(env_profile.game_metadata, dict) else {}
        raw_rule = gm.get("predefined_outcome_rule")
        if isinstance(raw_rule, dict):
            predefined_rule = dict(raw_rule)
        strict_run = scen.strict_design_v1
        n_from_scen = scen.num_participants
        lineup_from_scen = scen.lineup
        params_run = scen.params if params is None else params
    else:
        strict_run = quartet
        params_run = params or NegotiationTimelineParams(
            D=8,
            s_max_per_day=2,
            max_session_rounds=12,
            max_total_turns_per_session=32,
        )

    if num_participants is not None:
        n = num_participants
    elif n_from_scen is not None:
        n = n_from_scen
    else:
        n = 4 if quartet else 2

    effective_lineup = lineup or lineup_from_scen or NEGOTIATION_LINEUP_WITH_INSTITUTIONAL
    if effective_lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"unknown negotiation lineup {effective_lineup!r}; expected one of "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}"
        )
    if n < 2 or n > 4:
        raise ValueError(f"effective num_participants must be 2..4, got {n}")

    try:
        roster = default_negotiation_roster(num_participants=n, lineup=effective_lineup)
        if len(roster) < 4 or effective_lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
            strict_run = False
        agents_map = build_llm_negotiation_agents(
            model_dict, roster, negotiation_run_config=negotiation_run_config
        )
        if environment_profile_pk:
            role_addons = _build_role_addons_from_env_binding(environment_profile_pk, roster)
            for role, addon in role_addons.items():
                ag = agents_map.get(role)
                if ag is None:
                    continue
                base_goal = str(getattr(ag, "goal", "") or "")
                extra = f"\n\n[Loaded profile+relationship context for this episode]\n{addon}"
                ag.goal = (base_goal + extra).strip() if base_goal else extra.strip()

            raw_closure = gm.get("deal_closure_pressure")
            if isinstance(raw_closure, dict) and int(raw_closure.get("version") or 0) == 1:
                for role, ag in agents_map.items():
                    closer = goal_addon_for_deal_closure_pressure(role, raw_closure)
                    if not closer:
                        continue
                    base_goal = str(getattr(ag, "goal", "") or "")
                    ag.goal = (base_goal + "\n\n" + closer).strip() if base_goal else closer.strip()

        # V2: resolve initial resources with game_metadata priority
        init_res = _resolve_initial_resources(
            environment_profile_pk=environment_profile_pk,
            roster=roster,
            game_metadata=gm,
        )

        # V2: use ExtendedLongTermNegotiationEnv
        env = ExtendedLongTermNegotiationEnv(
            agents_map,
            params=params_run,
            strict_design_v1=strict_run,
            predefined_outcome_rule=predefined_rule,
            initial_resources=init_res,
        )

        terminal = await env.run_episode_async(max_macro_steps=max_macro_steps)

        # V2: use V2-aware rule metrics
        rule_metrics = _compute_v2_rule_metrics(
            env, predefined_outcome_rule=predefined_rule
        )
        rule_eval_state = build_rule_evaluation_state_record(
            env, predefined_outcome_rule=predefined_rule
        )

        if write_execution_record and execution_trace_dir is not None:
            from .episode_execution_record import (
                safe_execution_trace_filename,
                write_episode_execution_record,
            )

            ex_path = Path(execution_trace_dir).resolve() / safe_execution_trace_filename(
                execution_trace_tag or "negotiation_episode"
            )
            write_episode_execution_record(
                env,
                ex_path,
                model_trace_dir=Path(_jsonl_dir).resolve() if trace_stem and _jsonl_dir else None,
                model_trace_stem=trace_stem,
            )

        llm_agg: ScriptEnvironmentResponse | None = None
        if run_terminal_llm_eval:
            ds_block: str | None = None
            raw_ds = gm.get("dialogue_style") if isinstance(gm.get("dialogue_style"), dict) else None
            if isinstance(raw_ds, dict):
                ev = raw_ds.get("evaluation_requirements_en")
                if isinstance(ev, str) and ev.strip():
                    ds_block = ev.strip()
            history = format_negotiation_episode_for_llm_eval(
                env,
                max_action_log=history_max_action_log,
                dialogue_eval_rubric_en=ds_block,
            )
            evaluator = EpisodeLLMEvaluator(
                model_name=model_dict["env"],
                response_format_class=EvaluationForAgents[evaluation_dimension_model],  # type: ignore[valid-type]
            )
            raw = await evaluator.__acall__(
                turn_number=-1,
                history=history,
                messages=None,
                num_agents_override=len(roster),
            )
            llm_agg = unweighted_aggregate_evaluate(list(raw))

        return LongTermNegotiationEvalResult(
            terminal=terminal,
            rule_metrics=rule_metrics,
            llm_aggregate=llm_agg,
            rule_evaluation_state=rule_eval_state,
        )
    finally:
        if trace_token is not None:
            from .model_trace import end_episode_trace

            end_episode_trace(trace_token)


def evaluate_extended_long_term_negotiation_llm_sync(
    model_dict: dict[str, str],
    **kwargs: Any,
) -> LongTermNegotiationEvalResult:
    """同步薄封装。"""
    return asyncio.run(
        run_extended_llm_negotiation_episode_evaluation(model_dict, **kwargs)
    )


__all__ = [
    "ExtendedLongTermNegotiationEnv",
    "run_extended_llm_negotiation_episode_evaluation",
    "evaluate_extended_long_term_negotiation_llm_sync",
    "_resolve_initial_resources",
    "_compute_v2_rule_metrics",
]
