"""长期谈判 **规则型** 可计算指标（与 ``envs/benchmark_evaluators`` 的键风格对齐）。

不依赖 ``ContractLedger`` / JsonModel；输入为 ``LongTermNegotiationEnv`` 或任意带
``ctrl`` + ``system_state`` 的对象，便于在 episode 结束后与 ``compute_individual_metrics``
等函数并列打印 / 落盘。

两组指标（合并写回 ``rule_metrics``）：

* ``compute_negotiation_rule_metrics``：terminal、动作日志计数、当前 ``system_state`` 现金等。
* ``compute_negotiation_final_state_metrics``：从 ``controller.state_snapshots`` 取 **最后一条
  intermediate snapshot**（end-of-day 写入），与 ``default_agent_resources_bundle`` 做 delta，
  并合成一个 ``negotiation_final_state_score ∈ [0, 1]`` 作为模型评价指标之一。
"""

from __future__ import annotations

from typing import Any

from .roles import default_agent_resources_bundle

# ``negotiation_final_state_score`` 的权重（合计 1.0，便于解释）。
_FINAL_STATE_WEIGHT_TERMINAL_SUCCESS = 0.3
_FINAL_STATE_WEIGHT_PRIMARY_CONTRACT = 0.2
_FINAL_STATE_WEIGHT_SOLVENCY = 0.15
_FINAL_STATE_WEIGHT_LIQUIDITY_PRESERVED = 0.1
_FINAL_STATE_WEIGHT_PREDEFINED_RULE = 0.25


def primary_contract_status_factor(status: str) -> float:
    st = str(status or "").lower()
    primary_map = {
        "proposed": 0.25,
        "amended": 0.5,
        "accepted": 0.75,
        "signed": 1.0,
        "rejected": 0.0,
        "failed": 0.0,
    }
    return float(primary_map.get(st, 0.0))


def _final_intermediate_snapshot(ctrl: Any) -> dict[str, Any] | None:
    snaps = list(getattr(ctrl, "state_snapshots", []) or [])
    if not snaps:
        return None
    return dict(snaps[-1])


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _compute_predefined_rule_payout_metrics(
    *,
    env: Any,
    primary_factor: float,
    predefined_outcome_rule: dict[str, Any] | None,
) -> tuple[dict[str, float], float]:
    out: dict[str, float] = {}
    if not isinstance(predefined_outcome_rule, dict):
        out["negotiation_predefined_rule_enabled"] = 0.0
        return out, 0.0

    out["negotiation_predefined_rule_enabled"] = 1.0
    contract_value = float(predefined_outcome_rule.get("contract_value_if_signed", 0.0) or 0.0)
    out["negotiation_predefined_rule_contract_value"] = contract_value
    out["negotiation_predefined_rule_news_signal"] = float(
        predefined_outcome_rule.get("news_signal", 0.0) or 0.0
    )

    mf = predefined_outcome_rule.get("margin_formula")
    mf_dict = mf if isinstance(mf, dict) else {}
    base_margin = float(mf_dict.get("base_margin", 0.05) or 0.05)
    news_weight = float(mf_dict.get("news_weight", 0.55) or 0.55)
    execution_weight = float(mf_dict.get("execution_weight", 0.45) or 0.45)

    bounds = predefined_outcome_rule.get("profit_margin_bounds")
    if isinstance(bounds, list | tuple) and len(bounds) >= 2:
        lo = float(bounds[0])
        hi = float(bounds[1])
    else:
        lo, hi = -0.25, 0.35
    if lo >= hi:
        lo, hi = -0.25, 0.35

    execution_signal = float(primary_factor * 2.0 - 1.0)
    raw_margin = base_margin + news_weight * out["negotiation_predefined_rule_news_signal"] + execution_weight * execution_signal
    realized_margin = _clip(raw_margin, lo, hi)
    out["negotiation_predefined_rule_realized_margin"] = realized_margin

    # 与 ``env._apply_contract_status_settlement_if_needed`` 一致：按主合同状态计「生效」，
    # 不依赖 ``terminal == success``；否则 timeout 但已签约时现金已结算而本指标恒为 0。
    is_contract_effective = 1.0 if primary_factor >= 0.75 else 0.0
    out["negotiation_predefined_rule_contract_effective"] = is_contract_effective
    total_profit = contract_value * realized_margin * is_contract_effective
    out["negotiation_predefined_rule_total_profit"] = total_profit

    share_raw = predefined_outcome_rule.get("company_profit_share")
    shares = share_raw if isinstance(share_raw, dict) else {}
    den = float(sum(float(v) for v in shares.values() if isinstance(v, (int, float))))
    if den <= 0.0:
        den = 1.0
    for role, val in shares.items():
        if not isinstance(role, str) or not isinstance(val, (int, float)):
            continue
        out[f"negotiation_predefined_rule_company_profit_{role}"] = total_profit * float(val) / den

    ind_raw = predefined_outcome_rule.get("individual_income_share")
    ind = ind_raw if isinstance(ind_raw, dict) else {}
    for role, val in ind.items():
        if not isinstance(role, str) or not isinstance(val, (int, float)):
            continue
        company_profit = out.get(f"negotiation_predefined_rule_company_profit_{role}", 0.0)
        out[f"negotiation_predefined_rule_individual_profit_{role}"] = company_profit * float(val)

    rule_factor = _clip((realized_margin - lo) / (hi - lo), 0.0, 1.0) * is_contract_effective
    out["negotiation_predefined_rule_score"] = rule_factor
    return out, rule_factor


def compute_predefined_rule_settlement_by_contract_status(
    *,
    env: Any,
    predefined_outcome_rule: dict[str, Any] | None,
    contract_status: str,
) -> dict[str, float]:
    """按合同状态计算应结算的个人资金变化（与数据构造规则同口径）。"""
    primary_factor = primary_contract_status_factor(contract_status)
    # 资金结算按“合同状态”驱动，不额外要求 terminal=success。
    out, _ = _compute_predefined_rule_payout_metrics(
        env=env,
        primary_factor=primary_factor,
        predefined_outcome_rule=predefined_outcome_rule,
    )
    return out


def compute_negotiation_final_state_metrics(
    env: Any,
    *,
    predefined_outcome_rule: dict[str, Any] | None = None,
) -> dict[str, float]:
    """从 ``ctrl.state_snapshots`` 的最后一条 intermediate state 计算评价指标。

    返回浮点字典，关键键：

    * ``negotiation_final_state_n_snapshots``：snapshot 总数。
    * ``negotiation_final_state_day_closed``：最后一次 end-of-day 关闭的自然日（无则 0）。
    * ``negotiation_final_state_total_cash`` / ``_total_cash_delta``：现金总量及与 t=0 的差。
    * ``negotiation_final_state_min_cash``：参与者最低现金。
    * ``negotiation_final_state_n_solvent`` / ``_solvency_ratio``：现金>0 参与者数 / 比例。
    * ``negotiation_final_state_score``：composite ∈ [0, 1]，权重见模块常量。

    没有 snapshot 时返回 ``{negotiation_final_state_n_snapshots: 0.0, negotiation_final_state_score: 0.0}``。
    """
    ctrl = env.ctrl
    out: dict[str, float] = {}

    snap = _final_intermediate_snapshot(ctrl)
    out["negotiation_final_state_n_snapshots"] = float(
        len(getattr(ctrl, "state_snapshots", []) or [])
    )
    if snap is None:
        out["negotiation_final_state_score"] = 0.0
        return out

    out["negotiation_final_state_day_closed"] = float(
        int(snap.get("day_closed") or snap.get("day") or 0)
    )

    initial_resources = default_agent_resources_bundle()
    final_resources_raw = snap.get("agent_resources") or {}
    final_resources: dict[str, dict[str, float]] = {
        str(role): {k: float(v) for k, v in (vals or {}).items() if isinstance(v, (int, float))}
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
        total_final = float(sum(cash_final))
        total_initial = float(sum(cash_initial))
        out["negotiation_final_state_total_cash"] = total_final
        out["negotiation_final_state_total_cash_delta"] = total_final - total_initial
        out["negotiation_final_state_min_cash"] = float(min(cash_final))
        out["negotiation_final_state_n_solvent"] = float(sum(1 for c in cash_final if c > 0.0))
        out["negotiation_final_state_solvency_ratio"] = (
            out["negotiation_final_state_n_solvent"] / float(len(cash_final))
        )
    else:
        out["negotiation_final_state_total_cash"] = 0.0
        out["negotiation_final_state_total_cash_delta"] = 0.0
        out["negotiation_final_state_min_cash"] = 0.0
        out["negotiation_final_state_n_solvent"] = 0.0
        out["negotiation_final_state_solvency_ratio"] = 0.0

    term = (getattr(ctrl, "terminal", None) or "").lower()
    success_factor = 1.0 if term == "success" else 0.0

    primary_factor = 0.0
    pcs = getattr(ctrl, "primary_contract_id", None)
    if pcs:
        c = getattr(ctrl, "contracts", {}).get(pcs)
        if c is not None:
            status = str(getattr(c, "status", "") or "").lower()
            primary_factor = primary_contract_status_factor(status)

    solvency_factor = float(out["negotiation_final_state_solvency_ratio"])
    liquidity_factor = (
        1.0 if out["negotiation_final_state_total_cash_delta"] >= 0.0 else 0.0
    )

    score = (
        _FINAL_STATE_WEIGHT_TERMINAL_SUCCESS * success_factor
        + _FINAL_STATE_WEIGHT_PRIMARY_CONTRACT * primary_factor
        + _FINAL_STATE_WEIGHT_SOLVENCY * solvency_factor
        + _FINAL_STATE_WEIGHT_LIQUIDITY_PRESERVED * liquidity_factor
    )
    rule_out, rule_factor = _compute_predefined_rule_payout_metrics(
        env=env,
        primary_factor=primary_factor,
        predefined_outcome_rule=predefined_outcome_rule,
    )
    out.update(rule_out)
    score += _FINAL_STATE_WEIGHT_PREDEFINED_RULE * rule_factor
    out["negotiation_final_state_score"] = float(max(0.0, min(1.0, score)))
    out["negotiation_final_state_score_component_terminal_success"] = (
        _FINAL_STATE_WEIGHT_TERMINAL_SUCCESS * success_factor
    )
    out["negotiation_final_state_score_component_primary_contract"] = (
        _FINAL_STATE_WEIGHT_PRIMARY_CONTRACT * primary_factor
    )
    out["negotiation_final_state_score_component_solvency"] = (
        _FINAL_STATE_WEIGHT_SOLVENCY * solvency_factor
    )
    out["negotiation_final_state_score_component_liquidity_preserved"] = (
        _FINAL_STATE_WEIGHT_LIQUIDITY_PRESERVED * liquidity_factor
    )
    out["negotiation_final_state_score_component_predefined_rule"] = (
        _FINAL_STATE_WEIGHT_PREDEFINED_RULE * rule_factor
    )
    return out


def compute_negotiation_rule_metrics(
    env: Any,
    *,
    predefined_outcome_rule: dict[str, Any] | None = None,
) -> dict[str, float]:
    """从 ``LongTermNegotiationEnv``（或鸭子类型）抽取浮点指标。

    Keys 前缀 ``negotiation_*`` ，与 ``benchmark_evaluators`` 的 ``individual_*`` 等区分开。
    自动合并 ``compute_negotiation_final_state_metrics`` 的结果，便于 JSONL / 日志同时落盘。
    """
    ctrl = env.ctrl
    st = env.system_state
    term = getattr(ctrl, "terminal", None) or ""
    out: dict[str, float] = {}
    out["negotiation_terminal_is_success"] = 1.0 if term == "success" else 0.0
    out["negotiation_terminal_is_timeout"] = 1.0 if term == "timeout" else 0.0
    out["negotiation_terminal_is_failure"] = 1.0 if term == "failure" else 0.0
    out["negotiation_terminal_is_max_steps_cap"] = 1.0 if term == "max_steps" or term == "" else 0.0
    macro = float(getattr(env, "last_episode_macro_steps", 0) or 0)
    out["negotiation_macro_steps_used"] = macro
    out["negotiation_n_session_log"] = float(len(getattr(ctrl, "session_log", []) or []))
    out["negotiation_n_action_log"] = float(len(getattr(ctrl, "action_log", []) or []))
    out["negotiation_n_message_log"] = float(len(getattr(ctrl, "message_log", []) or []))
    vh = getattr(ctrl, "visible_history", {}) or {}
    out["negotiation_visible_history_total_lines"] = float(sum(len(v) for v in vh.values()))

    pcs = getattr(ctrl, "primary_contract_id", None)
    if pcs:
        c = getattr(ctrl, "contracts", {}).get(pcs)
        if c is not None:
            stmap = {"proposed": 1.0, "amended": 2.0, "accepted": 3.0, "signed": 4.0, "rejected": -1.0}
            raw = getattr(c, "status", "") or ""
            out["negotiation_primary_contract_phase"] = float(stmap.get(str(raw), 0.0))

    cash_list = [float(st.agent_resources.get(a, {}).get("cash", 0.0)) for a in st.agent_keys]
    if cash_list:
        out["negotiation_participant_mean_cash"] = float(sum(cash_list) / len(cash_list))
        out["negotiation_participant_min_cash"] = float(min(cash_list))

    out.update(
        compute_negotiation_final_state_metrics(
            env,
            predefined_outcome_rule=predefined_outcome_rule,
        )
    )
    return out


__all__ = [
    "compute_predefined_rule_settlement_by_contract_status",
    "compute_negotiation_final_state_metrics",
    "compute_negotiation_rule_metrics",
    "primary_contract_status_factor",
]
