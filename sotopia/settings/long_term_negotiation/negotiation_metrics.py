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
_FINAL_STATE_WEIGHT_SCHEDULING_EFFECTIVENESS = 0.0


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


def _scene_score_weights(predefined_outcome_rule: dict[str, Any] | None) -> dict[str, float]:
    base = {
        "terminal_success": _FINAL_STATE_WEIGHT_TERMINAL_SUCCESS,
        "primary_contract": _FINAL_STATE_WEIGHT_PRIMARY_CONTRACT,
        "solvency": _FINAL_STATE_WEIGHT_SOLVENCY,
        "liquidity_preserved": _FINAL_STATE_WEIGHT_LIQUIDITY_PRESERVED,
        "predefined_rule": _FINAL_STATE_WEIGHT_PREDEFINED_RULE,
        "scheduling_effectiveness": _FINAL_STATE_WEIGHT_SCHEDULING_EFFECTIVENESS,
    }
    if not isinstance(predefined_outcome_rule, dict):
        return base
    raw = predefined_outcome_rule.get("score_weights")
    if not isinstance(raw, dict):
        return base
    out = dict(base)
    for k in out:
        v = raw.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _scheduling_effectiveness_factor(ctrl: Any) -> float:
    logs = list(getattr(ctrl, "session_log", []) or [])
    if not logs:
        return 0.0
    rows = [r for r in logs if isinstance(r, dict) and r.get("kind") == "post_session_bookkeeping"]
    if not rows:
        return 0.0
    no_session = sum(
        1
        for r in rows
        if str(r.get("slot_closure_reason") or "") == "scheduling_yielded_no_session"
    )
    return _clip(1.0 - float(no_session) / float(len(rows)), 0.0, 1.0)


def _primary_contract_price(env: Any) -> tuple[float, Any]:
    """主合同 ``terms.price``（与 ``NegotiationWorldController`` 校验字段一致）。"""
    ctrl = getattr(env, "ctrl", None)
    if ctrl is None:
        return 0.0, None
    pcs = getattr(ctrl, "primary_contract_id", None)
    if not pcs:
        return 0.0, None
    c = getattr(ctrl, "contracts", {}).get(pcs)
    if c is None:
        return 0.0, None
    terms = getattr(c, "terms", {}) or {}
    raw = terms.get("price", 0) if isinstance(terms, dict) else 0
    try:
        return float(raw or 0), c
    except (TypeError, ValueError):
        return 0.0, c


def _compute_procurement_savings_metrics(
    *,
    env: Any,
    primary_factor: float,
    rule: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """个人采购/竞价口径：不按 margin 分红，按「成交价相对参考价的节省」计分与结算。"""
    out: dict[str, float] = {}
    out["negotiation_predefined_rule_enabled"] = 1.0
    out["negotiation_predefined_rule_payout_mode_procurement"] = 1.0

    contract_value = float(rule.get("contract_value_if_signed", 0.0) or 0.0)
    out["negotiation_predefined_rule_contract_value"] = contract_value
    reference = float(rule.get("reference_unit_price", rule.get("reference_price", 0.0)) or 0.0)
    out["negotiation_predefined_rule_reference_price"] = reference

    realized, _c = _primary_contract_price(env)
    out["negotiation_predefined_rule_realized_price"] = realized

    is_contract_effective = 1.0 if primary_factor >= 0.75 else 0.0
    out["negotiation_predefined_rule_contract_effective"] = is_contract_effective

    # 与 margin 路径区分：不产出 realized_margin / 公司分红中间量
    out["negotiation_predefined_rule_realized_margin"] = 0.0
    out["negotiation_predefined_rule_news_signal"] = float(rule.get("news_signal", 0.0) or 0.0)
    out["negotiation_predefined_rule_total_profit"] = 0.0

    full_frac = float(rule.get("full_score_savings_fraction", 0.12) or 0.12)
    if full_frac <= 1e-9:
        full_frac = 0.12

    rule_factor = 0.0
    savings = 0.0
    savings_ratio = 0.0
    if reference > 0 and is_contract_effective > 0 and realized > 0:
        savings = reference - realized
        savings_ratio = savings / reference
        if savings_ratio > 0:
            rule_factor = _clip(savings_ratio / full_frac, 0.0, 1.0) * is_contract_effective

    out["negotiation_predefined_rule_buyer_savings_per_unit"] = savings * is_contract_effective
    out["negotiation_predefined_rule_buyer_savings_ratio"] = max(0.0, savings_ratio) * is_contract_effective
    out["negotiation_predefined_rule_score"] = rule_factor

    buyers_raw = rule.get("buyer_roles")
    buyers: list[str] = (
        [str(x).strip() for x in buyers_raw if str(x).strip()]
        if isinstance(buyers_raw, (list, tuple))
        else []
    )
    if not buyers:
        buyers = ["firm_a"]

    sellers_raw = rule.get("seller_roles")
    sellers: list[str] = (
        [str(x).strip() for x in sellers_raw if str(x).strip()]
        if isinstance(sellers_raw, (list, tuple))
        else []
    )

    scale = float(rule.get("savings_cash_scale", 1.0) or 1.0)
    bonus_total = float(rule.get("seller_closure_bonus_total", 0.0) or 0.0)

    n_buy = max(1, len(buyers))

    eff_savings = max(0.0, savings) * is_contract_effective
    buyer_pool = eff_savings * scale
    for role in buyers:
        out[f"negotiation_predefined_rule_individual_profit_{role}"] = buyer_pool / float(n_buy)

    if sellers and bonus_total > 0 and is_contract_effective > 0:
        per = bonus_total / float(len(sellers))
        for role in sellers:
            out[f"negotiation_predefined_rule_individual_profit_{role}"] = (
                float(out.get(f"negotiation_predefined_rule_individual_profit_{role}", 0.0)) + per
            )

    return out, rule_factor


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

    # V2: 委托扩展模块的确定性 payout
    if str(predefined_outcome_rule.get("version") or "") == "v2":
        from .extended_negotiation_metrics import compute_v2_payout

        return compute_v2_payout(
            env=env,
            primary_factor=primary_factor,
            rule=predefined_outcome_rule,
        )

    if str(predefined_outcome_rule.get("payout_mode") or "").strip() == "procurement_savings":
        return _compute_procurement_savings_metrics(
            env=env,
            primary_factor=primary_factor,
            rule=predefined_outcome_rule,
        )

    out["negotiation_predefined_rule_enabled"] = 1.0
    out["negotiation_predefined_rule_payout_mode_procurement"] = 0.0
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
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
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
    scheduling_factor = _scheduling_effectiveness_factor(ctrl)
    out["negotiation_scheduling_effectiveness_factor"] = scheduling_factor
    weights = _scene_score_weights(predefined_outcome_rule)

    score = (
        weights["terminal_success"] * success_factor
        + weights["primary_contract"] * primary_factor
        + weights["solvency"] * solvency_factor
        + weights["liquidity_preserved"] * liquidity_factor
        + weights["scheduling_effectiveness"] * scheduling_factor
    )
    rule_out, rule_factor = _compute_predefined_rule_payout_metrics(
        env=env,
        primary_factor=primary_factor,
        predefined_outcome_rule=predefined_outcome_rule,
    )
    out.update(rule_out)
    score += weights["predefined_rule"] * rule_factor
    out["negotiation_final_state_score"] = float(max(0.0, min(1.0, score)))
    out["negotiation_final_state_score_component_terminal_success"] = (
        weights["terminal_success"] * success_factor
    )
    out["negotiation_final_state_score_component_primary_contract"] = (
        weights["primary_contract"] * primary_factor
    )
    out["negotiation_final_state_score_component_solvency"] = (
        weights["solvency"] * solvency_factor
    )
    out["negotiation_final_state_score_component_liquidity_preserved"] = (
        weights["liquidity_preserved"] * liquidity_factor
    )
    out["negotiation_final_state_score_component_predefined_rule"] = (
        weights["predefined_rule"] * rule_factor
    )
    out["negotiation_final_state_score_component_scheduling_effectiveness"] = (
        weights["scheduling_effectiveness"] * scheduling_factor
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


def _negotiation_contract_to_jsonable(c: Any, *, history_tail_max: int) -> dict[str, Any]:
    """将 ``NegotiationContract`` 压成可 JSON 序列化的 dict（合同 history 仅保留尾部）。"""
    parties = getattr(c, "parties", None) or set()
    visibility = getattr(c, "visibility", None) or set()
    hist = list(getattr(c, "history", []) or [])
    if history_tail_max > 0 and len(hist) > history_tail_max:
        hist = hist[-history_tail_max:]
    acc = getattr(c, "acceptances", None) or {}
    sig = getattr(c, "signatures", None) or {}
    return {
        "contract_id": str(getattr(c, "contract_id", "") or ""),
        "parent_id": getattr(c, "parent_id", None),
        "status": str(getattr(c, "status", "") or ""),
        "terms": dict(getattr(c, "terms", {}) or {}),
        "created_by": str(getattr(c, "created_by", "") or ""),
        "created_at": dict(getattr(c, "created_at", {}) or {}),
        "parties": sorted(str(p) for p in parties),
        "acceptances": {str(k): v for k, v in dict(acc).items()},
        "visibility": sorted(str(v) for v in visibility),
        "signatures": {str(k): bool(v) for k, v in dict(sig).items()},
        "financing": dict(getattr(c, "financing", {}) or {}),
        "regulatory": dict(getattr(c, "regulatory", {}) or {}),
        "history_tail": list(hist),
        "created_day": int(getattr(c, "created_day", 0) or 0),
        "created_slot": int(getattr(c, "created_slot", 0) or 0),
    }


def build_rule_evaluation_state_record(
    env: Any,
    *,
    predefined_outcome_rule: dict[str, Any] | None = None,
    contract_history_tail_max: int = 40,
) -> dict[str, Any]:
    """采集与 ``compute_negotiation_rule_metrics`` 口径对齐的**终局状态**快照，便于评测落盘复现。

    - ``state_snapshot_for_rule_metrics``：与 ``compute_negotiation_final_state_metrics`` 相同，
      取 ``ctrl.state_snapshots`` 的**最后一条**（episode 末尾 ``after_terminal`` 写入）。
    - ``system_state_agent_resources_end``：``env.system_state.agent_resources`` 终值（与部分
      ``negotiation_participant_*`` 指标同源）。
    - ``contracts``：当前控制器合同账本摘要（history 截断）。
    - ``predefined_outcome_rule_used``：若调用方传入非空 dict，则回显本次规则计算所用参数。
    """
    ctrl = getattr(env, "ctrl", None)
    if ctrl is None:
        return {"error": "missing_controller"}

    snap = _final_intermediate_snapshot(ctrl)
    snap_out: dict[str, Any] | None = dict(snap) if snap else None

    st = getattr(env, "system_state", None)
    system_resources: dict[str, Any] | None = None
    if st is not None:
        keys = list(getattr(st, "agent_keys", []) or [])
        ar = getattr(st, "agent_resources", {}) or {}
        system_resources = {str(k): dict(ar.get(k, {}) or {}) for k in keys}

    contracts_raw = getattr(ctrl, "contracts", {}) or {}
    contracts_out: dict[str, Any] = {}
    for cid, c in contracts_raw.items():
        contracts_out[str(cid)] = _negotiation_contract_to_jsonable(
            c, history_tail_max=contract_history_tail_max
        )

    out: dict[str, Any] = {
        "state_snapshot_for_rule_metrics": snap_out,
        "n_state_snapshots": len(list(getattr(ctrl, "state_snapshots", []) or [])),
        "system_state_agent_resources_end": system_resources,
        "primary_contract_id": getattr(ctrl, "primary_contract_id", None),
        "contracts": contracts_out,
        "controller_terminal": str(getattr(ctrl, "terminal", "") or ""),
    }
    if isinstance(predefined_outcome_rule, dict):
        out["predefined_outcome_rule_used"] = dict(predefined_outcome_rule)
    return out


__all__ = [
    "build_rule_evaluation_state_record",
    "compute_predefined_rule_settlement_by_contract_status",
    "compute_negotiation_final_state_metrics",
    "compute_negotiation_rule_metrics",
    "primary_contract_status_factor",
]
