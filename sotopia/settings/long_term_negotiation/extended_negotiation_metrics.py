"""合同经济学确定性化 V2 — 扩展结算计算。

扩展 payout 计算：
- V2 合作合同：直接从 predetermined_payouts 取金额，乘以合同生效状态
- V2 买卖合同：买方 profit = max(0, reference - agreed_price)，卖方 profit = max(0, agreed_price - cost_price)
- V1 规则：委托给原始函数
"""

from __future__ import annotations

from typing import Any

from .extended_types import (
    CONTRACT_TYPE_BUY_SELL,
    CONTRACT_TYPE_COOPERATION,
    PAYOUT_MODE_PREDETERMINED,
    PAYOUT_MODE_PRICE_DIFFERENCE,
    PREDETERMINED_OUTCOME_RULE_VERSION,
)
from .negotiation_metrics import (
    _clip,
    primary_contract_status_factor,
)


def _is_v2_rule(rule: dict[str, Any] | None) -> bool:
    if not isinstance(rule, dict):
        return False
    return str(rule.get("version") or "") == PREDETERMINED_OUTCOME_RULE_VERSION


def compute_v2_payout(
    *,
    env: Any,
    primary_factor: float,
    rule: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """计算 V2 规则的 payout metrics。

    返回 (metrics_dict, rule_factor) 与原始 ``_compute_predefined_rule_payout_metrics`` 兼容。
    """
    contract_type = str(rule.get("contract_type") or "")
    if contract_type == CONTRACT_TYPE_BUY_SELL:
        return _compute_v2_buy_sell_payout(
            env=env, primary_factor=primary_factor, rule=rule
        )
    else:
        return _compute_v2_cooperation_payout(
            env=env, primary_factor=primary_factor, rule=rule
        )


def _compute_v2_cooperation_payout(
    *,
    env: Any,
    primary_factor: float,
    rule: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """V2 合作合同：直接从 predetermined_payouts 取金额。"""
    out: dict[str, float] = {}
    out["negotiation_predefined_rule_enabled"] = 1.0
    out["negotiation_predefined_rule_payout_mode_v2_cooperation"] = 1.0

    contract_value = float(rule.get("contract_value", 0.0) or 0.0)
    out["negotiation_predefined_rule_contract_value"] = contract_value
    out["negotiation_predefined_rule_news_signal"] = float(
        rule.get("news_signal", 0.0) or 0.0
    )

    is_contract_effective = 1.0 if primary_factor >= 0.75 else 0.0
    out["negotiation_predefined_rule_contract_effective"] = is_contract_effective

    # V2: no runtime margin computation — use predetermined payouts directly
    out["negotiation_predefined_rule_realized_margin"] = 0.0

    payouts = rule.get("predetermined_payouts")
    payouts_dict = payouts if isinstance(payouts, dict) else {}
    total_profit = sum(
        float(v) for v in payouts_dict.values() if isinstance(v, (int, float))
    )
    out["negotiation_predefined_rule_total_profit"] = total_profit * is_contract_effective

    for role, amount in payouts_dict.items():
        if not isinstance(role, str) or not isinstance(amount, (int, float)):
            continue
        out[f"negotiation_predefined_rule_individual_profit_{role}"] = (
            float(amount) * is_contract_effective
        )

    # rule_factor: contract signed = full score, else proportional to status
    rule_factor = primary_factor * is_contract_effective
    out["negotiation_predefined_rule_score"] = rule_factor
    return out, rule_factor


def _compute_v2_buy_sell_payout(
    *,
    env: Any,
    primary_factor: float,
    rule: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """V2 买卖合同：买方 profit = max(0, reference - agreed)，卖方 profit = max(0, agreed - cost)。"""
    out: dict[str, float] = {}
    out["negotiation_predefined_rule_enabled"] = 1.0
    out["negotiation_predefined_rule_payout_mode_v2_buy_sell"] = 1.0

    contract_value = float(rule.get("contract_value", 0.0) or 0.0)
    out["negotiation_predefined_rule_contract_value"] = contract_value

    reference = float(rule.get("reference_price", 0.0) or 0.0)
    cost = float(rule.get("cost_price", 0.0) or 0.0)
    out["negotiation_predefined_rule_reference_price"] = reference
    out["negotiation_predefined_rule_cost_price"] = cost

    # Get agreed price from the primary contract
    from .negotiation_metrics import _primary_contract_price

    agreed, _c = _primary_contract_price(env)
    out["negotiation_predefined_rule_realized_price"] = agreed

    is_contract_effective = 1.0 if primary_factor >= 0.75 else 0.0
    out["negotiation_predefined_rule_contract_effective"] = is_contract_effective

    out["negotiation_predefined_rule_realized_margin"] = 0.0
    out["negotiation_predefined_rule_news_signal"] = float(
        rule.get("news_signal", 0.0) or 0.0
    )
    out["negotiation_predefined_rule_total_profit"] = 0.0

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

    buyer_savings_per_unit = max(0.0, reference - agreed) * is_contract_effective
    seller_earnings_per_unit = max(0.0, agreed - cost) * is_contract_effective

    out["negotiation_predefined_rule_buyer_savings_per_unit"] = buyer_savings_per_unit
    out["negotiation_predefined_rule_seller_earnings_per_unit"] = seller_earnings_per_unit

    # Savings ratio for scoring
    if reference > 0:
        savings_ratio = max(0.0, (reference - agreed) / reference) * is_contract_effective
    else:
        savings_ratio = 0.0
    out["negotiation_predefined_rule_buyer_savings_ratio"] = savings_ratio

    # Distribute to individual roles
    n_buy = max(1, len(buyers))
    for role in buyers:
        out[f"negotiation_predefined_rule_individual_profit_{role}"] = (
            buyer_savings_per_unit / float(n_buy)
        )

    n_sell = max(1, len(sellers))
    for role in sellers:
        out[f"negotiation_predefined_rule_individual_profit_{role}"] = (
            seller_earnings_per_unit / float(n_sell)
        )

    # rule_factor: based on savings ratio vs a target fraction
    full_frac = 0.15  # reasonable target savings fraction
    rule_factor = _clip(savings_ratio / full_frac, 0.0, 1.0) * is_contract_effective
    out["negotiation_predefined_rule_score"] = rule_factor
    return out, rule_factor


def compute_v2_settlement_by_contract_status(
    *,
    env: Any,
    predefined_outcome_rule: dict[str, Any] | None,
    contract_status: str,
) -> dict[str, float]:
    """V2 版合同状态结算：按合同状态计算应结算的个人资金变化。

    与原始 ``compute_predefined_rule_settlement_by_contract_status`` 同接口，
    但 V2 规则走确定性 payout 逻辑。
    """
    if not _is_v2_rule(predefined_outcome_rule):
        from .negotiation_metrics import compute_predefined_rule_settlement_by_contract_status

        return compute_predefined_rule_settlement_by_contract_status(
            env=env,
            predefined_outcome_rule=predefined_outcome_rule,
            contract_status=contract_status,
        )

    primary_factor = primary_contract_status_factor(contract_status)
    out, _ = compute_v2_payout(
        env=env,
        primary_factor=primary_factor,
        rule=predefined_outcome_rule,  # type: ignore[arg-type]
    )
    return out


def compute_v2_rule_payout_metrics(
    *,
    env: Any,
    primary_factor: float,
    predefined_outcome_rule: dict[str, Any] | None,
) -> tuple[dict[str, float], float]:
    """V2 兼容的 payout metrics 计算：V2 走确定性逻辑，V1 委托原始函数。"""
    if not _is_v2_rule(predefined_outcome_rule):
        from .negotiation_metrics import _compute_predefined_rule_payout_metrics

        return _compute_predefined_rule_payout_metrics(
            env=env,
            primary_factor=primary_factor,
            predefined_outcome_rule=predefined_outcome_rule,
        )

    return compute_v2_payout(
        env=env,
        primary_factor=primary_factor,
        rule=predefined_outcome_rule,  # type: ignore[arg-type]
    )


__all__ = [
    "compute_v2_payout",
    "compute_v2_settlement_by_contract_status",
    "compute_v2_rule_payout_metrics",
    "_is_v2_rule",
]
