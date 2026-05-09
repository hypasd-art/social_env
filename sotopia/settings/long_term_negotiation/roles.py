"""与设计文档 §1.1 对齐的参与者集合与默认禀赋（仅存数值型资源键，便于 ``SystemState`` 使用）。"""

from __future__ import annotations

CANONICAL_NEGOTIATION_ROSTER: frozenset[str] = frozenset(
    {"firm_a", "firm_b", "investor", "regulator"}
)

ROLE_SUMMARY_EN: dict[str, str] = {
    "firm_a": "Buyer / acquirer (budget and price drive financing need).",
    "firm_b": "Seller / target firm.",
    "investor": "External financing; may be contingent required party.",
    "regulator": "May be contingent required party for approval.",
}


def validate_canonical_negotiation_roster(agent_names: tuple[str, ...]) -> None:
    """严格按设计文档要求四方参与者时调用。"""
    got = frozenset(agent_names)
    if got != CANONICAL_NEGOTIATION_ROSTER:
        raise ValueError(
            "Design §1.1 roster must be exactly "
            f"{sorted(CANONICAL_NEGOTIATION_ROSTER)}; got {sorted(agent_names)}"
        )


def default_agent_resources_bundle() -> dict[str, dict[str, float]]:
    """§1.1 ``B_i(0)`` 的 V1 标量占位（扩展到 ``SystemState.agent_resources`` 可读的 float 槽位）。"""
    return {
        "firm_a": {
            "cash": 400.0,
            "asset": 0.0,
            "liability": 0.0,
        },
        "firm_b": {
            "cash": 400.0,
            "asset": 1.0,
            "liability": 0.0,
        },
        "investor": {
            "cash": 500.0,
            "deployable_capital": 500.0,
            "asset": 0.0,
            "liability": 0.0,
        },
        "regulator": {
            "cash": 0.0,
            "institutional_credibility": 80.0,
            "asset": 0.0,
            "liability": 0.0,
        },
    }
