"""与设计文档 §1.1 对齐的参与者集合与默认禀赋（仅存数值型资源键，便于 ``SystemState`` 使用）。

本模块同时定义 **公司角色（firm_a..firm_d）** 与 **机构位（investor / regulator）**：

- 双家公司 + 机构（``with_institutional`` lineup）：``firm_a`` + ``firm_b`` + 0~2 个机构位，
  对应历史 ``bilat`` / ``tri`` / ``quartet`` 模式，与 design_1 §1.1 完全兼容。
- 三家及以上公司（``firms_only`` lineup）：``firm_a``/``firm_b``/``firm_c``/``firm_d`` 的前缀，
  ``investor`` / ``regulator`` 不在世界里；融资 / 监管路径自然成为 no-op，
  controller 的 ``PRINCIPAL_PARTY_ROLES & session.participants`` 决定合同主体。
"""

from __future__ import annotations

#: design_1 §1.1 的 V1 经典 4 角色（``with_institutional`` lineup 用）。
CANONICAL_NEGOTIATION_ROSTER_V1: frozenset[str] = frozenset(
    {"firm_a", "firm_b", "investor", "regulator"}
)

#: 谈判世界中受支持的 **全部** canonical 角色键（含扩展公司位 firm_c / firm_d）。
CANONICAL_NEGOTIATION_ROSTER: frozenset[str] = frozenset(
    {"firm_a", "firm_b", "firm_c", "firm_d", "investor", "regulator"}
)

#: 公司角色稳定排序：``firms_only`` lineup 取本元组前缀作为 roster。
FIRM_ROLES_ORDER: tuple[str, ...] = ("firm_a", "firm_b", "firm_c", "firm_d")
FIRM_ROLES: frozenset[str] = frozenset(FIRM_ROLES_ORDER)

#: 机构（非公司）角色：``with_institutional`` lineup 时按需挂入。
INSTITUTIONAL_ROLES: frozenset[str] = frozenset({"investor", "regulator"})

ROLE_SUMMARY_EN: dict[str, str] = {
    "firm_a": "Buyer / acquirer firm (budget and price drive financing need).",
    "firm_b": "Seller / target firm.",
    "firm_c": "Third commercial party (joint bidder, partner-investor firm or co-seller).",
    "firm_d": "Fourth commercial party (additional bidder / consortium member).",
    "investor": "External financing; may be contingent required party.",
    "regulator": "May be contingent required party for approval.",
}


def validate_canonical_negotiation_roster(agent_names: tuple[str, ...]) -> None:
    """严格按 design_1 §1.1 要求 V1 四方参与者（``strict_design_v1=True`` 时调用）。"""
    got = frozenset(agent_names)
    if got != CANONICAL_NEGOTIATION_ROSTER_V1:
        raise ValueError(
            "Design §1.1 roster must be exactly "
            f"{sorted(CANONICAL_NEGOTIATION_ROSTER_V1)}; got {sorted(agent_names)}"
        )


def default_agent_resources_bundle() -> dict[str, dict[str, float]]:
    """§1.1 ``B_i(0)`` 的 V1 标量占位（扩展到 ``SystemState.agent_resources`` 可读的 float 槽位）。

    新增 ``firm_c`` / ``firm_d``：默认现金与 firm_a/firm_b 同量级；可在 LLM 生成场景或 V2 快照时按需覆写。
    """
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
        "firm_c": {
            "cash": 400.0,
            "asset": 0.0,
            "liability": 0.0,
        },
        "firm_d": {
            "cash": 400.0,
            "asset": 0.0,
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
