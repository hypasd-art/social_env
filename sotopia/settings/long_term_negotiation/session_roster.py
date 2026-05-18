"""``design_1.md`` §7 — 按 session 参与方组合给的行动直觉与硬约束钩子。"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from .roles import CANONICAL_NEGOTIATION_ROSTER


# 与 §1.1 一致，供 §7 会话组合分类使用。
CANONICAL_SESSION_ROLES = CANONICAL_NEGOTIATION_ROSTER


class SessionRosterKind(str, Enum):
    """粗粒度 roster 分类（用于 observation）。"""

    DUO_TRADE = "duo_trade"
    DUO_OTHER = "duo_other"
    MULTILATERAL = "multilateral"
    UNKNOWN = "unknown"


def classify_session_roster(participants: Iterable[str]) -> SessionRosterKind:
    ps = frozenset(participants)
    if len(ps) < 2:
        return SessionRosterKind.UNKNOWN
    unknown = [p for p in ps if p not in CANONICAL_SESSION_ROLES]
    if unknown:
        return SessionRosterKind.UNKNOWN

    if len(ps) == 2:
        mapping: dict[frozenset[str], SessionRosterKind] = {
            frozenset({"firm_a", "firm_b"}): SessionRosterKind.DUO_TRADE,
        }
        return mapping.get(ps, SessionRosterKind.DUO_OTHER)

    return SessionRosterKind.MULTILATERAL


def roster_blocks_trade_contract_drafting(participants: Iterable[str]) -> bool:
    """无机构角色，无特殊约束；始终返回 False。"""
    _ = participants
    return False


_SECTION7_BASE: dict[SessionRosterKind, str] = {
    SessionRosterKind.DUO_TRADE: (
        "§7.1 (firm_a+firm_b): valuation/payment/closing/compliance/penalty messaging; "
        "propose/amend/accept/reject/sign trade contracts when visible."
    ),
    SessionRosterKind.DUO_OTHER: (
        "§7 (duo, non-canonical pair): use messages; formal actions still gated by role, visibility, §6 budget."
    ),
    SessionRosterKind.MULTILATERAL: (
        "§7 (multilateral subset): combine §7.1 intents for who is present; "
        "every formal op still requires visibility + role + budget."
    ),
    SessionRosterKind.UNKNOWN: (
        "§7: roster contains unknown role ids — rely on generic rules; verify agent_names configuration."
    ),
}


def section7_session_hints(kind: SessionRosterKind, *, viewer: str) -> str:
    """将 §7 的会话层面行动直觉注为 observation 前缀。"""
    base = _SECTION7_BASE.get(kind, "")
    if not base:
        return ""
    return f"[session §7 — roster={kind.value} viewer={viewer}]\n{base}\n"
