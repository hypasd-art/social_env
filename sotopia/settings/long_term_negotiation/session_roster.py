"""``design_1.md`` §7 — 按 session 参与方组合给的行动直觉与硬约束钩子。"""

from __future__ import annotations

from enum import Enum

from .roles import CANONICAL_NEGOTIATION_ROSTER


# 与 §1.1 一致，供 §7 会话组合分类使用。
CANONICAL_SESSION_ROLES = CANONICAL_NEGOTIATION_ROSTER


class SessionRosterKind(str, Enum):
    """粗粒度 roster 分类（用于 observation 与 §7.5 约束）。"""

    DUO_TRADE = "duo_trade"  # §7.1
    DUO_FIRM_A_INVESTOR = "duo_firm_a_investor"  # §7.2
    DUO_FIRM_B_INVESTOR = "duo_firm_b_investor"  # §7.3
    DUO_FIRM_A_REGULATOR = "duo_firm_a_regulator"  # §7.4
    DUO_FIRM_B_REGULATOR = "duo_firm_b_regulator"  # §7.4
    DUO_INVESTOR_REGULATOR = "duo_investor_regulator"  # §7.5
    DUO_OTHER = "duo_other"
    FULL_QUARTET = "full_quartet"  # §7.6 四方
    MULTILATERAL = "multilateral"  # 三方等非上列全集
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
            frozenset({"firm_a", "investor"}): SessionRosterKind.DUO_FIRM_A_INVESTOR,
            frozenset({"firm_b", "investor"}): SessionRosterKind.DUO_FIRM_B_INVESTOR,
            frozenset({"firm_a", "regulator"}): SessionRosterKind.DUO_FIRM_A_REGULATOR,
            frozenset({"firm_b", "regulator"}): SessionRosterKind.DUO_FIRM_B_REGULATOR,
            frozenset({"investor", "regulator"}): SessionRosterKind.DUO_INVESTOR_REGULATOR,
        }
        return mapping.get(ps, SessionRosterKind.DUO_OTHER)

    if ps == CANONICAL_SESSION_ROLES:
        return SessionRosterKind.FULL_QUARTET
    return SessionRosterKind.MULTILATERAL


def roster_blocks_trade_contract_drafting(participants: Iterable[str]) -> bool:
    """§7.5 — investor 与 regulator 二人的 session 不直接起草 / amend 买卖双方合同本体。"""
    return classify_session_roster(participants) == SessionRosterKind.DUO_INVESTOR_REGULATOR


_SECTION7_BASE: dict[SessionRosterKind, str] = {
    SessionRosterKind.DUO_TRADE: (
        "§7.1 (firm_a+firm_b): valuation/payment/closing/compliance/penalty messaging; "
        "propose/amend/accept/reject/sign trade contracts when visible."
    ),
    SessionRosterKind.DUO_FIRM_A_INVESTOR: (
        "§7.2 (firm_a+investor): financing terms & risk; request_financing_review on visible contracts; "
        "investor finance_commit / finance_decline on visible pending financing."
    ),
    SessionRosterKind.DUO_FIRM_B_INVESTOR: (
        "§7.3 (firm_b+investor): payment structure, protections, seller guarantees; "
        "investor comments on finance-affecting clauses; may propose_amend only if roster allows (no §7.5 block)."
    ),
    SessionRosterKind.DUO_FIRM_A_REGULATOR: (
        "§7.4 (firm_a+regulator): compliance/disclosure/approval conditions; "
        "request_regulatory_review; regulator regulatory_approve / regulatory_block on visible contracts."
    ),
    SessionRosterKind.DUO_FIRM_B_REGULATOR: (
        "§7.4 (firm_b+regulator): same regulatory collaboration pattern as §7.4 for the seller side."
    ),
    SessionRosterKind.DUO_INVESTOR_REGULATOR: (
        "§7.5 (investor+regulator): align finance vs regulatory feasibility; "
        "**Controller blocks propose_contract / amend_contract here** (no direct buyer-seller drafting). "
        "Still use messages; reference existing contracts only if already in your visibility_set."
    ),
    SessionRosterKind.DUO_OTHER: (
        "§7 (duo, non-canonical pair): use messages; formal actions still gated by role, visibility, §6 budget."
    ),
    SessionRosterKind.FULL_QUARTET: (
        "§7.6 (all four): consolidate constraints; multilateral contract visible to participants; "
        "financing + regulatory paths may all be live—respect visibility before formal ops."
    ),
    SessionRosterKind.MULTILATERAL: (
        "§7 (multilateral subset): combine §7.1–7.4 intents for who is present; "
        "every formal op still requires visibility + role + budget."
    ),
    SessionRosterKind.UNKNOWN: (
        "§7: roster contains unknown role ids — rely on generic rules; verify agent_names configuration."
    ),
}


def section7_session_hints(kind: SessionRosterKind, *, viewer: str) -> str:
    """供 observation 拼接的单段中文提示（含当前 viewer 角色提示）。"""
    base = _SECTION7_BASE.get(kind, _SECTION7_BASE[SessionRosterKind.UNKNOWN])
    return f"§7 session type `{kind.value}` (you={viewer}): {base}"


__all__ = [
    "CANONICAL_SESSION_ROLES",
    "SessionRosterKind",
    "classify_session_roster",
    "roster_blocks_trade_contract_drafting",
    "section7_session_hints",
]
