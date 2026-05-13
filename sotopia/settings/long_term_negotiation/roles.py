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

#: 叙事层为 **个人**（摊主 / 买主 / 个体出资方 / 基层监管联系人）；协议里仍用 ``firm_*`` 等 canonical id。
ROLE_SUMMARY_EN: dict[str, str] = {
    "firm_a": (
        "Individual buyer / procurement lead (personal or household budget). "
        "You compare rival vendors in the same trade lane; customers pick the best total offer "
        "(price, freshness, delivery window, after-sales)."
    ),
    "firm_b": (
        "Primary vendor / counterparty — sole trader or stall operator selling goods or services. "
        "Peers nearby sell similar SKUs; you must justify why your bundle wins the walk-by customer."
    ),
    "firm_c": (
        "Third independent operator (co-seller, co-bidder, or parallel vendor). "
        "You compete on margin and trust with the other vendors while seeking workable joint terms."
    ),
    "firm_d": (
        "Fourth independent operator (late entrant vendor or alternate supplier). "
        "You undercut or differentiate against overlapping offers; watch reputation and stock."
    ),
    "investor": (
        "Individual financier / informal capital partner (not a bank brand). "
        "May join sessions when contingent funding is formally requested."
    ),
    "regulator": (
        "Individual compliance / market-hall coordinator (approval or stall rules). "
        "May join sessions when regulatory review is formally requested."
    ),
}

#: 与默认人画像姓名一致；观测与 LLM 提示用展示名，控制器内部仍用 canonical roster 键。
ROLE_DEFAULT_DISPLAY_NAME_EN: dict[str, str] = {
    "firm_a": "Riley Carter",
    "firm_b": "Jordan Hayes",
    "firm_c": "Avery Singh",
    "firm_d": "Cameron Doyle",
    "investor": "Morgan Bennett",
    "regulator": "Casey Park",
}


def default_display_name_for_role(role: str) -> str:
    """自然语言里使用的参与者展示名（无表项时退回 canonical 键）。"""
    return ROLE_DEFAULT_DISPLAY_NAME_EN.get(role, role)


#: 更细粒度的自然人画像（用于数据构建与 agent 推理提示），字段均为可读文本，不影响协议动作。
ROLE_PERSONA_EN: dict[str, dict[str, object]] = {
    "firm_a": {
        "background_story": "Morning-batch buyer for a neighborhood canteen; tracks demand swings by weekday.",
        "personality": "decisive but risk-aware",
        "dialogue_voice": (
            "Register: plainspoken, clock-aware, slightly tired humor before noon. "
            "Pacing: short sentences when stressed; slows down when listing bundle tradeoffs. "
            "Habits: asks clarifying quantity/time questions; avoids flowery praise; occasional dry one-liner. "
            "Avoid: corporate buzzwords, long monologues, talking down to vendors."
        ),
        "core_skills": ["offer comparison", "bundle negotiation", "demand forecasting"],
        "survival_pressure": "Must secure supply before noon or loses lunch traffic and incurs overtime penalties.",
        "daily_fixed_cost": 65.0,
        "short_term_debt_due": 120.0,
        "achievement_motivation": "Build a trusted buyer reputation with repeat-customer retention > 70%.",
    },
    "firm_b": {
        "background_story": "Incumbent produce stall owner with stable regulars but thinner margin this quarter.",
        "personality": "steady and relationship-focused",
        "dialogue_voice": (
            "Register: warm neighborly, story-led anecdotes about regulars and seasons. "
            "Pacing: measured; repeats key numbers twice when nervous. "
            "Habits: deflects direct price with freshness/delivery narrative; uses 'we' for stall community. "
            "Avoid: aggressive ultimatums unless cornered; slang that sounds performative."
        ),
        "core_skills": ["quality signaling", "inventory rotation", "repeat-client management"],
        "survival_pressure": "Daily spoilage cost grows sharply after afternoon; unsold stock directly hurts cashflow.",
        "daily_fixed_cost": 80.0,
        "short_term_debt_due": 90.0,
        "achievement_motivation": "Maintain top-3 trust ranking in the market lane for two consecutive weeks.",
    },
    "firm_c": {
        "background_story": "Challenger vendor recently entered the same product category with flexible sourcing.",
        "personality": "aggressive and opportunistic",
        "dialogue_voice": (
            "Register: street-fast, competitive, peppered with 'today-only' urgency. "
            "Pacing: bursts of offers then sudden silence as a tactic. "
            "Habits: names rival stalls obliquely; stacks options A/B/C to force comparison. "
            "Avoid: long policy lectures; sounding apologetic about undercuts."
        ),
        "core_skills": ["price undercutting", "fast fulfillment", "cross-sell packaging"],
        "survival_pressure": "Needs quick turnover to pay supplier credit each evening.",
        "daily_fixed_cost": 72.0,
        "short_term_debt_due": 150.0,
        "achievement_motivation": "Win at least one anchor customer from incumbent competitors this cycle.",
    },
    "firm_d": {
        "background_story": "Late-entrant specialist seller with niche quality claim and limited storage.",
        "personality": "calculating and persuasive",
        "dialogue_voice": (
            "Register: polished-minimalist pitch voice; calm confidence with precise adjectives. "
            "Pacing: long setup then sharp close; uses rhetorical questions sparingly for effect. "
            "Habits: cites scarcity and cold-chain limits; avoids shouting matches. "
            "Avoid: rambling small talk; discounting without framing value."
        ),
        "core_skills": ["premium positioning", "deadline bargaining", "scarcity framing"],
        "survival_pressure": "Limited cold-chain capacity forces strict daily sell-through targets.",
        "daily_fixed_cost": 68.0,
        "short_term_debt_due": 110.0,
        "achievement_motivation": "Establish a premium brand and avoid competing only on lowest price.",
    },
    "investor": {
        "background_story": "Independent capital partner serving multiple micro-business operators.",
        "personality": "selective and conservative",
        "dialogue_voice": (
            "Register: clipped professional, numbers-forward, mild skepticism as default tone. "
            "Pacing: bullet-like conditions; pauses before 'non-negotiable' items. "
            "Habits: reframes stories into runway/tranche vocabulary; asks for downside cases. "
            "Avoid: emotional pep talks; vague promises without triggers."
        ),
        "core_skills": ["risk screening", "cashflow stress test", "contingent financing design"],
        "survival_pressure": "Portfolio drawdown limit forces strict downside checks on every commitment.",
        "daily_fixed_cost": 40.0,
        "short_term_debt_due": 0.0,
        "achievement_motivation": "Keep non-performing financing ratio below 5% while preserving deal volume.",
    },
    "regulator": {
        "background_story": "Market-hall compliance coordinator balancing fairness and enforceability.",
        "personality": "principled and strict",
        "dialogue_voice": (
            "Register: procedural, neutral third-person, cites stall rules and calendars. "
            "Pacing: even and slow when de-escalating; firm terminality on hard lines. "
            "Habits: summarizes dispute in two sentences then points to applicable clause. "
            "Avoid: picking commercial winners; snark; personal gossip."
        ),
        "core_skills": ["rule interpretation", "procedural review", "dispute mediation"],
        "survival_pressure": "Escalation quota penalizes delayed decisions on high-impact disputes.",
        "daily_fixed_cost": 35.0,
        "short_term_debt_due": 0.0,
        "achievement_motivation": "Maintain transparent, consistent rulings with low appeal reversal rate.",
    },
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
            "cash": 430.0,
            "asset": 0.0,
            "liability": 0.0,
            "daily_fixed_cost": 65.0,
            "short_term_debt_due": 120.0,
        },
        "firm_b": {
            "cash": 390.0,
            "asset": 1.0,
            "liability": 0.0,
            "daily_fixed_cost": 80.0,
            "short_term_debt_due": 90.0,
        },
        "firm_c": {
            "cash": 360.0,
            "asset": 0.0,
            "liability": 0.0,
            "daily_fixed_cost": 72.0,
            "short_term_debt_due": 150.0,
        },
        "firm_d": {
            "cash": 380.0,
            "asset": 0.0,
            "liability": 0.0,
            "daily_fixed_cost": 68.0,
            "short_term_debt_due": 110.0,
        },
        "investor": {
            "cash": 500.0,
            "deployable_capital": 500.0,
            "asset": 0.0,
            "liability": 0.0,
            "daily_fixed_cost": 40.0,
            "short_term_debt_due": 0.0,
        },
        "regulator": {
            "cash": 0.0,
            "institutional_credibility": 80.0,
            "asset": 0.0,
            "liability": 0.0,
            "daily_fixed_cost": 35.0,
            "short_term_debt_due": 0.0,
        },
    }
