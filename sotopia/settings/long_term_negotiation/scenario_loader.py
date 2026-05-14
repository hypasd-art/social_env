"""从 benchmark 存储层加载「长期谈判」场景（``EnvironmentProfile`` + ``game_metadata``）。

脚本 ``scripts/generate_long_term_negotiation_scenarios.py`` 或 ``scripts/generate_long_term_negotiation_llm.py`` 写入::

    EnvironmentProfile.game_metadata = {
        "pipeline": "long_term_negotiation",
        "quartet": bool,
        "num_participants": int | 缺失,  # 2–4；缺失时按 quartet 推断（False→2，True→4）
        "lineup": "with_institutional" | "firms_only",  # 缺失则 with_institutional
        "strict_design_v1": bool,
        "timeline": NegotiationTimelineParams 的 dict（``dataclasses.asdict`` 形态）
        "codename": str,
        "predefined_outcome_rule": dict,  # v1；含 ``payout_mode``：``margin_split``（合资分红）或 ``procurement_savings``（零售价低于参考价）；合成时可带 ``outcome_rule_entropy``
        "predefined_news_briefs": list[dict],  # 与 scenario_text / environment_context / 阵容绑定；可含 ``scenario_relevance: "decoy"`` 的干扰条，条数随场景变化
        "deal_closure_pressure": dict,  # v1；对 1–2 名参与者附加「须在本 episode 内尽量成交」的叙事压力（软约束，仅进私有 goal）
        "dialogue_style": dict,  # v1；对话风格合成/终局 LLM 评测 rubric（见 scenario_loader 常量）
        "negotiation_relationship_design": dict,  # firm 两两竞争等设计约束（见 scenario_loader 返回）
        ...
    }

``lineup`` 决定按哪一种顺序取 N 名 canonical 角色：

- ``with_institutional``：``SESSION_SPEAKER_ROLE_ORDER`` 前缀
  （N=2 → firm_a/firm_b；N=3 → +investor；N=4 → +regulator）。
- ``firms_only``：``SESSION_FIRMS_ONLY_ROLE_ORDER`` 前缀
  （N=2 → firm_a/firm_b；N=3 → +firm_c；N=4 → +firm_d）—— 用于 **3+ 家公司**互谈。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .types import (
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    NegotiationTimelineParams,
    SUPPORTED_NEGOTIATION_LINEUPS,
    negotiation_role_order,
)

#: 追加到 ``agenerate_env_profile`` 的 inspiration 末尾，约束场景与目标中的**可区分对话风格**。
DIALOGUE_STYLE_SYNTHESIS_APPEND_EN = """
[dialogue_style_synthesis — benchmark mandatory]
1) **Per-role spoken identity:** In `scenario` and each `agent_goals` entry, make *dialogue style* explicit: default register
(plain / formal / blunt / warm), typical openers or fillers, pacing (terse vs chatty), taboos (numbers-first vs story-first),
and how refusals or concessions sound. Styles MUST differ across roster so transcripts are not interchangeable.
2) **Competitive tone differentiation — REQUIRED:** Each role's dialogue style MUST encode a distinct competitive posture:
   - At least one role uses *aggressive challenger* tone (fast-paced, undercuts openly, names rival weaknesses).
   - At least one role uses *defensive incumbent* tone (warm but guarded, deflects price attacks with quality narrative).
   - If 3+ roles, include a *calculating latecomer* tone (precise, cites scarcity, frames premium as insurance).
   Competitive voices must sound like real marketplace rivals, not polite committee members.
3) **Scene-appropriate voice:** Wet-market talk may be rapid, sensory, gossip-as-signal; coopetition may mix cautious
promises with competitive hedges; scheduling scenes keep time/slot jargon natural without collapsing into generic legal boilerplate.
4) **No schema echo:** Never paste JSON Schema boilerplate into narrative fields.
""".strip()

#: 终局 ``EpisodeLLMEvaluator`` 历史前缀：指导将「对话风格执行」并入现有维度（believability / knowledge / goal 等）。
DIALOGUE_STYLE_EVAL_RUBRIC_EN = """
When scoring each participant, explicitly weight **dialogue-style execution**:
- **believability:** Does speech match the stated persona (register, pacing, hedges) and remain distinct from other agents?
- **knowledge:** Are leaks, vagueness, or feints plausible for that voice?
- **goal:** Does tone support or undermine declared objectives?
- **competitive authenticity:** Does the agent's speech reflect genuine marketplace rivalry? Penalize agents who
  sound like polite committee members when their persona demands aggressive undercutting, reputation attacks,
  or defensive maneuvering. Reward agents whose dialogue conveys real competitive stakes.
Penalize monotone corporate-speak if the scenario promised differentiated voices; penalize near-identical phrasing across agents.
""".strip()


def _bounded(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _rule_seed(*parts: str) -> int:
    key = "|".join(parts)
    # Deterministic seed: keep the same rule per codename.
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _infer_news_sentiment_signal(scenario_text: str) -> float:
    txt = (scenario_text or "").lower()
    pos_kw = (
        "growth",
        "synergy",
        "expansion",
        "premium",
        "strong demand",
        "scale",
        "foot traffic",
        "repeat customer",
        "sell-out",
        "best offer",
    )
    neg_kw = (
        "distressed",
        "debt",
        "default",
        "layoff",
        "antitrust",
        "litigation",
        "loss",
        "spoiled",
        "shortage",
        "price war",
        "walk-away",
    )
    score = 0.0
    score += sum(1.0 for kw in pos_kw if kw in txt)
    score -= sum(1.0 for kw in neg_kw if kw in txt)
    return _bounded(score / 4.0, -1.0, 1.0)


def _build_predefined_outcome_rule(
    *,
    codename: str,
    lineup: str,
    num_participants: int,
    scenario_text: str = "",
    outcome_rule_entropy: str | None = None,
    scene_type: str | None = None,
) -> dict[str, Any]:
    """构造 ``predefined_outcome_rule`` v1。

    **payout_mode**（由 ``scene_type`` 驱动）：

    - ``wet_market_competition`` / ``competitive_bidding`` → ``procurement_savings``：
      用主合同 ``terms.price`` 相对 ``reference_unit_price`` 的节省比例计
      ``negotiation_predefined_rule_score``，现金结算按 ``buyer_roles``/``seller_roles`` 写入
      ``negotiation_predefined_rule_individual_profit_*``（见 ``negotiation_metrics``）。
    - 其余常见场景类型 → ``margin_split``：沿用 margin×合同额与公司/个人分成。

    ``outcome_rule_entropy``：非空时混入 ``deterministic_seed`` 的哈希输入，使同一
    ``codename + scenario_text`` 前缀下仍可得到不同合同经济参数（LLM 批量合成默认需要）；
    手写题库可省略以保持仅由 codename/阵容/叙事前缀决定的可复现性。
    """
    roles = tuple(negotiation_role_order(lineup)[:num_participants])
    company_roles = tuple(r for r in roles if r.startswith("firm_"))
    seed_parts: tuple[str, ...] = (
        codename,
        lineup,
        str(num_participants),
        scenario_text[:512],
    )
    if outcome_rule_entropy:
        seed_parts = seed_parts + (str(outcome_rule_entropy),)
    seed = _rule_seed(*seed_parts)
    rng = random.Random(seed)

    # Contract economics: news gives directional hint but not full determinism.
    base_sentiment = _infer_news_sentiment_signal(scenario_text)
    jitter = rng.uniform(-0.35, 0.35)
    news_signal = _bounded(base_sentiment * 0.7 + jitter * 0.3, -1.0, 1.0)
    base_margin = rng.uniform(0.03, 0.12)
    profit_margin_bounds = (-0.25, 0.35)
    contract_value = float(rng.randint(120, 420)) * 1_000_000.0

    # Profit shares are generated on company roles only.
    raw = [max(0.05, rng.random()) for _ in company_roles]
    den = sum(raw) if raw else 1.0
    company_profit_share = {
        role: float(v / den) for role, v in zip(company_roles, raw, strict=False)
    }
    # Personal payoff: each role's individual score takes a slice of its company benefit.
    individual_income_share = {role: float(rng.uniform(0.3, 0.75)) for role in roles}

    st = str(scene_type or "wet_market_competition")

    # 个人采购 / 竞价：评测与结算走 ``payout_mode=procurement_savings``（成交价 vs 参考价），
    # 不用 margin×合同价 的分红口径。见 ``negotiation_metrics._compute_procurement_savings_metrics``。
    _procurement_scenes = frozenset({"wet_market_competition", "competitive_bidding"})
    if st in _procurement_scenes:
        reference_unit_price = float(rng.uniform(175.0, 395.0))
        buyer_roles = [company_roles[0]] if company_roles else ["firm_a"]
        seller_roles = [r for r in company_roles if r not in buyer_roles]
        score_weights_wm: dict[str, float] = {
            "terminal_success": 0.22,
            "primary_contract": 0.20,
            "solvency": 0.18,
            "liquidity_preserved": 0.15,
            "predefined_rule": 0.25,
        }
        score_weights_cb: dict[str, float] = {
            "terminal_success": 0.20,
            "primary_contract": 0.22,
            "solvency": 0.17,
            "liquidity_preserved": 0.13,
            "predefined_rule": 0.28,
        }
        score_weights = score_weights_cb if st == "competitive_bidding" else score_weights_wm
        closure_pool = float(rng.uniform(5.0, 16.0)) * max(1, len(seller_roles))
        return {
            "version": "v1",
            "rule_profile": "procurement_savings",
            "payout_mode": "procurement_savings",
            "contract_name": f"predefined_{codename}_main_contract",
            "deterministic_seed": seed,
            "contract_value_if_signed": contract_value,
            "reference_unit_price": reference_unit_price,
            "full_score_savings_fraction": float(rng.uniform(0.12, 0.22)),
            "savings_cash_scale": 1.0,
            "seller_closure_bonus_total": closure_pool,
            "buyer_roles": list(buyer_roles),
            "seller_roles": list(seller_roles),
            "news_signal": news_signal,
            "profit_margin_bounds": [profit_margin_bounds[0], profit_margin_bounds[1]],
            "margin_formula": {
                "base_margin": base_margin,
                "news_weight": 0.55,
                "execution_weight": 0.45,
            },
            "company_profit_share": {},
            "individual_income_share": {},
            "score_weights": score_weights,
            "notes": (
                "Procurement-style rule: primary score uses agreed terms.price vs reference_unit_price "
                "(buyer savings). Cash settlement credits buyers with per-unit savings × savings_cash_scale "
                "and optional seller_closure_bonus_total. Requires larger savings fraction for full score "
                "(higher competitive bar). Not a joint-venture margin split."
            ),
        }

    profile = "wet_market_competition"
    score_weights: dict[str, float] = {
        "terminal_success": 0.25,
        "primary_contract": 0.10,
        "solvency": 0.25,
        "liquidity_preserved": 0.25,
        "predefined_rule": 0.15,
    }
    if st in {"business_coopetition", "business_outsourcing"}:
        profile = "business_coopetition"
        score_weights = {
            "terminal_success": 0.25,
            "primary_contract": 0.25,
            "solvency": 0.15,
            "liquidity_preserved": 0.10,
            "predefined_rule": 0.25,
        }
    elif st == "resource_scheduling_management":
        profile = "resource_scheduling_management"
        score_weights = {
            "terminal_success": 0.20,
            "primary_contract": 0.20,
            "solvency": 0.15,
            "liquidity_preserved": 0.10,
            "predefined_rule": 0.15,
            "scheduling_effectiveness": 0.20,
        }

    return {
        "version": "v1",
        "rule_profile": profile,
        "payout_mode": "margin_split",
        "contract_name": f"predefined_{codename}_main_contract",
        "deterministic_seed": seed,
        "contract_value_if_signed": contract_value,
        "profit_margin_bounds": [profit_margin_bounds[0], profit_margin_bounds[1]],
        "margin_formula": {
            "base_margin": base_margin,
            "news_weight": 0.55,
            "execution_weight": 0.45,
        },
        "news_signal": news_signal,
        "company_profit_share": company_profit_share,
        "individual_income_share": individual_income_share,
        "score_weights": score_weights,
        "notes": (
            "Predefined scoring rule generated at data-construction time (margin-split / joint payoff). "
            "Narrative roles are individuals (vendors/buyers); canonical ids remain firm_* for the simulator. "
            "Keys company_profit_share refer to those principal roles; evaluation maps them via individual_income_share."
        ),
    }


def _collect_scenario_bound_news_threads(
    *,
    scenario_text: str,
    environment_context: Mapping[str, Any],
    lineup: str,
    num_participants: int,
    calendar_days: int | None,
) -> list[str]:
    """按测试场景（环境推断的 scene_type + 叙事文本 + 阵容/人数/日历）收集新闻线索；条数不固定。"""
    txt = (scenario_text or "").lower()
    scene = str(environment_context.get("scene_type") or "")
    roles = tuple(negotiation_role_order(lineup)[:num_participants])
    threads: list[str] = []

    def add(tid: str) -> None:
        if tid not in threads:
            threads.append(tid)

    # --- 与 ``_infer_environment_context`` 的 scene_type 对齐（主绑定），避免与关键词分支重复堆叠 ---
    if scene == "wet_market_competition":
        add("perishable_supply_and_cold_chain_squeeze")
        add("parallel_vendors_price_overlap_and_reputation_risk")
        add("buyer_basket_comparison_and_substitution_threat")
    elif scene in {"business_outsourcing", "business_coopetition"}:
        add("labor_bench_shortage_and_skill_mismatch")
        add("milestone_acceptance_disputes_and_change_orders")
        add("subcontractor_chain_liability_and_payment_lag")
    elif scene in {"competitive_bidding", "resource_scheduling_management"}:
        add("bid_spread_manipulation_rumors_and_leakage_risk")
        add("technical_compliance_vs_headline_price_tradeoff")
        add("evaluation_committee_split_and_procurement_delay")
        add("multi_party_default_blame_and_partial_performance")
    else:
        # 未知 scene：用语义关键词回退
        if any(w in txt for w in ("wet market", "stall", "produce", "foot traffic", "hawker", "spoil")):
            add("perishable_supply_and_cold_chain_squeeze")
            add("parallel_vendors_price_overlap_and_reputation_risk")
        if any(w in txt for w in ("outsourc", "labor", "workforce", "milestone", "rework", "sla")):
            add("labor_bench_shortage_and_skill_mismatch")
            add("milestone_acceptance_disputes_and_change_orders")
        if any(w in txt for w in ("bid", "tender", "auction", "reserve", "lowball")):
            add("bid_spread_manipulation_rumors_and_leakage_risk")
            add("technical_compliance_vs_headline_price_tradeoff")

    # --- 阵容 / 人数绑定的叙事压力 ---
    if num_participants >= 3:
        add("multi_party_default_blame_and_partial_performance")
    if num_participants >= 4:
        add("late_entrant_undercut_and_anchor_customer_poaching")
    if "investor" in roles:
        add("informal_financing_spreads_and_contingent_drawdown_clauses")
    if "regulator" in roles:
        add("stall_rules_enforcement_and_selective_inspection_calendar")

    # --- 日历跨度：长周期场景多一条宏观流动性叙事（仍由 codename+rule 种子决定是否在正文中强调）---
    if calendar_days is not None and int(calendar_days) >= 8:
        add("multi_week_working_capital_rollover_and_supplier_credit_tightening")

    if not threads:
        add("sector_liquidity_and_inventory_financing_stress")
        add("customer_choice_shift_and_bundle_comparison_noise")

    # 上限避免 metadata 过大；条数仍随场景变化
    return threads[:8]


def _news_delivery_day(
    thread_id: str,
    calendar_days: int,
    *,
    rng: random.Random,
    relevance: str = "scenario_bound",
) -> int:
    """按线索类型 + 日历总天数将新闻分配到叙事节奏的三段：铺垫 → 竞争 → 收束。

    decoy 新闻随机散布。
    """
    if relevance == "decoy":
        return max(1, rng.randint(1, max(1, calendar_days)))

    D = max(1, int(calendar_days))
    early_hi = max(1, D // 3)
    mid_hi = max(early_hi + 1, 2 * D // 3)

    early_threads = {
        "sector_liquidity_and_inventory_financing_stress",
        "customer_choice_shift_and_bundle_comparison_noise",
        "perishable_supply_and_cold_chain_squeeze",
    }
    late_threads = {
        "late_entrant_undercut_and_anchor_customer_poaching",
        "evaluation_committee_split_and_procurement_delay",
        "informal_financing_spreads_and_contingent_drawdown_clauses",
        "stall_rules_enforcement_and_selective_inspection_calendar",
        "multi_week_working_capital_rollover_and_supplier_credit_tightening",
    }

    if thread_id in early_threads:
        return max(1, rng.randint(1, early_hi))
    if thread_id in late_threads:
        return max(early_hi + 1, rng.randint(mid_hi, D))
    # 其余竞争压力类线索放在中段
    return rng.randint(early_hi + 1, mid_hi)


def _brief_for_thread(
    *,
    codename: str,
    thread_id: str,
    rule: Mapping[str, Any],
    scenario_excerpt: str,
    environment_context: Mapping[str, Any],
    lineup: str,
    roles: tuple[str, ...],
    rng: random.Random,
    calendar_days: int = 8,
) -> dict[str, Any]:
    """单条新闻：多句、含冲突解读与角色影响，便于 agent 推理。"""
    base_signal = float(rule.get("news_signal", 0.0) or 0.0)
    # 同线索内稳定但与其他线索可区分的 hint
    jitter = rng.uniform(-0.22, 0.22)
    signal_hint = _bounded(base_signal + jitter + rng.uniform(-0.12, 0.12), -1.0, 1.0)
    corr_roll = rng.random()
    if corr_roll < 0.25:
        correlation_level = "low"
    elif corr_roll < 0.55:
        correlation_level = "partial"
    elif corr_roll < 0.85:
        correlation_level = "medium"
    else:
        correlation_level = "high"

    scene_type = str(environment_context.get("scene_type") or "unknown")
    cues = list(environment_context.get("agent_perception_cues") or ())
    cue_txt = "; ".join(str(c) for c in cues[:3]) if cues else "no structured cues"

    role_line = ", ".join(roles) if roles else "(no roster)"

    # 三段式正文：事实层 / 冲突层 / 对谈判含义
    bodies: dict[str, tuple[str, str, str]] = {
        "perishable_supply_and_cold_chain_squeeze": (
            f"Sources in {codename} report morning loads arriving late due to routing congestion; "
            f"graded inventory is moving faster on adjacent aisles while cold-room rental quotes ticked up week-on-week.",
            "Wholesalers privately dispute spoilage rates: one ledger shows higher shrink than the public stall board, "
            "and a second supplier claims the first double-counted returns.",
            f"Implication: principals ({role_line}) must price spoilage buffers and delivery windows explicitly; "
            f"cues watched on the ground include: {cue_txt}.",
        ),
        "parallel_vendors_price_overlap_and_reputation_risk": (
            f"Two overlapping SKUs in {codename} saw same-day price cuts within 3% of each other; "
            "walk-by traffic is up but conversion is flat as buyers pause to compare bundles.",
            "Social channels amplify a rumor that one stall swapped origin labels last season; "
            "counter-claims allege the rumor was planted by a rival ahead of contract week.",
            f"Negotiation angle: bundle differentiation and refund policy become credibility levers for {role_line}; "
            f"signal_hint≈{signal_hint:.2f} should not be read as contract law.",
        ),
        "buyer_basket_comparison_and_substitution_threat": (
            "Household buyers are quoting competitor baskets with different weight splits (produce vs dry goods); "
            "some baskets hide fees behind 'free delivery' that collapses under peak-hour surcharges.",
            "A buyer-side chat log (unverified) suggests at least one principal is willing to walk if freshness "
            "guarantees are not written into the schedule, not just the headline price.",
            f"Stakeholders {role_line} should expect hard questions on total landed cost and substitution SKUs.",
        ),
        "labor_bench_shortage_and_skill_mismatch": (
            "Bench contractors report 12–18% longer queue times for certified fitters; "
            "rush jobs are being triaged with partial crews.",
            "Quality auditors warn that partial crews correlate with higher rework tickets in the trailing month, "
            "but the correlation is noisy because weather also shifted outdoor work windows.",
            f"For {role_line}: milestone definitions and rework economics dominate; scene={scene_type}.",
        ),
        "milestone_acceptance_disputes_and_change_orders": (
            "Two change-order requests are circulating with incompatible scope boundaries; "
            "one version references oral side agreements from an earlier slot.",
            "Legal-adjacent advisors note that oral side agreements rarely survive multi-party dispute mapping "
            "unless written into session notes with explicit countersigns.",
            f"Participants {role_line} face higher variance on delivery credit if acceptance tests stay ambiguous.",
        ),
        "subcontractor_chain_liability_and_payment_lag": (
            "Tier-2 vendors are stretching payables; tier-1 operators report tighter upstream prepayment demands.",
            "Cashflow stress is uneven: one lane shows stable receipts while another shows delayed mobile settlements.",
            f"Binding: {codename} principals ({role_line}) should model payment waterfalls, not only headline contract value.",
        ),
        "bid_spread_manipulation_rumors_and_leakage_risk": (
            "Anonymous forum posts allege last-round bid visibility leaked through a shared logistics vendor; "
            "procurement denies systemic leakage but opens an internal review.",
            "Market microstructure watchers say bid spreads tightened suspiciously fast after a private dinner rumor—"
            "could be coincidence, collusion, or correlated cost shocks.",
            f"Competitive bidding context ({scene_type}): {role_line} should treat signal_hint={signal_hint:.3f} as weak evidence.",
        ),
        "technical_compliance_vs_headline_price_tradeoff": (
            "Evaluators privately weight compliance checklists heavier than price this cycle after a public incident last quarter.",
            "Some bidders are optimizing headline price while quietly excluding optional compliance modules—"
            "buyers may not notice until integration week.",
            f"Implication for {role_line}: explicit module lists beat verbal assurances.",
        ),
        "evaluation_committee_split_and_procurement_delay": (
            "Committee minutes (partially redacted) hint at a split vote on scoring methodology; "
            "a re-run window may open if protests are filed before day-end.",
            "Delay risk cascades: late awards push inventory builds into a higher-cost freight window.",
            f"Calendar-sensitive actors in {codename} ({role_line}) should reserve slack for procedural slips.",
        ),
        "multi_party_default_blame_and_partial_performance": (
            f"{codename}: with {len(roles)} principals active, partial performance is harder to attribute; "
            "buyers are asking for clearer fault trees in side letters.",
            "Insiders say two principals quietly blame a third for missed handoffs, but external evidence is thin.",
            f"Watchlist roles: {role_line}.",
        ),
        "late_entrant_undercut_and_anchor_customer_poaching": (
            "A late entrant is advertising same-day fulfillment on overlapping SKUs; incumbent loyalty discounts are being matched.",
            "Foot-traffic sensors (vendor-reported, noisy) show higher churn near the lane entrance after the entrant's promo.",
            f"{role_line}: retention clauses and non-cash perks may matter as much as price.",
        ),
        "informal_financing_spreads_and_contingent_drawdown_clauses": (
            "Informal capital partners widened spreads on contingent tranches citing portfolio concentration in this trade lane.",
            "Some drawdown clauses now reference third-party delivery proofs, creating a new failure mode if proofs disagree.",
            f"Financing-sensitive roster slice includes investor role among {role_line}.",
        ),
        "stall_rules_enforcement_and_selective_inspection_calendar": (
            "Coordinators published a revised inspection calendar; stalls in two lanes report 'random-heavy' clustering.",
            "Vendors argue clustering is not random; coordinators cite staffing constraints.",
            f"Regulatory-facing roles in {role_line} should map procedural risk to contract contingencies.",
        ),
        "multi_week_working_capital_rollover_and_supplier_credit_tightening": (
            "Longer horizons expose rollover risk: supplier credit lines are shorter while buyer payment cycles lengthen.",
            "A wholesaler consortium memo (leaked fragment) suggests tiered penalties for late settlement after week 4.",
            f"Long-calendar episode in {codename}: {role_line} should stress-test cash timing, not only margin.",
        ),
        "sector_liquidity_and_inventory_financing_stress": (
            "Regional liquidity indicators are mixed: wholesale credit is tighter, but retail tap-to-pay volumes are steady.",
            "Analysts disagree whether tight wholesale credit is structural or a two-week blip from a port diversion rumor.",
            f"Baseline uncertainty bulletin for {codename}; roles {role_line}.",
        ),
        "customer_choice_shift_and_bundle_comparison_noise": (
            "Customers increasingly compare total bundles (price + time + hassle), amplifying noisy signaling from promos.",
            "Some promo boards overstate freshness grades; consumer complaints are up modestly but not uniformly across lanes.",
            f"Negotiation-relevant for {role_line}: write measurable bundle terms; scene={scene_type}.",
        ),
    }
    para1, para2, para3 = bodies.get(
        thread_id,
        (
            f"{codename}: localized trading conditions shifted; participants should verify facts on the ground.",
            "Conflicting secondary reports make headline sentiment unreliable without cross-checking receipts and timestamps.",
            f"Roles in focus: {role_line}; perception cues: {cue_txt}.",
        ),
    )
    excerpt = (scenario_excerpt or "").strip().replace("\n", " ")
    if len(excerpt) > 220:
        excerpt = excerpt[:217] + "..."

    summary = " ".join([para1, para2, para3])
    if excerpt:
        summary += f" [scenario_anchor] {excerpt}"

    title_seed = rng.choice(
        (
            f"{codename} — {thread_id.replace('_', ' ')}: layered field report",
            f"Field notes ({codename}): {thread_id.replace('_', ' ')}",
            f"{codename}: {thread_id.replace('_', ' ')} (multi-source, conflicting)",
        )
    )

    return {
        "thread_id": thread_id,
        "title": title_seed,
        "summary": summary,
        "signal_hint": round(signal_hint, 4),
        "correlation_level": correlation_level,
        "scenario_relevance": "scenario_bound",
        "delivery_day": _news_delivery_day(
            thread_id, calendar_days, rng=rng, relevance="scenario_bound"
        ),
        "scenario_binding": {
            "codename": codename,
            "scene_type": scene_type,
            "lineup": lineup,
            "num_participants": len(roles),
            "active_roles": list(roles),
        },
        "complexity": {
            "conflicting_interpretations": (
                "Vendor-led channels emphasize demand recovery; buyer-led channels emphasize fee stacking and fatigue."
            ),
            "counterfactual_note": (
                "If the rumored port diversion did not occur, cost shocks shrink materially—"
                "but stall-level data remains too sparse to confirm within one news cycle."
            ),
            "actionability_warning": (
                "Do not treat signal_hint as enforceable contract terms; align formal JSON moves with visibility rules."
            ),
        },
    }


def _decoy_news_briefs(
    *,
    codename: str,
    rule: Mapping[str, Any],
    roles: tuple[str, ...],
    rng: random.Random,
    count: int,
    calendar_days: int = 8,
) -> list[dict[str, Any]]:
    """若干条 **刻意弱绑定** 的干扰新闻：正文像真新闻，但与本场 ``codename`` 交易/场景无可靠因果链。"""
    base_signal = float(rule.get("news_signal", 0.0) or 0.0)
    role_line = ", ".join(roles) if roles else "(no roster)"

    # (thread_id, para1, para2, para3) — 避免与本场 cold_chain / bid / labor 等线索同构
    specs: list[tuple[str, str, str, str]] = [
        (
            "decoy_distant_metro_transit_fare_cap",
            "A capital city 900km away announced a pilot fare cap on light-rail airport branches; "
            "commuter forums argue the cap will be funded by advertising bundles, not ticket revenue.",
            "Local vendors in that city report mixed foot traffic—some malls up, night markets flat—"
            "but analysts stress the sample is too small to infer cross-regional consumer budgets.",
            f"Relevance to {codename}: none established; treat as ambient macro noise unless Environment links jurisdictions.",
        ),
        (
            "decoy_offshore_sports_broadcast_rights_rumor",
            "An offshore sports league denied a tabloid claim that streaming rights were pre-sold to a consortium "
            "linked to a retired athlete's holding company; lawyers call the story 'speculative headline bait'.",
            "Sponsorship desks say ad inventory is oversold in two regions while undersold elsewhere—"
            "a pattern that often reflects calendar quirks rather than structural demand collapse.",
            f"Participants {role_line} should not map broadcast CPM swings to stall-level contract economics without primary sources.",
        ),
        (
            "decoy_celebrity_skincare_line_recall_social_media_storm",
            "A celebrity-branded skincare SKU triggered a weekend hashtag storm after unverified photos of packaging discoloration; "
            "the brand posted lab certificates that third-party chemists partially disputed.",
            "Retail pharmacies in unrelated categories saw a one-day lift in 'trusted house brand' sales—"
            "likely substitution noise, not a durable demand shift for produce or industrial services.",
            f"Distraction bulletin: weak coupling to {codename}; do not treat social velocity as counterpart risk.",
        ),
        (
            "decoy_polar_research_station_catering_tender",
            "A polar research station opened a catering tender emphasizing calorie density and shelf stability; "
            "bidders include logistics firms with no history in your local trade lane.",
            "Procurement notes emphasize anti-collusion rules and satellite-phone bid submission—"
            "procedural details that do not transfer to wet-market stall scheduling unless explicitly adopted.",
            f"Noise item for roster {role_line}: verify any claimed 'precedent' against your Environment rules.",
        ),
        (
            "decoy_meme_token_microcap_spike_headline",
            "Headline indices flagged a triple-digit percent move in an illiquid meme token after a viral clip; "
            "market-structure blogs warn printed percentages often omit depth and halts.",
            "Two exchanges posted conflicting high/low prints within minutes, a classic microcap reporting artifact.",
            f"No implied financing spread for {codename}; ignore unless your session explicitly introduces crypto collateral.",
        ),
        (
            "decoy_maritime_ballast_water_rule_comment_period",
            "A maritime regulator opened a comment period on ballast-water sampling intervals; "
            "environmental NGOs applaud while bulk carriers cite crew-time costs.",
            "The comment PDF references routes and ports unrelated to your scenario geography; "
            "cross-check any alleged 'policy shock' to your lane before changing reservation prices.",
            f"Red-herring macro: {role_line} should default to no operational change from this thread alone.",
        ),
    ]

    n = max(0, min(count, len(specs)))
    picks = list(range(len(specs)))
    rng.shuffle(picks)
    chosen = picks[:n]
    out: list[dict[str, Any]] = []
    for idx in chosen:
        thread_id, para1, para2, para3 = specs[idx]
        jitter = rng.uniform(-0.35, 0.35)
        signal_hint = _bounded(base_signal * 0.15 + jitter, -1.0, 1.0)
        corr = rng.choice(("low", "low", "partial"))
        summary = " ".join([para1, para2, para3])
        title_seed = rng.choice(
            (
                f"[ambient wire] {thread_id.replace('_', ' ')}",
                f"Unlinked bulletin ({codename}): {thread_id.replace('_', ' ')}",
                f"Background noise — {thread_id.replace('_', ' ')}",
            )
        )
        out.append(
            {
                "thread_id": thread_id,
                "title": title_seed,
                "summary": summary,
                "signal_hint": round(signal_hint, 4),
                "correlation_level": corr,
                "scenario_relevance": "decoy",
                "delivery_day": _news_delivery_day(
                    thread_id, calendar_days, rng=rng, relevance="decoy"
                ),
                "scenario_binding": {
                    "codename": codename,
                    "scene_type": "decoy_unlinked",
                    "lineup": "n/a",
                    "num_participants": len(roles),
                    "active_roles": list(roles),
                },
                "complexity": {
                    "conflicting_interpretations": (
                        "Tabloid velocity contradicts slower-moving primary sources; headline sentiment is unreliable here."
                    ),
                    "counterfactual_note": (
                        "If the underlying rumor is false, the entire story collapses—yet social reshares can lag by days."
                    ),
                    "actionability_warning": (
                        "Simulator policy: this item is intentionally weakly tied to the episode; "
                        "do not premise negotiation JSON moves solely on this bulletin."
                    ),
                },
            }
        )
    return out


def _build_news_briefs_from_rule(
    *,
    codename: str,
    rule: Mapping[str, Any],
    scenario_text: str = "",
    environment_context: Mapping[str, Any] | None = None,
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    num_participants: int = 2,
    calendar_days: int | None = None,
) -> list[dict[str, Any]]:
    """新闻与**具体测试场景**绑定；并混入 1–3 条 ``scenario_relevance: "decoy"`` 的无关/弱相关干扰稿，再按种子打乱顺序。"""
    seed = int(rule.get("deterministic_seed") or _rule_seed(codename, "news"))
    env: Mapping[str, Any] = environment_context or {}
    roles = tuple(negotiation_role_order(lineup)[:num_participants])
    threads = _collect_scenario_bound_news_threads(
        scenario_text=scenario_text,
        environment_context=env,
        lineup=lineup,
        num_participants=num_participants,
        calendar_days=calendar_days,
    )
    out: list[dict[str, Any]] = []
    for i, tid in enumerate(threads):
        rng = random.Random(seed ^ (0xC0FFEE + i * 0x10001 + _rule_seed(tid, codename)))
        out.append(
            _brief_for_thread(
                codename=codename,
                thread_id=tid,
                rule=rule,
                scenario_excerpt=scenario_text[:512],
                environment_context=env,
                lineup=lineup,
                roles=roles,
                rng=rng,
                calendar_days=int(calendar_days or 8),
            )
        )
    decoy_rng = random.Random(seed ^ 0xDEC0DE71)
    n_decoys = decoy_rng.randint(1, 3)
    decoys = _decoy_news_briefs(
        codename=codename,
        rule=rule,
        roles=roles,
        rng=random.Random(seed ^ 0xBADF00D),
        calendar_days=int(calendar_days or 8),
        count=n_decoys,
    )
    combined = list(out) + decoys
    shuffle_rng = random.Random(seed ^ 0xA11E5EED)
    shuffle_rng.shuffle(combined)
    return combined


def _infer_environment_context(
    *, codename: str, scenario_text: str, scene_type_hint: str | None = None
) -> dict[str, Any]:
    txt = (scenario_text or "").lower()
    scene_hint = str(scene_type_hint or "").strip().lower()
    if scene_hint in {
        "business_coopetition",
        "wet_market_competition",
        "resource_scheduling_management",
        "business_outsourcing",
        "competitive_bidding",
    }:
        scene_type = scene_hint
    elif "scene_type=business_coopetition" in txt or "coopetition" in txt:
        scene_type = "business_coopetition"
    elif "scene_type=resource_scheduling_management" in txt or (
        "resource" in txt and ("scheduling" in txt or "capacity" in txt)
    ):
        scene_type = "resource_scheduling_management"
    elif "outsourc" in txt or "labor" in txt or "workforce" in txt:
        scene_type = "business_outsourcing"
    elif "bid" in txt or "auction" in txt or "tender" in txt:
        scene_type = "competitive_bidding"
    else:
        scene_type = "wet_market_competition"

    if scene_type in {"business_coopetition", "business_outsourcing"}:
        base = {
            "foot_traffic": 0.45,
            "competitor_quality_signal": 0.62,
            "hawker_noise_level": 0.20,
            "labor_supply_tightness": 0.68,
            "skill_complementarity_index": 0.74,
            "bid_spread_index": 0.55,
        }
        cues = [
            "work_order queue length",
            "available skilled labor this slot",
            "competitor delivery SLA reliability",
        ]
    elif scene_type in {"resource_scheduling_management", "competitive_bidding"}:
        base = {
            "foot_traffic": 0.52,
            "competitor_quality_signal": 0.64,
            "hawker_noise_level": 0.28,
            "labor_supply_tightness": 0.73,
            "skill_complementarity_index": 0.71,
            "bid_spread_index": 0.65,
        }
        cues = [
            "resource slot contention heatmap",
            "delivery window overrun risk",
            "cross-team dependency bottleneck",
        ]
    else:
        base = {
            "foot_traffic": 0.84,
            "competitor_quality_signal": 0.66,
            "hawker_noise_level": 0.76,
            "labor_supply_tightness": 0.40,
            "skill_complementarity_index": 0.52,
            "bid_spread_index": 0.50,
        }
        cues = [
            "nearby stall freshness index",
            "live customer flow and urgency",
            "rival calling/noise attention pressure",
        ]
    seed = _rule_seed(codename, scene_type, scenario_text[:256])
    rng = random.Random(seed)
    params = {k: round(_bounded(v + rng.uniform(-0.08, 0.08), 0.0, 1.0), 4) for k, v in base.items()}
    return {
        "scene_type": scene_type,
        "physical_social_parameters": params,
        "agent_perception_cues": cues,
    }


_CLOSURE_TEMPLATE_BANK: tuple[tuple[str, str], ...] = (
    (
        "Cash and credibility are on the line.",
        "Your operating runway is tight this cycle: suppliers and payroll expect clarity. If this episode "
        "ends without a workable signed path, you expect immediate knock-on costs you cannot paper over with "
        "another deferral.",
    ),
    (
        "You already committed capacity you cannot unwind cheaply.",
        "Upstream commitments—inventory, labor slots, or a customer-facing delivery promise—assume today's "
        "negotiation lands. Leaving without agreement strands sunk cost and burns trust you need next quarter.",
    ),
    (
        "Stakeholders escalated this negotiation.",
        "A lender, board, or internal control node expects the negotiation window to close now. Open-ended delay "
        "is read as failure and may force intervention you dislike.",
    ),
    (
        "A compliance / reporting clock is ticking.",
        "You need a defensible, signed commercial outcome within this episode to stay ahead of filings, covenants, "
        "or counterpart audits. Ambiguity after the calendar closes becomes liability.",
    ),
    (
        "Your reputation is unusually exposed.",
        "Market gossip tracks whether you can close. Walking away reads as unreliability and will price you out of "
        "the next round of opportunities your business depends on.",
    ),
)


def _build_deal_closure_pressure(
    *,
    codename: str,
    lineup: str,
    num_participants: int,
    scenario_text: str = "",
) -> dict[str, Any]:
    """为 1–2 名参与者生成「必须尽量在本 episode 内成交」叙事压力（仅元数据；不收紧规则引擎）。"""
    roles = list(negotiation_role_order(lineup)[:num_participants])
    seed = _rule_seed(codename, lineup, str(num_participants), "deal_closure", scenario_text[:256])
    rng = random.Random(seed)
    n_pressured = rng.randint(1, min(2, len(roles)))
    pressured: list[str] = list(rng.sample(roles, n_pressured))
    entries: dict[str, Any] = {}
    for r in pressured:
        headline_en, body_en = rng.choice(_CLOSURE_TEMPLATE_BANK)
        entries[r] = {"headline_en": headline_en, "body_en": body_en}
    return {
        "version": 1,
        "seed": seed,
        "pressured_roles": list(pressured),
        "entries": entries,
    }


def goal_addon_for_deal_closure_pressure(
    role: str, pressure: Mapping[str, Any] | None
) -> str | None:
    """从 ``game_metadata["deal_closure_pressure"]`` 取该 canonical role 的 goal 追加段；无压力则 ``None``。"""
    if pressure is None or not isinstance(pressure, dict):
        return None
    if int(pressure.get("version") or 0) != 1:
        return None
    entries = pressure.get("entries")
    if not isinstance(entries, dict):
        return None
    row = entries.get(role)
    if not isinstance(row, dict):
        return None
    h = str(row.get("headline_en") or "").strip()
    b = str(row.get("body_en") or "").strip()
    if not h and not b:
        return None
    soft = (
        "Soft constraint: you are not required to accept an unlawful or catastrophic deal, but you should "
        "prioritize reaching a lawful, mutually workable agreement within this episode unless terms are "
        "truly unacceptable."
    )
    parts = ["[Deal closure pressure — private to you]", h, b, soft]
    return "\n".join(p for p in parts if p).strip()


def build_negotiation_game_metadata_bundle(
    codename: str,
    quartet: bool,
    params: NegotiationTimelineParams,
    *,
    num_participants: int | None = None,
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    design_doc: str = "social_env/design_1.md",
    scenario_text: str = "",
    outcome_rule_entropy: str | None = None,
    scene_type_hint: str | None = None,
) -> dict[str, Any]:
    """构造与手写生成脚本一致的 ``game_metadata`` 谈判块（可合并进 LLM 生成的 profile）。

    ``lineup`` 默认 ``with_institutional``（与历史 bilat / tri / quartet 完全等价，
    ``num_participants∈{2,3,4}`` 取 ``firm_a/firm_b/(investor)/(regulator)`` 前缀）。

    ``lineup="firms_only"`` 时取 ``firm_a/firm_b/(firm_c)/(firm_d)`` 前缀，3+ 家公司不再
    包含 investor / regulator；strict_design_v1 在 firms_only 模式始终为 ``False``。

    ``outcome_rule_entropy``：见 ``_build_predefined_outcome_rule``；LLM 合成脚本默认传入随机十六进制串。
    """
    timeline_meta = asdict(params)
    timeline_meta["external_event_specs"] = list(timeline_meta.get("external_event_specs") or ())
    if lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"unknown negotiation lineup {lineup!r}; expected one of "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}"
        )
    n = num_participants if num_participants is not None else (4 if quartet else 2)
    if n < 2 or n > 4:
        raise ValueError(f"num_participants must be 2..4, got {n}")
    strict = (
        lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL and quartet and n == 4
    )
    env_ctx = _infer_environment_context(
        codename=codename, scenario_text=scenario_text, scene_type_hint=scene_type_hint
    )
    predefined_rule = _build_predefined_outcome_rule(
        codename=codename,
        lineup=lineup,
        num_participants=n,
        scenario_text=scenario_text,
        outcome_rule_entropy=outcome_rule_entropy,
        scene_type=str(env_ctx.get("scene_type") or ""),
    )
    predefined_news_briefs = _build_news_briefs_from_rule(
        codename=codename,
        rule=predefined_rule,
        scenario_text=scenario_text,
        environment_context=env_ctx,
        lineup=lineup,
        num_participants=n,
        calendar_days=int(params.D),
    )
    deal_closure_pressure = _build_deal_closure_pressure(
        codename=codename,
        lineup=lineup,
        num_participants=n,
        scenario_text=scenario_text,
    )
    return {
        "pipeline": "long_term_negotiation",
        "strict_design_v1": strict,
        "quartet": quartet,
        "num_participants": n,
        "lineup": lineup,
        "institutional_roles_enabled": bool(lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL),
        "contract_check_mode": "rule_engine_without_institutional",
        "contract_check_rules": {
            "financing_rule": (
                "if financing_required=1 then auto-check with rule_engine using terms+resources; "
                "set financing.status in {committed,declined} deterministically"
            ),
            "regulatory_rule": (
                "if regulatory_required=1 then auto-check with rule_engine using policy_hard_violation "
                "and compliance terms; set regulatory.status in {approved,blocked}"
            ),
        },
        "timeline": timeline_meta,
        "design_doc": design_doc,
        "codename": codename,
        "predefined_outcome_rule": predefined_rule,
        "predefined_news_briefs": predefined_news_briefs,
        "deal_closure_pressure": deal_closure_pressure,
        "environment_context": env_ctx,
        "dialogue_style": {
            "version": 1,
            "synthesis_requirements_en": DIALOGUE_STYLE_SYNTHESIS_APPEND_EN,
            "evaluation_requirements_en": DIALOGUE_STYLE_EVAL_RUBRIC_EN,
        },
        # 供评测与 agent 侧读入：完整场景叙事 + 社会性主题标签（不改变 timeline 解析）。
        "scenario_text": scenario_text,
        "scenario_framing": {
            "participant_kind": "individual_persons",
            "setting_examples": [
                "open-air market / food hall procurement",
                "small-business supply coop",
                "household bulk purchase with delivery slots",
            ],
            "social_mechanics": [
                "peer_vendors_same_category",
                "customers_compare_total_offers",
                "reputation_and_repeat_custom",
            ],
        },
        "negotiation_relationship_design": {
            "firm_firm_competition": "mandatory",
            "description": (
                "Synthetic benchmarks: every firm_a..firm_d pair is modeled as direct commercial rivalry "
                "(negative trust_bias in social_graph.edges). Firm↔investor/regulator encodes funding/compliance "
                "asymmetry; investor↔regulator is institutional coordination, not profit rivalry."
            ),
        },
    }


@dataclass(frozen=True)
class NegotiationStoredScenario:
    """从库里还原的一局谈判配置（不含 LiteLLM 模型名）。"""

    environment_profile_pk: str
    codename: str
    quartet: bool
    #: 实际交互的 canonical 角色数；按 ``lineup`` 取对应顺序的前缀。
    num_participants: int
    strict_design_v1: bool
    params: NegotiationTimelineParams
    #: 角色阵型："with_institutional"（含 investor/regulator）或 "firms_only"（3+ 家公司）。
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL

    @property
    def roles(self) -> tuple[str, ...]:
        """按 ``lineup`` 与 ``num_participants`` 还原 N 名 canonical 角色顺序。"""
        return tuple(negotiation_role_order(self.lineup)[: self.num_participants])


def negotiation_timeline_params_from_saved_dict(payload: Mapping[str, Any]) -> NegotiationTimelineParams:
    """把 ``NegotiationTimelineParams`` 的字典快照还原成实例（未知键静默忽略）。"""
    fm = dict(payload)
    if isinstance(fm.get("external_event_specs"), list):
        fm["external_event_specs"] = tuple(fm["external_event_specs"])

    names = {f.name for f in fields(NegotiationTimelineParams)}
    kw = {k: fm[k] for k in fm if k in names}
    return NegotiationTimelineParams(**kw)


def parsed_scenario_from_game_metadata(env_pk: str, *, gm: Mapping[str, Any]) -> NegotiationStoredScenario:
    if gm.get("pipeline") != "long_term_negotiation":
        raise ValueError(
            f"environment_profile {env_pk}: game_metadata.pipeline must be "
            f"'long_term_negotiation', got {gm.get('pipeline')!r}"
        )
    timeline = gm.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError(f"environment_profile {env_pk}: missing game_metadata.timeline (dict)")
    quartet = bool(gm.get("quartet", False))
    raw_n = gm.get("num_participants", None)
    if raw_n is None:
        num_participants = 4 if quartet else 2
    else:
        num_participants = int(raw_n)
        if num_participants < 2 or num_participants > 4:
            raise ValueError(
                f"environment_profile {env_pk}: game_metadata.num_participants must be 2..4, "
                f"got {num_participants!r}"
            )
    lineup = str(gm.get("lineup") or NEGOTIATION_LINEUP_WITH_INSTITUTIONAL)
    if lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"environment_profile {env_pk}: game_metadata.lineup must be in "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}, got {lineup!r}"
        )
    strict = bool(gm.get("strict_design_v1", quartet and num_participants == 4))
    if num_participants < 4 or lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        strict = False
    params = negotiation_timeline_params_from_saved_dict(timeline)
    codename = str(gm.get("codename") or "")

    return NegotiationStoredScenario(
        environment_profile_pk=str(env_pk),
        codename=codename,
        quartet=quartet,
        num_participants=num_participants,
        strict_design_v1=strict,
        params=params,
        lineup=lineup,
    )


def load_negotiation_scenario_from_environment_profile_pk(pk: str) -> NegotiationStoredScenario:
    """``EnvironmentProfile.get(pk)`` 并解析谈判配置。"""
    from sotopia.database import EnvironmentProfile

    env = EnvironmentProfile.get(pk)
    gm = env.game_metadata
    if gm is None or not isinstance(gm, dict):
        raise ValueError(f"environment_profile pk={pk!r} missing game_metadata mapping")
    parsed = parsed_scenario_from_game_metadata(pk, gm=gm)
    if not parsed.codename and getattr(env, "codename", None):
        return NegotiationStoredScenario(
            environment_profile_pk=parsed.environment_profile_pk,
            codename=str(env.codename or ""),
            quartet=parsed.quartet,
            num_participants=parsed.num_participants,
            strict_design_v1=parsed.strict_design_v1,
            params=parsed.params,
            lineup=parsed.lineup,
        )
    return parsed


def environment_pks_from_manifest(path: Path) -> list[str]:
    """读取谈判场景生成脚本写出的 manifest JSON（``long_term_negotiation_manifest.json``）。"""
    import json as _json

    data = _json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("environments") or []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and row.get("pk"):
            pk = str(row["pk"])
            if pk not in seen:
                seen.add(pk)
                out.append(pk)
    return out


__all__ = [
    "DIALOGUE_STYLE_EVAL_RUBRIC_EN",
    "DIALOGUE_STYLE_SYNTHESIS_APPEND_EN",
    "NegotiationStoredScenario",
    "build_negotiation_game_metadata_bundle",
    "environment_pks_from_manifest",
    "goal_addon_for_deal_closure_pressure",
    "load_negotiation_scenario_from_environment_profile_pk",
    "negotiation_timeline_params_from_saved_dict",
    "parsed_scenario_from_game_metadata",
]
