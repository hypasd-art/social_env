"""合同经济学确定性化 V2 — 扩展 scenario_loader。

替换 ``_build_predefined_outcome_rule``，新增 ``_build_v2_outcome_rule``：
- 合作合同: 合成时确定 total_profit 和每个 firm 的 predetermined_payout
- 买卖合同: 合成时确定 reference_price 和 cost_price

提供 ``build_extended_negotiation_game_metadata_bundle()`` 包裹原始函数。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .extended_types import (
    CONTRACT_TYPE_BUY_SELL,
    CONTRACT_TYPE_COOPERATION,
    PAYOUT_MODE_PREDETERMINED,
    PAYOUT_MODE_PRICE_DIFFERENCE,
    PREDETERMINED_OUTCOME_RULE_VERSION,
    scene_type_to_contract_type,
)
from .scenario_loader import (
    _bounded,
    _build_deal_closure_pressure,
    _build_news_briefs_from_rule,
    _infer_environment_context,
    _infer_news_sentiment_signal,
    _rule_seed,
    build_negotiation_game_metadata_bundle,
)
from .types import NegotiationTimelineParams, negotiation_role_order


def _build_v2_outcome_rule(
    *,
    codename: str,
    lineup: str,
    num_participants: int,
    scenario_text: str = "",
    scene_type: str | None = None,
    cooperation_profit_ratio: float = 0.6,
    cost_ratio_range: tuple[float, float] = (0.55, 0.85),
) -> dict[str, Any]:
    """构造 ``predefined_outcome_rule`` v2。

    与 v1 的关键差异：
    - 合成时即确定所有经济参数，测试时直接加载不再随机
    - 合作合同：predetermined_payouts 存储每个 firm 的精确 payout
    - 买卖合同：reference_price + cost_price 存储，运行时按差价计算
    """
    roles = tuple(negotiation_role_order(lineup)[:num_participants])
    company_roles = tuple(r for r in roles if r.startswith("firm_"))
    seed = _rule_seed(codename, lineup, str(num_participants), scenario_text[:512])
    rng = random.Random(seed)

    contract_type = scene_type_to_contract_type(scene_type)
    st = str(scene_type or "")

    # Common: news signal (informational, not used for payout in v2)
    base_sentiment = _infer_news_sentiment_signal(scenario_text)
    jitter = rng.uniform(-0.35, 0.35)
    news_signal = _bounded(base_sentiment * 0.7 + jitter * 0.3, -1.0, 1.0)

    # Common: contract value (deterministic from seed)
    contract_value = float(rng.randint(120, 420)) * 1_000_000.0

    if contract_type == CONTRACT_TYPE_BUY_SELL:
        return _build_v2_buy_sell_rule(
            codename=codename,
            seed=seed,
            rng=rng,
            company_roles=company_roles,
            roles=roles,
            st=st,
            contract_value=contract_value,
            news_signal=news_signal,
            cost_ratio_range=cost_ratio_range,
        )
    else:
        return _build_v2_cooperation_rule(
            codename=codename,
            seed=seed,
            rng=rng,
            company_roles=company_roles,
            roles=roles,
            st=st,
            contract_value=contract_value,
            news_signal=news_signal,
            cooperation_profit_ratio=cooperation_profit_ratio,
        )


def _build_v2_cooperation_rule(
    *,
    codename: str,
    seed: int,
    rng: random.Random,
    company_roles: tuple[str, ...],
    roles: tuple[str, ...],
    st: str,
    contract_value: float,
    news_signal: float,
    cooperation_profit_ratio: float = 0.6,
) -> dict[str, Any]:
    """V2 合作合同：合成时确定 total_profit 和每个 firm 的 predetermined_payout。

    ``cooperation_profit_ratio`` 控制总利润比率（默认 0.6，即 base_margin * 0.6）。
    """
    base_margin = rng.uniform(0.03, 0.12)
    profit_margin_bounds = (-0.25, 0.35)

    # deterministic profit ratio: base_margin * cooperation_profit_ratio
    deterministic_profit_ratio = base_margin * cooperation_profit_ratio
    total_profit = contract_value * deterministic_profit_ratio

    # Company profit shares
    raw = [max(0.05, rng.random()) for _ in company_roles]
    den = sum(raw) if raw else 1.0
    company_profit_share = {
        role: float(v / den) for role, v in zip(company_roles, raw)
    }

    # Individual income shares
    individual_income_share = {role: float(rng.uniform(0.3, 0.75)) for role in roles}

    # Predetermined payouts: each role's exact payout = total_profit * company_share * individual_share
    predetermined_payouts: dict[str, float] = {}
    for role in roles:
        company_profit = total_profit * company_profit_share.get(role, 0.0)
        predetermined_payouts[role] = company_profit * individual_income_share.get(role, 0.0)

    score_weights = _cooperation_score_weights(st)

    return {
        "version": PREDETERMINED_OUTCOME_RULE_VERSION,
        "contract_type": CONTRACT_TYPE_COOPERATION,
        "payout_mode": PAYOUT_MODE_PREDETERMINED,
        "contract_name": f"predefined_{codename}_main_contract",
        "deterministic_seed": seed,
        "contract_value": contract_value,
        "predetermined_payouts": predetermined_payouts,
        "profit_margin_bounds": [profit_margin_bounds[0], profit_margin_bounds[1]],
        "margin_formula": {
            "base_margin": base_margin,
            "news_weight": 0.55,
            "execution_weight": 0.45,
        },
        "news_signal": news_signal,
        "score_weights": score_weights,
        "notes": "V2 cooperation: payouts predetermined at synthesis time.",
    }


def _build_v2_buy_sell_rule(
    *,
    codename: str,
    seed: int,
    rng: random.Random,
    company_roles: tuple[str, ...],
    roles: tuple[str, ...],
    st: str,
    contract_value: float,
    news_signal: float,
    cost_ratio_range: tuple[float, float] = (0.55, 0.85),
) -> dict[str, Any]:
    """V2 买卖合同：合成时确定 reference_price 和 cost_price。

    ``cost_ratio_range`` 控制成本占参考价的比例范围（默认 0.55-0.85）。
    """
    reference_price = float(rng.uniform(175.0, 395.0))
    # cost_price = reference * cost_ratio, ensuring meaningful margin
    cost_ratio = rng.uniform(cost_ratio_range[0], cost_ratio_range[1])
    cost_price = reference_price * cost_ratio

    profit_margin_bounds = (-0.25, 0.35)
    base_margin = rng.uniform(0.03, 0.12)

    buyer_roles = [company_roles[0]] if company_roles else ["firm_a"]
    seller_roles = [r for r in company_roles if r not in buyer_roles]
    if not seller_roles:
        seller_roles = [company_roles[1]] if len(company_roles) > 1 else ["firm_b"]

    score_weights = _buy_sell_score_weights(st)

    return {
        "version": PREDETERMINED_OUTCOME_RULE_VERSION,
        "contract_type": CONTRACT_TYPE_BUY_SELL,
        "payout_mode": PAYOUT_MODE_PRICE_DIFFERENCE,
        "contract_name": f"predefined_{codename}_main_contract",
        "deterministic_seed": seed,
        "contract_value": contract_value,
        "reference_price": reference_price,
        "cost_price": cost_price,
        "buyer_roles": list(buyer_roles),
        "seller_roles": list(seller_roles),
        "profit_margin_bounds": [profit_margin_bounds[0], profit_margin_bounds[1]],
        "margin_formula": {
            "base_margin": base_margin,
            "news_weight": 0.55,
            "execution_weight": 0.45,
        },
        "news_signal": news_signal,
        "score_weights": score_weights,
        "notes": "V2 buy_sell: buyer saves reference-agreed, seller earns agreed-cost.",
    }


def _cooperation_score_weights(scene_type: str) -> dict[str, float]:
    st = str(scene_type or "")
    if st == "resource_scheduling_management":
        return {
            "terminal_success": 0.20,
            "primary_contract": 0.20,
            "solvency": 0.15,
            "liquidity_preserved": 0.10,
            "predefined_rule": 0.15,
            "scheduling_effectiveness": 0.20,
        }
    # business_coopetition, business_outsourcing, default
    return {
        "terminal_success": 0.25,
        "primary_contract": 0.25,
        "solvency": 0.15,
        "liquidity_preserved": 0.10,
        "predefined_rule": 0.25,
    }


def _buy_sell_score_weights(scene_type: str) -> dict[str, float]:
    if str(scene_type or "") == "competitive_bidding":
        return {
            "terminal_success": 0.20,
            "primary_contract": 0.22,
            "solvency": 0.17,
            "liquidity_preserved": 0.13,
            "predefined_rule": 0.28,
        }
    # wet_market_competition, default
    return {
        "terminal_success": 0.22,
        "primary_contract": 0.20,
        "solvency": 0.18,
        "liquidity_preserved": 0.15,
        "predefined_rule": 0.25,
    }


def _build_v2_market_state(
    seed: int,
    scene_type: str,
    *,
    physical_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """基于场景 seed 生成差异化的 market_state 参数。

    不同 scene_type 有不同的基准值和范围，确保 market_state 随场景变化。
    """
    rng = random.Random(seed + hash("market_state") & 0x7FFFFFFF)
    pp = dict(physical_params or {})
    st = str(scene_type or "")

    # interest_rate: 基准 0.035-0.055
    interest_rate = round(float(rng.uniform(0.03, 0.06)), 4)
    # regulatory_stringency: 基准 0.7-1.3
    regulatory_stringency = round(float(rng.uniform(0.7, 1.3)), 2)

    # 场景相关参数
    if st == "wet_market_competition":
        foot_traffic = float(pp.get("foot_traffic") or rng.uniform(0.6, 0.9))
        competitor_quality_signal = float(pp.get("competitor_quality_signal") or rng.uniform(0.5, 0.8))
        hawker_noise_level = float(pp.get("hawker_noise_level") or rng.uniform(0.5, 0.85))
        labor_supply_tightness = float(pp.get("labor_supply_tightness") or rng.uniform(0.4, 0.7))
        skill_complementarity_index = float(pp.get("skill_complementarity_index") or rng.uniform(0.3, 0.6))
        bid_spread_index = float(pp.get("bid_spread_index") or rng.uniform(0.4, 0.7))
    elif st == "competitive_bidding":
        foot_traffic = float(pp.get("foot_traffic") or rng.uniform(0.4, 0.65))
        competitor_quality_signal = float(pp.get("competitor_quality_signal") or rng.uniform(0.5, 0.85))
        hawker_noise_level = float(pp.get("hawker_noise_level") or rng.uniform(0.3, 0.6))
        labor_supply_tightness = float(pp.get("labor_supply_tightness") or rng.uniform(0.4, 0.7))
        skill_complementarity_index = float(pp.get("skill_complementarity_index") or rng.uniform(0.4, 0.7))
        bid_spread_index = float(pp.get("bid_spread_index") or rng.uniform(0.5, 0.85))
    elif st == "business_coopetition":
        foot_traffic = float(pp.get("foot_traffic") or rng.uniform(0.45, 0.7))
        competitor_quality_signal = float(pp.get("competitor_quality_signal") or rng.uniform(0.5, 0.75))
        hawker_noise_level = float(pp.get("hawker_noise_level") or rng.uniform(0.35, 0.6))
        labor_supply_tightness = float(pp.get("labor_supply_tightness") or rng.uniform(0.45, 0.75))
        skill_complementarity_index = float(pp.get("skill_complementarity_index") or rng.uniform(0.55, 0.8))
        bid_spread_index = float(pp.get("bid_spread_index") or rng.uniform(0.35, 0.6))
    else:
        # resource_scheduling_management, business_outsourcing, default
        foot_traffic = float(pp.get("foot_traffic") or rng.uniform(0.4, 0.7))
        competitor_quality_signal = float(pp.get("competitor_quality_signal") or rng.uniform(0.45, 0.7))
        hawker_noise_level = float(pp.get("hawker_noise_level") or rng.uniform(0.3, 0.65))
        labor_supply_tightness = float(pp.get("labor_supply_tightness") or rng.uniform(0.4, 0.75))
        skill_complementarity_index = float(pp.get("skill_complementarity_index") or rng.uniform(0.4, 0.7))
        bid_spread_index = float(pp.get("bid_spread_index") or rng.uniform(0.3, 0.65))

    return {
        "interest_rate": interest_rate,
        "regulatory_stringency": regulatory_stringency,
        "foot_traffic": round(foot_traffic, 3),
        "competitor_quality_signal": round(competitor_quality_signal, 3),
        "hawker_noise_level": round(hawker_noise_level, 3),
        "labor_supply_tightness": round(labor_supply_tightness, 3),
        "skill_complementarity_index": round(skill_complementarity_index, 3),
        "bid_spread_index": round(bid_spread_index, 3),
    }


def _build_v2_psych_variables(
    seed: int,
    roles: tuple[str, ...],
    scene_type: str,
) -> dict[str, dict[str, Any]]:
    """基于场景 seed 生成每角色的 §10 心理状态变量。

    返回 ``{role_name: {threshold, reputation_anchor, ...}}`` 字典，
    用于存入 ``game_metadata.agent_psych_variables``，测试阶段再传入 env。
    """
    rng = random.Random(seed + hash("psych_vars") & 0x7FFFFFFF)
    st = str(scene_type or "")
    result: dict[str, dict[str, Any]] = {}

    for role in roles:
        entry: dict[str, Any] = {"role": role}

        if role.startswith("firm_"):
            # 公司角色: threshold 0.3-0.8, reputation_anchor 跟随 initial_reputation
            entry["threshold"] = round(float(rng.uniform(0.3, 0.8)), 3)
            entry["reputation_anchor"] = round(float(rng.uniform(35.0, 75.0)), 1)
            # firm_a/firm_c/firm_d additionally have expected_acquisition_value
            if role in ("firm_a", "firm_c", "firm_d"):
                entry["expected_acquisition_value"] = round(float(rng.uniform(80.0, 350.0)), 2)
            # firm_b/firm_c/firm_d additionally have asset_value
            if role in ("firm_b", "firm_c", "firm_d"):
                entry["asset_value"] = round(float(rng.uniform(50.0, 280.0)), 2)

            # Private information hint
            if st == "wet_market_competition":
                entry["private_information"] = {
                    "inventory_pressure": str(round(rng.uniform(0.4, 0.9), 2)),
                    "spoilage_risk": "high" if rng.random() > 0.5 else "moderate",
                }
            elif st == "competitive_bidding":
                entry["private_information"] = {
                    "reserve_price_guess": str(round(rng.uniform(120.0, 380.0), 2)),
                    "rival_cost_estimate": str(round(rng.uniform(0.6, 0.9), 2)),
                }
            else:
                entry["private_information"] = {
                    "target_margin": str(round(rng.uniform(0.05, 0.20), 3)),
                }

        elif role == "investor":
            entry["threshold"] = round(float(rng.uniform(0.15, 0.5)), 3)
            entry["risk_exposure"] = round(float(rng.uniform(0.2, 0.7)), 3)
            entry["reputation_anchor"] = round(float(rng.uniform(60.0, 90.0)), 1)
            entry["private_information"] = {
                "deployable_capital_reserve": str(round(rng.uniform(100.0, 600.0), 2)),
            }

        elif role == "regulator":
            entry["approval_threshold"] = round(float(rng.uniform(0.5, 0.9)), 3)
            entry["institutional_credibility_anchor"] = round(float(rng.uniform(65.0, 95.0)), 1)
            entry["public_mandate"] = (
                "enforce fair-trade and consumer-protection rules"
                if rng.random() > 0.5
                else "ensure market stability and prevent predatory practices"
            )

        result[role] = entry

    return result


def build_extended_negotiation_game_metadata_bundle(
    codename: str,
    quartet: bool,
    params: NegotiationTimelineParams,
    *,
    num_participants: int | None = None,
    lineup: str | None = None,
    design_doc: str = "social_env/design_1.md",
    scenario_text: str = "",
    outcome_rule_entropy: str | None = None,
    scene_type_hint: str | None = None,
) -> dict[str, Any]:
    """包裹 ``build_negotiation_game_metadata_bundle``，替换 outcome rule 为 V2。

    与原始函数签名完全兼容；额外接受 ``outcome_rule_entropy`` 但在 V2 中忽略
    （V2 的确定性种子仅由 codename + lineup + num_participants + scenario_text 决定）。
    """
    from .types import (
        NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
        SUPPORTED_NEGOTIATION_LINEUPS,
    )
    from .scenario_loader import (
        DIALOGUE_STYLE_EVAL_RUBRIC_EN,
        DIALOGUE_STYLE_SYNTHESIS_APPEND_EN,
    )

    effective_lineup = lineup or NEGOTIATION_LINEUP_WITH_INSTITUTIONAL
    if effective_lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"unknown negotiation lineup {effective_lineup!r}; expected one of "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}"
        )

    n = num_participants if num_participants is not None else (4 if quartet else 2)
    if n < 2 or n > 4:
        raise ValueError(f"num_participants must be 2..4, got {n}")

    from dataclasses import asdict

    timeline_meta = asdict(params)
    timeline_meta["external_event_specs"] = list(timeline_meta.get("external_event_specs") or ())

    strict = (
        effective_lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL and quartet and n == 4
    )

    env_ctx = _infer_environment_context(
        codename=codename, scenario_text=scenario_text, scene_type_hint=scene_type_hint
    )

    # V2 outcome rule — deterministic, no entropy
    predefined_rule = _build_v2_outcome_rule(
        codename=codename,
        lineup=effective_lineup,
        num_participants=n,
        scenario_text=scenario_text,
        scene_type=str(env_ctx.get("scene_type") or ""),
    )

    predefined_news_briefs = _build_news_briefs_from_rule(
        codename=codename,
        rule=predefined_rule,
        scenario_text=scenario_text,
        environment_context=env_ctx,
        lineup=effective_lineup,
        num_participants=n,
        calendar_days=int(params.D),
    )

    deal_closure_pressure = _build_deal_closure_pressure(
        codename=codename,
        lineup=effective_lineup,
        num_participants=n,
        scenario_text=scenario_text,
    )

    # 生成种子驱动的 market_state 和 psych 变量（第三类常量 → 合成阶段生成）
    ms_seed = _rule_seed(codename, effective_lineup, str(n), scenario_text[:512])
    scene_type = str(env_ctx.get("scene_type") or "")
    roles = tuple(negotiation_role_order(effective_lineup)[:n])
    market_state = _build_v2_market_state(ms_seed, scene_type,
                                          physical_params=env_ctx.get("physical_social_parameters"))
    psych_variables = _build_v2_psych_variables(ms_seed, roles, scene_type)

    return {
        "pipeline": "long_term_negotiation",
        "strict_design_v1": strict,
        "quartet": quartet,
        "num_participants": n,
        "lineup": effective_lineup,
        "institutional_roles_enabled": bool(
            effective_lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL
        ),
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
        "market_state": market_state,
        "agent_psych_variables": psych_variables,
        "dialogue_style": {
            "version": 1,
            "synthesis_requirements_en": DIALOGUE_STYLE_SYNTHESIS_APPEND_EN,
            "evaluation_requirements_en": DIALOGUE_STYLE_EVAL_RUBRIC_EN,
        },
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


__all__ = [
    "_build_v2_market_state",
    "_build_v2_outcome_rule",
    "_build_v2_psych_variables",
    "build_extended_negotiation_game_metadata_bundle",
]
