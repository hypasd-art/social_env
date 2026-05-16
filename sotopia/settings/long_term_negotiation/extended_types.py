"""合同经济学确定性化 V2 — 纯常量与类型定义。

与设计文档对齐：
- 合同类型由场景决定：合作合同（cooperation）vs 买卖合同（buy_sell）
- V2 outcome rule 版本标识
- scene_type → contract_type 映射
"""

from __future__ import annotations

#: 合作合同：合作产生利润 → 按份额分配（business_coopetition, resource_scheduling_management）
CONTRACT_TYPE_COOPERATION: str = "cooperation"

#: 买卖合同：买方省钱 = reference - agreed，卖方赚钱 = agreed - cost
CONTRACT_TYPE_BUY_SELL: str = "buy_sell"

#: V2 outcome rule 版本标识
PREDETERMINED_OUTCOME_RULE_VERSION: str = "v2"

#: payout_mode 取值
PAYOUT_MODE_PREDETERMINED: str = "predetermined"
PAYOUT_MODE_PRICE_DIFFERENCE: str = "price_difference"

# scene_type → contract_type 映射
_SCENE_TO_CONTRACT_TYPE: dict[str, str] = {
    "business_coopetition": CONTRACT_TYPE_COOPERATION,
    "business_outsourcing": CONTRACT_TYPE_COOPERATION,
    "resource_scheduling_management": CONTRACT_TYPE_COOPERATION,
    "wet_market_competition": CONTRACT_TYPE_BUY_SELL,
    "competitive_bidding": CONTRACT_TYPE_BUY_SELL,
}


def scene_type_to_contract_type(scene_type: str | None) -> str:
    """由场景类型推导合同类型；未知场景默认视为合作合同。"""
    if not scene_type:
        return CONTRACT_TYPE_COOPERATION
    return _SCENE_TO_CONTRACT_TYPE.get(str(scene_type).strip(), CONTRACT_TYPE_COOPERATION)


def is_cooperation_contract(scene_type: str | None) -> bool:
    return scene_type_to_contract_type(scene_type) == CONTRACT_TYPE_COOPERATION


def is_buy_sell_contract(scene_type: str | None) -> bool:
    return scene_type_to_contract_type(scene_type) == CONTRACT_TYPE_BUY_SELL


__all__ = [
    "CONTRACT_TYPE_COOPERATION",
    "CONTRACT_TYPE_BUY_SELL",
    "PREDETERMINED_OUTCOME_RULE_VERSION",
    "PAYOUT_MODE_PREDETERMINED",
    "PAYOUT_MODE_PRICE_DIFFERENCE",
    "scene_type_to_contract_type",
    "is_cooperation_contract",
    "is_buy_sell_contract",
]
