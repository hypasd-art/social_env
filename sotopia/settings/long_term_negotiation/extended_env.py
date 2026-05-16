"""合同经济学确定性化 V2 — 继承 LongTermNegotiationEnv。

ExtendedLongTermNegotiationEnv 仅重写 ``_apply_contract_status_settlement_if_needed``：
- 检测 rule version，V2 走新逻辑，V1 调 super()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .env import LongTermNegotiationEnv
from .extended_negotiation_metrics import (
    _is_v2_rule,
    compute_v2_settlement_by_contract_status,
)


class ExtendedLongTermNegotiationEnv(LongTermNegotiationEnv):
    """V2 扩展：继承 LongTermNegotiationEnv，重写合同结算逻辑。

    与父类完全兼容，仅结算路径按 rule version 分发：
    - V2 → ``compute_v2_settlement_by_contract_status``（确定性 payout）
    - V1 → ``super()._apply_contract_status_settlement_if_needed()``
    """

    def _apply_contract_status_settlement_if_needed(self) -> None:
        """按主合同状态自动结算（V2 路径）。

        V2 规则：使用确定性 payout（合作合同取 predetermined_payouts，
        买卖合同按 reference - agreed / agreed - cost 计算）。
        """
        if self._contract_status_settlement_applied:
            return
        if not self.predefined_outcome_rule:
            return
        pcs = getattr(self.ctrl, "primary_contract_id", None)
        if not pcs:
            return
        c = getattr(self.ctrl, "contracts", {}).get(pcs)
        if c is None:
            return
        status = str(getattr(c, "status", "") or "").lower()
        if status not in {"proposed", "amended", "accepted", "signed", "rejected", "failed"}:
            return

        if not _is_v2_rule(self.predefined_outcome_rule):
            # V1: delegate to parent
            super()._apply_contract_status_settlement_if_needed()
            return

        metrics = compute_v2_settlement_by_contract_status(
            env=self,
            predefined_outcome_rule=self.predefined_outcome_rule,
            contract_status=status,
        )

        settlement_by_agent: dict[str, float] = {}
        for agent in self.system_state.agent_keys:
            ind_key = f"negotiation_predefined_rule_individual_profit_{agent}"
            if ind_key in metrics:
                payout = float(metrics.get(ind_key, 0.0) or 0.0)
            else:
                payout = 0.0
            if payout == 0.0:
                continue
            res = self.system_state.agent_resources.setdefault(agent, {})
            cash0 = float(res.get("cash", 0.0) or 0.0)
            res["cash"] = cash0 + payout
            settlement_by_agent[agent] = payout

        self._contract_status_settlement_applied = True
        if settlement_by_agent:
            self.ctrl.record_execution_event(
                "contract_settlement_applied_v2",
                "已按主合同状态与 V2 predefined_outcome_rule 自动结算并更新现金。",
                primary_contract_status=status,
                settlement_by_agent=settlement_by_agent,
                total_settlement=float(sum(settlement_by_agent.values())),
            )


__all__ = ["ExtendedLongTermNegotiationEnv"]
