"""将结构化 ``AgentAction`` 映射到 ``SystemState`` 与 ``ContractLedger``。

支持两类入口：

1. **显式 ``action_type``**（设计文档 §A）：``transfer_resource`` / ``propose_contract`` / …  
2. **兼容 PoC**：``action_type == \"action\"`` 且 ``argument`` 为 JSON 字符串或 dict，
   内含 ``op`` 字段的老格式。
"""

from __future__ import annotations

import json
from typing import Any

from sotopia.benchmark_v2_data_models import Contract
from sotopia.messages.message_classes import AgentAction

from sotopia.state.contracts import ContractLedger
from sotopia.state.system_state import SystemState


def _as_dict(arg: str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(arg, dict):
        return arg
    if isinstance(arg, str) and arg.strip().startswith("{"):
        try:
            o = json.loads(arg)
            return o if isinstance(o, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _pk_from_argument(act: AgentAction) -> str:
    d = _as_dict(act.argument)
    if d and "pk" in d:
        return str(d["pk"])
    if isinstance(act.argument, str):
        return act.argument.strip()
    return ""


class ActionDispatcher:
    def dispatch(
        self,
        state: SystemState,
        ledger: ContractLedger,
        *,
        complied_actions: dict[str, AgentAction],
        calendar_day: int,
        episode_pk: str,
        agent_names: list[str],
    ) -> str:
        lines: list[str] = []
        for agent, act in complied_actions.items():
            at = act.action_type
            if at == "action":
                sub = _as_dict(act.argument)
                if sub and "op" in sub:
                    r = self._dispatch_legacy_op(
                        state,
                        ledger,
                        agent=agent,
                        payload=sub,
                        calendar_day=calendar_day,
                        episode_pk=episode_pk,
                    )
                    if r:
                        lines.append(r)
                continue

            if at == "transfer_resource":
                r = self._transfer_resource(
                    state, agent, _as_dict(act.argument) or {}, agent_names
                )
                if r:
                    lines.append(r)
            elif at == "propose_contract":
                d = _as_dict(act.argument)
                if d:
                    lines.append(
                        self._propose_contract(
                            ledger, agent, d, calendar_day, episode_pk
                        )
                    )
            elif at == "accept":
                pk = _pk_from_argument(act)
                if pk and ledger.get(pk) is not None:
                    ledger.promote_status(
                        pk,
                        "active",
                        note={"day": calendar_day, "by": agent},
                    )
                    lines.append(f"{agent}: accept {pk[:8]}")
            elif at == "reject":
                pk = _pk_from_argument(act)
                if pk and ledger.get(pk) is not None:
                    ledger.promote_status(
                        pk,
                        "cancelled",
                        note={"day": calendar_day, "by": agent},
                    )
                    lines.append(f"{agent}: reject {pk[:8]}")
            elif at == "defect":
                lines.append(self._defect(state, agent, agent_names))
            elif at == "invest":
                lines.append(self._invest(state, agent, _as_dict(act.argument) or {}))
            elif at == "withdraw":
                lines.append(
                    self._withdraw(state, agent, _as_dict(act.argument) or {})
                )
            elif at == "vote":
                lines.append(self._vote(state, agent, _as_dict(act.argument) or {}))

        return "; ".join(x for x in lines if x)

    def _dispatch_legacy_op(
        self,
        state: SystemState,
        ledger: ContractLedger,
        *,
        agent: str,
        payload: dict[str, Any],
        calendar_day: int,
        episode_pk: str,
    ) -> str:
        op = payload.get("op")
        if op == "trust_delta":
            other = str(payload.get("target", ""))
            delta = float(payload.get("value", 0.0))
            state.trust_matrix.setdefault(agent, {})
            cur = float(state.trust_matrix[agent].get(other, 0.0))
            state.trust_matrix[agent][other] = cur + delta
            return f"{agent}: trust {other} {delta:+.2f}"
        if op == "transfer_resource":
            return self._transfer_resource(
                state,
                agent,
                {
                    "to": payload.get("to"),
                    "resource": payload.get("resource", "cash"),
                    "amount": payload.get("amount", 0),
                },
                list(state.agent_keys),
            )
        if op == "propose_contract":
            return self._propose_contract(
                ledger, agent, payload, calendar_day, episode_pk
            )
        if op == "activate_contract":
            pk = str(payload.get("pk", ""))
            if ledger.get(pk) is not None:
                ledger.promote_status(
                    pk,
                    "active",
                    note={"day": calendar_day, "by": agent},
                )
                return f"{agent}: activate_contract {pk[:8]}"
        if op == "fulfill_contract":
            pk = str(payload.get("pk", ""))
            if ledger.get(pk) is not None:
                ledger.patch_contract_terms(pk, {"fulfilled": True})
                return f"{agent}: fulfill_contract {pk[:8]}"
        return ""

    def _transfer_resource(
        self,
        state: SystemState,
        agent: str,
        payload: dict[str, Any],
        agent_names: list[str],
    ) -> str:
        resource = str(payload.get("resource", "cash"))
        amount = float(payload.get("amount", 0.0))
        to = str(payload.get("to", ""))
        if not to or amount <= 0:
            return ""
        state.agent_resources.setdefault(agent, {})
        state.agent_resources.setdefault(to, {})
        fr = float(state.agent_resources[agent].get(resource, 0.0))
        if fr < amount:
            return f"{agent}: transfer_resource rejected"
        state.agent_resources[agent][resource] = fr - amount
        tt = float(state.agent_resources[to].get(resource, 0.0))
        state.agent_resources[to][resource] = tt + amount
        return f"{agent} -> {to} {resource} {amount:.2f}"

    def _propose_contract(
        self,
        ledger: ContractLedger,
        agent: str,
        payload: dict[str, Any],
        calendar_day: int,
        episode_pk: str,
    ) -> str:
        cps = payload.get("counterparties") or []
        if not isinstance(cps, list):
            cps = []
        ledger.add_contract(
            Contract(
                episode_pk=episode_pk,
                proposer_pk=agent,
                counterparties=[str(x) for x in cps],
                contract_type=str(payload.get("contract_type", "agreement")),
                terms=dict(payload.get("terms", {})),
                penalty=dict(payload.get("penalty", {})),
                proposed_day=calendar_day,
                expiry_day=payload.get("expiry_day"),
                status="proposed",
            )
        )
        return f"{agent}: propose_contract"

    def _defect(
        self, state: SystemState, agent: str, agent_names: list[str]
    ) -> str:
        for other in agent_names:
            if other == agent:
                continue
            state.trust_matrix.setdefault(agent, {})
            cur = float(state.trust_matrix[agent].get(other, 0.0))
            state.trust_matrix[agent][other] = cur - 0.15
        rep = float(state.agent_reputation.get(agent, 50.0))
        state.agent_reputation[agent] = max(0.0, rep - 8.0)
        return f"{agent}: defect"

    def _invest(
        self, state: SystemState, agent: str, payload: dict[str, Any]
    ) -> str:
        amount = float(payload.get("amount", 0.0))
        pool_key = str(payload.get("pool_key", "invested_capital"))
        state.agent_resources.setdefault(agent, {})
        cash = float(state.agent_resources[agent].get("cash", 0.0))
        if amount <= 0 or cash < amount:
            return f"{agent}: invest rejected"
        state.agent_resources[agent]["cash"] = cash - amount
        state.resource_pool[pool_key] = float(state.resource_pool.get(pool_key, 0.0)) + amount
        return f"{agent}: invest {amount:.2f} -> {pool_key}"

    def _withdraw(
        self, state: SystemState, agent: str, payload: dict[str, Any]
    ) -> str:
        amount = float(payload.get("amount", 0.0))
        pool_key = str(payload.get("pool_key", "invested_capital"))
        if amount <= 0:
            return ""
        pool = float(state.resource_pool.get(pool_key, 0.0))
        if pool < amount:
            return f"{agent}: withdraw rejected"
        state.resource_pool[pool_key] = pool - amount
        state.agent_resources.setdefault(agent, {})
        cash = float(state.agent_resources[agent].get("cash", 0.0))
        state.agent_resources[agent]["cash"] = cash + amount
        return f"{agent}: withdraw {amount:.2f}"

    def _vote(
        self, state: SystemState, agent: str, payload: dict[str, Any]
    ) -> str:
        pid = str(payload.get("proposal_id", "default"))
        value = float(payload.get("value", 1.0))
        votes = state.scratch.setdefault("votes", {})
        prop = votes.setdefault(pid, {})
        prop[agent] = value
        return f"{agent}: vote {pid}={value}"


__all__ = ["ActionDispatcher"]
