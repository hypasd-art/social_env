"""运行时合约账本：生命周期与「日终」结算（到期、违约惩罚）。"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Callable, Iterable

from sotopia.benchmark_v2_data_models import Contract, ContractStatus

from .system_state import SystemState


ApplyPenaltyFn = Callable[[SystemState, str, dict], None]


def _default_penalty(state: SystemState, agent_key: str, penalty: dict) -> None:
    rep_d = penalty.get("reputation_delta")
    if rep_d is not None:
        cur = float(state.agent_reputation.get(agent_key, 50.0))
        state.agent_reputation[agent_key] = cur + float(rep_d)

    cash_d = penalty.get("cash_delta")
    if cash_d is not None:
        state.agent_resources.setdefault(agent_key, {})
        cash = float(state.agent_resources[agent_key].get("cash", 0.0))
        state.agent_resources[agent_key]["cash"] = cash + float(cash_d)


class ContractLedger:
    """内存中的合约集；可选用 ``persist`` 写入 JsonModel backend。"""

    def __init__(
        self,
        *,
        episode_pk: str = "",
        on_mutate: Callable[[Contract], None] | None = None,
        apply_penalty: ApplyPenaltyFn = _default_penalty,
    ) -> None:
        self.episode_pk = episode_pk
        self._by_pk: dict[str, Contract] = {}
        self._last_changed_pks: list[str] = []
        self.on_mutate = on_mutate
        self.apply_penalty = apply_penalty

    def set_episode(self, episode_pk: str) -> None:
        self.episode_pk = episode_pk

    def clear(self) -> None:
        self._by_pk.clear()
        self._last_changed_pks.clear()

    def drain_last_changed_pks(self) -> list[str]:
        out = list(self._last_changed_pks)
        self._last_changed_pks.clear()
        return out

    def get(self, pk: str) -> Contract | None:
        return self._by_pk.get(pk)

    def iter_contracts(self) -> Iterable[Contract]:
        return self._by_pk.values()

    def touch(self, c: Contract) -> None:
        self._last_changed_pks.append(c.pk or "")  # type: ignore[list-item]
        if self.on_mutate:
            self.on_mutate(c)

    # --- authoring ---------------------------------------------------------

    def add_contract(self, prototype: Contract) -> Contract:
        """插入一条合约（常为 proposed）。"""
        data = prototype.model_dump()
        if not data.get("pk"):
            data["pk"] = uuid.uuid4().hex
        c = Contract(**data)
        c.episode_pk = self.episode_pk or c.episode_pk
        self._by_pk[c.pk or ""] = c
        self.touch(c)
        return c

    def promote_status(
        self, pk: str, new_status: ContractStatus, *, note: dict | None = None
    ) -> None:
        c = self._by_pk.get(pk)
        if c is None:
            return
        entry: dict[str, Any] = {"from": c.status, "to": new_status}
        if note:
            entry.update(note)
        c.history = [*list(c.history), entry]
        c.status = new_status
        self.touch(c)

    def patch_contract_terms(self, pk: str, patch: dict) -> None:
        """合并写入 ``terms``（避免 pydantic JsonModel in-place 变更不确定）。"""
        c = self.get(pk)
        if c is None:
            return
        data = c.model_dump()
        data["terms"] = {**dict(data.get("terms", {})), **patch}
        n = Contract(**data)
        new_pk = n.pk or pk
        if new_pk != pk:
            self._by_pk.pop(pk, None)
        self._by_pk[new_pk] = n
        self.touch(n)

    # --- day-end settlement ------------------------------------------------

    def end_of_day(self, finished_calendar_day: int, state: SystemState) -> None:
        """到期 → expired；对 active 且违反简单条件的合约施加 penalty（PoC：仅检查 maturity）。"""

        for c in list(self._by_pk.values()):
            if c.status not in ("active", "proposed"):
                continue

            exp = c.terms.get("maturity_day")
            if c.status == "active" and exp is not None:
                if finished_calendar_day >= int(exp):
                    self._expire_or_breach(c, finished_calendar_day, state)

            if c.status == "proposed" and c.expiry_day is not None:
                if finished_calendar_day >= int(c.expiry_day):
                    self.promote_status(
                        c.pk or "",
                        "expired",
                        note={"day": finished_calendar_day, "reason": "proposal_expired"},
                    )

    def _expire_or_breach(
        self, c: Contract, finished_calendar_day: int, state: SystemState
    ) -> None:
        fulfilled = bool(c.terms.get("fulfilled", False))
        if fulfilled:
            self.promote_status(
                c.pk or "",
                "fulfilled",
                note={"day": finished_calendar_day},
            )
            return
        self.promote_status(
            c.pk or "",
            "breached",
            note={"day": finished_calendar_day},
        )
        parties = [c.proposer_pk] + list(c.counterparties)
        pen = copy.deepcopy(c.penalty) if c.penalty else {}
        for ag in parties:
            self.apply_penalty(state, ag, pen)


__all__ = ["ContractLedger", "ApplyPenaltyFn", "_default_penalty"]
