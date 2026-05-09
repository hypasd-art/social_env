"""``design_1.md`` §8 — 外部事件（scheduled / condition_based）与观察注入。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sotopia.state import SystemState

from .types import Phase

EventTiming = str  # start_of_day | before_scheduling | after_formal_action | after_session | end_of_slot | end_of_day


def _effect_payload_row_as_dict(row: Any) -> dict[str, Any]:
    """将 ``effects`` 条目规范为 ``dict``（兼容纯 dict 与带 ``model_dump`` 的模型）。"""
    if isinstance(row, dict):
        return dict(row)
    md = getattr(row, "model_dump", None)
    if callable(md):
        return dict(md())
    return {}
@dataclass
class ExternalEventDefinition:
    """§8.1 事件对象（JSON / dict 可还原）。"""

    event_id: str
    event_type: str
    trigger: dict[str, Any]
    visibility_set: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    observation_text: str = ""
    #: 复用 ``EffectOp`` 机制层，作用于 ``SystemState``（market_state / agent_resources 等）
    effects: list[dict[str, Any]] = field(default_factory=list)
    #: 额外作用于 ``NegotiationWorldController`` 的合同 / 状态（见 ``apply_negotiation_ops``）
    negotiation_effects: list[dict[str, Any]] = field(default_factory=list)
    #: §8.3 / §8.4 — 仅指定 agent 可见的私有说明（不进入 ``visibility_set`` 的公共块）
    private_information: dict[str, str] = field(default_factory=dict)
    once: bool = True
    max_triggers: int = 1


def external_event_from_dict(raw: dict[str, Any]) -> ExternalEventDefinition:
    return ExternalEventDefinition(
        event_id=str(raw["event_id"]),
        event_type=str(raw.get("event_type", "generic")),
        trigger=dict(raw.get("trigger") or {}),
        visibility_set=list(raw.get("visibility_set") or []),
        payload=dict(raw.get("payload") or {}),
        observation_text=str(raw.get("observation_text") or ""),
        effects=list(raw.get("effects") or []),
        negotiation_effects=list(raw.get("negotiation_effects") or []),
        private_information={str(k): str(v) for k, v in dict(raw.get("private_information") or {}).items()},
        once=bool(raw.get("once", True)),
        max_triggers=int(raw.get("max_triggers", 1)),
    )


ConditionFn = Callable[[Any, SystemState, dict[str, Any]], bool]


def _pred_buyer_cash_below_primary_price(
    ctrl: Any, state: SystemState, tr: dict[str, Any]
) -> bool:
    _ = tr
    cid = ctrl.primary_contract_id
    if not cid:
        return False
    c = ctrl.contracts.get(cid)
    if not c:
        return False
    try:
        price = float(c.terms.get("price", 0) or 0)
        cash = float(state.agent_resources.get("firm_a", {}).get("cash", 0.0))
    except (TypeError, ValueError):
        return False
    return cash + 1e-9 < price


def _pred_contract_price_above_threshold(
    ctrl: Any, state: SystemState, tr: dict[str, Any]
) -> bool:
    _ = state
    cid = ctrl.primary_contract_id
    if not cid:
        return False
    c = ctrl.contracts.get(cid)
    if not c:
        return False
    try:
        price = float(c.terms.get("price", 0) or 0)
        th = float(tr.get("threshold", tr.get("params", {}).get("threshold", 0)) or 0)
    except (TypeError, ValueError):
        return False
    return price > th + 1e-9


def _pred_primary_contract_accepted(ctrl: Any, state: SystemState, tr: dict[str, Any]) -> bool:
    _ = state
    _ = tr
    cid = ctrl.primary_contract_id
    if not cid:
        return False
    return ctrl.contracts.get(cid, None) is not None and ctrl.contracts[cid].status == "accepted"


CONDITION_REGISTRY: dict[str, ConditionFn] = {
    "buyer_cash_below_required_payment": _pred_buyer_cash_below_primary_price,
    "buyer_cash_below_primary_price": _pred_buyer_cash_below_primary_price,
    "contract_valuation_above_threshold": _pred_contract_price_above_threshold,
    "primary_contract_accepted": _pred_primary_contract_accepted,
}


def _scheduled_matches(tr: dict[str, Any], *, timing: EventTiming, day: int, slot: int) -> bool:
    if str(tr.get("mode")) != "scheduled":
        return False
    if str(tr.get("timing")) != timing:
        return False
    if int(tr.get("day", -1)) != int(day):
        return False
    if "slot_id" in tr and tr["slot_id"] is not None:
        if int(tr["slot_id"]) != int(slot):
            return False
    return True


def _condition_matches(
    tr: dict[str, Any],
    *,
    timing: EventTiming,
    ctrl: Any,
    state: SystemState,
) -> bool:
    if str(tr.get("mode")) != "condition_based":
        return False
    if str(tr.get("timing")) != timing:
        return False
    name = str(tr.get("condition") or "")
    fn = CONDITION_REGISTRY.get(name)
    if fn is None:
        return False
    return bool(fn(ctrl, state, tr))


class NegotiationExternalEventRunner:
    """在指定 ``timing`` 评估 ``ExternalEventDefinition`` 列表并写 log / 观察队列。"""

    def __init__(self, events: list[ExternalEventDefinition]) -> None:
        self.events = list(events)
        self._fire_count: dict[str, int] = {}
        self._fired_once: set[str] = set()

    @classmethod
    def from_spec_dicts(cls, specs: list[dict[str, Any]]) -> NegotiationExternalEventRunner:
        return cls([external_event_from_dict(s) for s in specs])

    def reset(self) -> None:
        self._fire_count.clear()
        self._fired_once.clear()

    def _can_fire(self, ev: ExternalEventDefinition) -> bool:
        if ev.once and ev.event_id in self._fired_once:
            return False
        n = self._fire_count.get(ev.event_id, 0)
        return n < ev.max_triggers

    def _mark_fired(self, ev: ExternalEventDefinition) -> None:
        self._fire_count[ev.event_id] = self._fire_count.get(ev.event_id, 0) + 1
        if ev.once:
            self._fired_once.add(ev.event_id)

    def tick(
        self,
        timing: EventTiming,
        *,
        day: int,
        slot: int,
        phase: Phase,
        ctrl: Any,
        state: SystemState,
    ) -> list[str]:
        """执行本 timing 下命中的事件；返回触发的 ``event_id`` 列表。"""
        if ctrl.terminal or ctrl.phase == Phase.TERMINATED:
            return []

        triggered: list[str] = []
        for ev in self.events:
            if not self._can_fire(ev):
                continue
            tr = ev.trigger
            ok = False
            if str(tr.get("mode")) == "scheduled":
                ok = _scheduled_matches(tr, timing=timing, day=day, slot=slot)
            elif str(tr.get("mode")) == "condition_based":
                ok = _condition_matches(tr, timing=timing, ctrl=ctrl, state=state)

            if not ok:
                continue

            self._fire_one(ev, timing=timing, day=day, slot=slot, phase=phase, ctrl=ctrl, state=state)
            triggered.append(ev.event_id)

        return triggered

    def _fire_one(
        self,
        ev: ExternalEventDefinition,
        *,
        timing: str,
        day: int,
        slot: int,
        phase: Phase,
        ctrl: Any,
        state: SystemState,
    ) -> None:
        applied_effects: list[str] = []
        for raw in ev.effects:
            row = _effect_payload_row_as_dict(raw)
            if not row:
                continue
            try:
                state.apply_effect(row)
            except Exception:
                continue
            applied_effects.append(f"{row.get('op', '')}:{row.get('target', '')}")

        ctrl.apply_external_negotiation_effects(ev.negotiation_effects)

        obs_by_agent: dict[str, str] = {}
        common = (
            "External Event Observed:\n"
            f"- event_id={ev.event_id}\n"
            f"- event_type={ev.event_type}\n"
            f"- timing={timing} day={day} slot={slot} phase={phase.value}\n"
            f"- {ev.observation_text.strip()}\n"
        )
        if ev.payload:
            common += f"- payload={ev.payload!r}\n"

        vis = set(ev.visibility_set or [])
        for ag in ctrl.agent_names:
            if ag not in vis:
                continue
            ctrl.enqueue_external_event_notification(ag, common.strip())
            obs_by_agent[ag] = common

        for ag, priv in ev.private_information.items():
            if ag in ctrl.agent_names:
                blk = (
                    "External Event Observed (private to you):\n"
                    f"- event_id={ev.event_id}\n"
                    f"- event_type={ev.event_type}\n"
                    f"- {priv.strip()}\n"
                )
                ctrl.enqueue_external_event_notification(ag, blk.strip())

        ctrl.append_event_records(
            [
                {
                    "kind": "external_event_v8",
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "triggered_at": {
                        "timing": timing,
                        "calendar_day": day,
                        "slot_index": slot,
                        "phase": phase.value,
                    },
                    "visible_to": sorted(vis),
                    "state_effect_applied": applied_effects,
                    "observation_text_by_agent": obs_by_agent,
                    "negotiation_effects": list(ev.negotiation_effects),
                }
            ]
        )

        ctrl.mark_structural_progress()
        self._mark_fired(ev)


__all__ = [
    "CONDITION_REGISTRY",
    "ExternalEventDefinition",
    "NegotiationExternalEventRunner",
    "external_event_from_dict",
]
