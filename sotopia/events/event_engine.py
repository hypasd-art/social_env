"""外部事件：默认只在日终阶段评估是否触发，并对 ``SystemState`` 施加 effect。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sotopia.state.system_state import SystemState


def calendar_days_with_end_of_day_scripts(scripts: Sequence[Any]) -> set[int]:
    """与 ``EventEngine.scripts_for_end_of_day`` 一致：哪些 ``calendar_day`` 至少有一条 **非 intraday** 脚本在日终触发。

    用于校验或补齐「每个自然日至少一条 EventScript（EOD）」类约束。
    """
    days: set[int] = set()
    for s in scripts:
        if bool(getattr(s, "intraday", False)):
            continue
        apply_days = list(getattr(s, "apply_days", None) or [])
        step = getattr(s, "step", None)
        if apply_days:
            days.update(int(x) for x in apply_days)
        elif step is not None:
            days.add(int(step))
    return days


@dataclass
class EventEngineConfig:
    """控制事件管线；事件脚本为 duck-typing 对象（须有 ``intraday`` / ``apply_days`` / ``step`` / ``effects`` / ``pk``）。"""

    end_of_day_events_enabled: bool = True
    intraday_events_enabled: bool = False
    #: design_1 §2.1 — post-session / slot 收口后触发；
    #: 需在脚本上设 ``intraday=True``、``apply_days`` 含当日、``step`` 等于槽位 ``k``。
    post_session_slot_events_enabled: bool = True


class EventEngine:
    def __init__(self, config: EventEngineConfig | None = None) -> None:
        self.config = config or EventEngineConfig()

    def scripts_for_end_of_day(
        self, calendar_day: int, scripts: Sequence[Any]
    ) -> list[Any]:
        if not self.config.end_of_day_events_enabled:
            return []
        out: list[Any] = []
        for s in scripts:
            if s.intraday:
                continue
            if s.apply_days and calendar_day in s.apply_days:
                out.append(s)
            elif not s.apply_days and s.step is not None:
                if int(s.step) == int(calendar_day):
                    out.append(s)
        return out

    def scripts_for_intraday(self, turn_number: int, scripts: Sequence[Any]) -> list[Any]:
        if not self.config.intraday_events_enabled:
            return []
        return [s for s in scripts if s.intraday and s.step == turn_number]

    def scripts_for_post_session_slot(
        self,
        calendar_day: int,
        session_slot: int,
        scripts: Sequence[Any],
    ) -> list[Any]:
        """§2.1 slot 收口后：``intraday=True``、``apply_days`` 含 ``calendar_day``、``step=k``。"""
        if not self.config.post_session_slot_events_enabled:
            return []
        out: list[Any] = []
        k = int(session_slot)
        d = int(calendar_day)
        for s in scripts:
            if not s.intraday:
                continue
            if not s.apply_days or d not in s.apply_days:
                continue
            if s.step is None or int(s.step) != k:
                continue
            out.append(s)
        return out

    def apply_scripts(self, scripts: Sequence[Any], state: SystemState) -> list[str]:
        triggered: list[str] = []
        for s in scripts:
            for eff in s.effects:
                state.apply_effect(eff)
            pk = getattr(s, "pk", None)
            if pk:
                triggered.append(str(pk))
        return triggered


__all__ = ["EventEngine", "EventEngineConfig", "calendar_days_with_end_of_day_scripts"]
