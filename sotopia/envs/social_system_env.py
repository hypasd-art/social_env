"""长周期社会系统环境：在 ``ParallelSotopiaEnv`` 上叠加 ActionDispatcher、
``SystemState``、日终 ``EventEngine`` 与合约结算，并可选持久化 ``SystemStateSnapshot``。

时间轴（与 ``EnvironmentProfileV2`` 对齐）：
- 每个 async step = 一个「日内」交互步；连续 ``intra_day_steps`` 步结束后调用
  ``end_of_day()``（也可在外部手动调用）。
- ``end_of_day()``：按 config 触发外部事件 → 合约日终结算 → 写状态快照。

与主流程对接时，请在 ``reset(..., options={"episode_pk": "<EpisodeLogV2.pk>"})``
中传入 episode 主键，便于合约与快照外键一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, get_args

from sotopia.benchmark_v2_data_models import Contract, EventScript, SystemStateSnapshot
from sotopia.envs.action_dispatcher import ActionDispatcher
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.events import EventEngine, EventEngineConfig
from sotopia.messages.message_classes import ActionType, AgentAction, Observation, SimpleMessage
from sotopia.state import ContractLedger, SystemState, state_from_profile_init

_DEFAULT_SOCIAL_ACTION_TYPES: set[str] = set(get_args(ActionType))


@dataclass
class SocialSystemConfig:
    """运行时策略；可被 ``EnvironmentProfileV2`` 字段覆盖（见 ``SocialSystemEnv.reset``）。"""

    intra_day_steps: int = 1
    """完成多少个 ``astep`` 后触发一次 ``end_of_day``。"""

    episode_pk: str = ""
    """Episode 外键；也接受 ``reset(..., options={\"episode_pk\": ...})``。"""

    event_scripts: list[EventScript] | None = None
    """显式事件表；为 None 时尝试 ``profile.event_schedule_pk`` 加载单条 ``EventScript``。"""

    persist_snapshots: bool = False
    """为 True 时 ``SystemStateSnapshot.save()``。"""

    persist_contracts: bool = False
    """为 True 时合约变更 ``Contract.save()``。"""

    event_engine_config: EventEngineConfig = field(default_factory=EventEngineConfig)

    max_calendar_days: int | None = None
    """可选：达到后不再自动 ``end_of_day``（仅防无限跑；终止仍由评测器决定）。"""


class SocialSystemEnv(ParallelSotopiaEnv):
    def __init__(
        self,
        *args: Any,
        social_config: SocialSystemConfig | None = None,
        available_action_types: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if available_action_types is None and kwargs.get("available_action_types") is None:
            kwargs["available_action_types"] = set(_DEFAULT_SOCIAL_ACTION_TYPES)
        elif available_action_types is not None:
            kwargs["available_action_types"] = set(available_action_types)
        super().__init__(*args, **kwargs)
        self.social_config = social_config or SocialSystemConfig()
        self.dispatcher = ActionDispatcher()
        self.event_engine = EventEngine(self.social_config.event_engine_config)
        self.contracts = ContractLedger(
            episode_pk=self.social_config.episode_pk,
            on_mutate=self._persist_contract_if_needed,
        )
        self.system_state = SystemState([])
        self._event_scripts: list[EventScript] = []
        self._calendar_day = 1
        self._step_in_day = 0
        self._episode_pk = ""
        self.snapshot_pks: list[str] = []

    def set_social_episode_context(self, episode_pk: str) -> None:
        """在 ``reset`` 之外补设 episode 主键（例如 server 在创建 log 后才拿到 pk）。"""
        self.social_config.episode_pk = episode_pk
        self._episode_pk = episode_pk
        self.contracts.set_episode(episode_pk)

    def _persist_contract_if_needed(self, c: Contract) -> None:
        if self.social_config.persist_contracts:
            c.save()

    def _load_event_scripts(self) -> list[EventScript]:
        if self.social_config.event_scripts is not None:
            return list(self.social_config.event_scripts)
        sched_pk = getattr(self.profile, "event_schedule_pk", None)
        if not sched_pk:
            return []
        try:
            return [EventScript.get(sched_pk)]
        except Exception:
            return []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, str] | None = None,
        agents: Any = None,
        omniscient: bool = False,
        lite: bool = False,
        include_background_observations: bool | None = True,
    ) -> dict[str, Observation]:
        observations = super().reset(
            seed=seed,
            options=options,
            agents=agents,
            omniscient=omniscient,
            lite=lite,
            include_background_observations=include_background_observations,
        )

        opts = options or {}
        self._episode_pk = str(
            opts.get("episode_pk", self.social_config.episode_pk) or ""
        )
        self.contracts.set_episode(self._episode_pk)

        profile_steps = getattr(self.profile, "intra_day_steps", None)
        if profile_steps is not None:
            self.social_config.intra_day_steps = max(1, int(profile_steps))
        self.social_config.intra_day_steps = max(1, int(self.social_config.intra_day_steps))

        init_dict = getattr(self.profile, "system_state_init", None)
        if not isinstance(init_dict, dict):
            init_dict = {}
        self.system_state = state_from_profile_init(self.agents, init_dict)

        self.contracts.clear()
        self._event_scripts = self._load_event_scripts()
        self._calendar_day = 1
        self._step_in_day = 0
        self.snapshot_pks = []

        if self.social_config.persist_snapshots and self._episode_pk:
            z = self.system_state.to_snapshot(
                episode_pk=self._episode_pk,
                day=0,
                triggered_event_pks=[],
                contracts_changed_pks=[],
            )
            z.save()
            if z.pk:
                self.snapshot_pks.append(z.pk)

        return observations

    async def _after_actions_processed_astep(
        self, complied_actions: dict[str, AgentAction]
    ) -> None:
        note = self.dispatcher.dispatch(
            self.system_state,
            self.contracts,
            complied_actions=complied_actions,
            calendar_day=self._calendar_day,
            episode_pk=self._episode_pk,
            agent_names=list(self.agents),
        )
        if note:
            self.recv_message(
                "Environment",
                SimpleMessage(message=f"[action_dispatch] {note}"),
            )

    async def _before_return_astep(
        self,
        observations: dict[str, Observation],
        info: dict[str, dict[Any, Any]],
    ) -> None:
        for name in self.agents:
            if name in info:
                info[name]["social_system_calendar_day"] = self._calendar_day
                info[name]["system_state_line"] = self.system_state.digest_line(
                    viewer=name
                )
            if name in observations:
                extra = self.system_state.digest_line(viewer=name)
                o = observations[name]
                observations[name] = o.model_copy(
                    update={"last_turn": o.last_turn + "\n" + extra}
                )

        self._step_in_day += 1
        if self._step_in_day >= self.social_config.intra_day_steps:
            self._step_in_day = 0
            max_d = self.social_config.max_calendar_days
            if max_d is None or self._calendar_day <= max_d:
                await self.end_of_day()

    async def end_of_day(self) -> SystemStateSnapshot | None:
        """日终：事件 → 合约结算 → 快照；随后 ``_calendar_day`` +1。"""

        if self.social_config.max_calendar_days is not None:
            if self._calendar_day > self.social_config.max_calendar_days:
                return None

        to_apply = self.event_engine.scripts_for_end_of_day(
            self._calendar_day, self._event_scripts
        )
        triggered = self.event_engine.apply_scripts(to_apply, self.system_state)

        self.contracts.end_of_day(self._calendar_day, self.system_state)
        contract_delta = self.contracts.drain_last_changed_pks()

        snap = self.system_state.to_snapshot(
            episode_pk=self._episode_pk or "unknown_episode",
            day=self._calendar_day,
            triggered_event_pks=triggered,
            contracts_changed_pks=contract_delta,
        )
        if self.social_config.persist_snapshots:
            snap.save()
        if snap.pk:
            self.snapshot_pks.append(snap.pk)

        self.recv_message(
            "Environment",
            SimpleMessage(
                message=(
                    f"[end_of_day] closed_day={self._calendar_day} "
                    f"events={triggered or 'none'}"
                )
            ),
        )

        self._calendar_day += 1
        return snap


__all__ = ["SocialSystemEnv", "SocialSystemConfig"]
