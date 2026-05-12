"""长期商业谈判运行时外壳：编排 ``NegotiationWorldController`` + ``SystemState`` + LLM Agents。

执行轨迹与 ``ParallelSotopiaEnv`` 对齐的部分：

- 继承 ``MessengerMixin``：用 ``recv_message`` 累积 ``inbox``，供可选 ``terminal_evaluators``
  （与 ``parallel.ParallelSotopiaEnv.astep`` 中终局评测相同的 ``__acall__`` + ``unweighted_aggregate_evaluate`` 路径）。
- ``turn_number`` / ``evaluators`` / ``model_name`` 字段与同目录并行实验脚本中的命名习惯一致（中期 ``evaluators`` 预留给与日度 hook 对齐）。

与 ``SocialSystemEnv`` 对齐的部分：

- 将 ``system_state.digest_line(...)``（及谈判 bookkeeping）并入各参与者观测，等价于
  ``SocialSystemEnv._before_return_astep`` 里把 ``digest_line`` 拼到 ``Observation.last_turn`` 的做法。

**评测链中的位置：** LLM 批量/单次评测最终会 ``await LongTermNegotiationEnv.run_episode_async``
（见 ``llm_evaluation.run_llm_negotiation_episode_evaluation``）；本类负责宏观相位循环与各
``NegotiationEpisodeActor.aact`` 调度，实际动作解析落在 ``NegotiationWorldController`` 与 ``controller.parse_agent_action_payload``。
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from sotopia.events import EventEngine, EventEngineConfig
from sotopia.messages import AgentAction, Observation, ScriptEnvironmentResponse, SimpleMessage
from sotopia.messages.messenger import MessengerMixin
from sotopia.state import SystemState

from .agent_state_variables import (
    psych_bundle_from_agent_dicts,
    psych_state_to_prompt_addon,
)
from .controller import NegotiationWorldController, parse_agent_action_payload
from .external_events import NegotiationExternalEventRunner
from .types import NegotiationTimelineParams, Phase

if TYPE_CHECKING:
    from sotopia.envs.evaluators import Evaluator

log = logging.getLogger(__name__)


class NegotiationEpisodeActor(Protocol):
    """具备 ``aact`` 的参与者即可（无需 ``BaseAgent`` / redis 侧 profile）。"""

    async def aact(self, obs: Observation) -> AgentAction: ...


class LongTermNegotiationEnv(MessengerMixin):
    """长期商业谈判（design_1）最小可跑闭环。"""

    def __init__(
        self,
        agents: Mapping[str, NegotiationEpisodeActor],
        *,
        params: NegotiationTimelineParams | None = None,
        system_state: SystemState | None = None,
        initial_resources: dict[str, dict[str, float]] | None = None,
        event_scripts: list[Any] | None = None,
        event_engine_config: EventEngineConfig | None = None,
        strict_design_v1: bool = False,
        external_event_specs: list[dict[str, Any]] | None = None,
        agent_psych_variables: Mapping[str, Mapping[str, Any]] | None = None,
        evaluators: list[Any] | None = None,
        terminal_evaluators: list[Any] | None = None,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        MessengerMixin.__init__(self)
        names = tuple(sorted(agents.keys()))
        if strict_design_v1:
            from .roles import default_agent_resources_bundle, validate_canonical_negotiation_roster

            validate_canonical_negotiation_roster(names)

        self.agents: dict[str, NegotiationEpisodeActor] = dict(agents)
        self.params = params or NegotiationTimelineParams()
        self.ctrl = NegotiationWorldController(names, self.params)
        self.strict_design_v1 = strict_design_v1

        if system_state is not None:
            self.system_state = system_state
        elif initial_resources is not None:
            self.system_state = SystemState(list(names), agent_resources=initial_resources)
        elif strict_design_v1:
            from .roles import default_agent_resources_bundle

            bundle = default_agent_resources_bundle()
            self.system_state = SystemState(
                list(names),
                agent_resources={n: dict(bundle[n]) for n in names if n in bundle},
            )
        else:
            from .roles import default_agent_resources_bundle as _default_bundle

            bundle = _default_bundle()
            self.system_state = SystemState(
                list(names),
                agent_resources={
                    n: (
                        dict(bundle[n])
                        if n in bundle
                        else {"cash": 250.0 if n == "regulator" else 400.0}
                    )
                    for n in names
                },
            )
        self.event_scripts = list(event_scripts or [])
        self.event_engine = EventEngine(event_engine_config or EventEngineConfig())
        specs = (
            list(external_event_specs)
            if external_event_specs is not None
            else list(self.params.external_event_specs)
        )
        self._ext_runner = NegotiationExternalEventRunner.from_spec_dicts(specs)
        self._psych_by_agent = psych_bundle_from_agent_dicts(names, agent_psych_variables)
        #: 最近一次 ``run_episode_async`` 实际宏观步数（供 ``negotiation_metrics`` 读取）
        self.last_episode_macro_steps: int = 0

        #: 与 ``parallel.ParallelSotopiaEnv`` 对齐的评测器槽位（按需 ``from sotopia.envs.evaluators import ...``）
        self.evaluators: list[Any] = list(evaluators or [])
        self.terminal_evaluators: list[Any] = list(terminal_evaluators or [])
        #: 占位：与并行环境字段同名，便于实验配置透传（实际 LLM 调用仍在各 Agent）
        self.model_name: str = model_name
        self.turn_number: int = 0
        self.last_terminal_script_response: ScriptEnvironmentResponse | None = None

    def _ext_tick(
        self,
        timing: str,
        *,
        day: int | None = None,
        slot: int | None = None,
        phase: Phase | None = None,
    ) -> None:
        """§8.2 — 在固定世界节拍上评估外部事件。"""
        d = int(self.ctrl.day if day is None else day)
        s = int(self.ctrl.slot if slot is None else slot)
        ph = self.ctrl.phase if phase is None else phase
        self._ext_runner.tick(
            timing,
            day=d,
            slot=s,
            phase=ph,
            ctrl=self.ctrl,
            state=self.system_state,
        )

    def _digest(self, viewer: str) -> str:
        # 类比 ``SocialSystemEnv._before_return_astep``：在环境文本中附上 system_state 摘要。
        base = self.system_state.digest_line(viewer=viewer)
        extra = self.ctrl.negotiation_context_addon(viewer)
        psych = psych_state_to_prompt_addon(
            self._psych_by_agent.get(viewer),
            expose_threshold=self.params.expose_psych_threshold_in_observation,
        )
        parts = [base]
        if psych.strip():
            parts.append(psych.rstrip())
        if extra.strip():
            parts.append(extra)
        return "\n".join(parts)

    async def _run_terminal_evaluators_like_parallel_astep(self) -> ScriptEnvironmentResponse | None:
        """复用 ``ParallelSotopiaEnv._run_evaluators`` 的 gather + ``unweighted_aggregate_evaluate`` 形态。"""
        from sotopia.envs.evaluators import unweighted_aggregate_evaluate

        if not self.terminal_evaluators:
            return None
        merged = await asyncio.gather(
            *[
                ev.__acall__(turn_number=-1, messages=self.inbox, env=self)
                for ev in self.terminal_evaluators
            ]
        )
        flattened: list[Any] = list(itertools.chain(*merged))
        if not flattened:
            return None
        return unweighted_aggregate_evaluate(flattened)

    def _resources_snapshot(self) -> dict[str, dict[str, float]]:
        return {
            k: dict(self.system_state.agent_resources.get(k, {}))
            for k in self.system_state.agent_keys
        }

    async def run_episode_async(self, max_macro_steps: int = 2000) -> str:
        """推进直到 ``Phase.TERMINATED`` 或步数上限。返回 terminal 原因字符串。"""
        self.ctrl.reset()
        self.ctrl.start_episode()
        self._ext_runner.reset()
        self.turn_number = 0
        self.reset_inbox()
        self.last_terminal_script_response = None
        self.recv_message(
            "Environment",
            SimpleMessage(message="[NegotiationWorld] Episode start."),
        )

        steps = 0
        while not self.ctrl.terminated() and steps < max_macro_steps:
            steps += 1
            ph = self.ctrl.phase
            if ph == Phase.SCHEDULE_INVITE:
                if self.ctrl.slot == 1:
                    self._ext_tick("start_of_day")
                self._ext_tick("before_scheduling")
                await self._macro_sched_invite()
                self.ctrl.finish_invite_phase()
            elif ph == Phase.SCHEDULE_RESPONSE:
                await self._macro_sched_response()
                self.ctrl.resolve_scheduling()
            elif ph == Phase.SESSION:
                await self._macro_session_turn()
            elif ph == Phase.POST_SESSION:
                ps_day, ps_slot = self.ctrl.day, self.ctrl.slot
                self._ext_tick("after_session", day=ps_day, slot=ps_slot, phase=Phase.POST_SESSION)
                pst: list[str] = []
                n_ps_eff = 0
                if self.event_scripts:
                    ps_scripts = self.event_engine.scripts_for_post_session_slot(
                        ps_day, ps_slot, self.event_scripts
                    )
                    n_ps_eff = sum(len(s.effects) for s in ps_scripts)
                    pst = self.event_engine.apply_scripts(ps_scripts, self.system_state)

                self.ctrl.append_event_records(
                    [
                        {
                            "kind": "post_session_slot_scripts",
                            "calendar_day": ps_day,
                            "slot_index": ps_slot,
                            "triggered_script_pks": list(pst),
                            "effects_applied_count": int(n_ps_eff),
                            "note": (
                                "requires EventScript: intraday=True, apply_days includes day,"
                                " step equals slot index k per design_1 §2.1"
                            ),
                        }
                    ]
                )

                self.ctrl.advance_after_post_session()
                self._ext_tick("end_of_slot", day=ps_day, slot=ps_slot, phase=Phase.POST_SESSION)
            elif ph == Phase.END_OF_DAY:
                closing = self.ctrl.day
                self._ext_tick("end_of_day", day=closing, phase=Phase.END_OF_DAY)
                triggered: list[str] = []
                n_eff = 0
                if self.event_scripts:
                    to_apply = self.event_engine.scripts_for_end_of_day(
                        closing, self.event_scripts
                    )
                    n_eff = sum(len(s.effects) for s in to_apply)
                    triggered = self.event_engine.apply_scripts(to_apply, self.system_state)

                resource_copy = {
                    k: dict(self.system_state.agent_resources.get(k, {}))
                    for k in self.system_state.agent_keys
                }
                self.ctrl.append_event_records(
                    [
                        {
                            "kind": "end_of_day_bundle",
                            "closing_calendar_day": closing,
                            "scripts_triggered_pks": list(triggered),
                            "effects_applied_count": int(n_eff),
                        }
                    ]
                )
                self.ctrl.append_state_snapshot(
                    {"label": "after_end_of_day", "day_closed": closing, "agent_resources": resource_copy}
                )

                def _hook(_d: int) -> None:
                    return None

                self.ctrl.end_day_tick(event_hook=_hook)
            elif ph == Phase.INIT:
                self.ctrl.start_episode()
            else:
                break

            self.recv_message(
                "Environment",
                SimpleMessage(message=f"[NegotiationWorld] Macro step #{steps} phase={ph!s}."),
            )

        self.last_episode_macro_steps = int(steps)

        terminal_resource_copy = {
            k: dict(self.system_state.agent_resources.get(k, {}))
            for k in self.system_state.agent_keys
        }
        self.ctrl.append_state_snapshot(
            {
                "label": "after_terminal",
                "day_closed": int(self.ctrl.day),
                "terminal": str(self.ctrl.terminal or "max_steps"),
                "macro_steps": int(steps),
                "agent_resources": terminal_resource_copy,
            }
        )

        self.turn_number = steps
        if self.terminal_evaluators:
            try:
                self.last_terminal_script_response = await self._run_terminal_evaluators_like_parallel_astep()
            except Exception as exc:
                log.warning("terminal_evaluators failed after episode loop: %s", exc)
                self.last_terminal_script_response = None

        return self.ctrl.terminal or "max_steps"

    async def _macro_sched_invite(self) -> None:
        async def one(a: str) -> None:
            obs = self.ctrl.observation_for_scheduling_invite(a, self._digest(a))
            act = await self.agents[a].aact(obs)
            pl = parse_agent_action_payload(act.argument) if act.action_type == "action" else None
            if pl:
                self.ctrl.submit_invite_json(a, pl)
            self.ctrl.scheduling_log.append((self.ctrl.day, self.ctrl.slot, a, act.to_natural_language()))
            self.recv_message(a, act)

        await asyncio.gather(*(one(a) for a in self.ctrl.agent_names))

    async def _macro_sched_response(self) -> None:
        async def one(a: str) -> None:
            obs = self.ctrl.observation_for_scheduling_response(a, self._digest(a))
            act = await self.agents[a].aact(obs)
            pl = parse_agent_action_payload(act.argument) if act.action_type == "action" else None
            if pl:
                self.ctrl.submit_response_json(a, pl)
            self.ctrl.scheduling_log.append((self.ctrl.day, self.ctrl.slot, a, act.to_natural_language()))
            self.recv_message(a, act)

        await asyncio.gather(*(one(a) for a in self.ctrl.agent_names))

    async def _macro_session_turn(self) -> None:
        self.ctrl.ensure_session_structure()
        if self.ctrl.phase != Phase.SESSION:
            return
        actor = self.ctrl.current_actor_in_session()
        if actor is None:
            self.ctrl.ensure_session_structure()
            return
        sid_at_start = self.ctrl.current_session_id()
        if sid_at_start is None:
            return
        obs = self.ctrl.observation_for_session(actor, self._digest(actor))
        act = await self.agents[actor].aact(obs)
        self.recv_message(actor, act)
        self.ctrl.record_session_turn(actor, str(act.action_type), act.to_natural_language())
        if act.action_type == "leave":
            self.ctrl.on_session_turn_completed(actor, session_id=sid_at_start)
            self.ctrl.submit_session_payload(
                actor,
                {"negotiation_op": "session_control", "verb": "leave"},
                resources_snapshot=self._resources_snapshot,
            )
        elif act.action_type == "action":
            pl = parse_agent_action_payload(act.argument)
            if pl:
                self.ctrl.submit_session_payload(
                    actor,
                    pl,
                    resources_snapshot=self._resources_snapshot,
                )
                self._ext_tick("after_formal_action")
            self.ctrl.on_session_turn_completed(actor, session_id=sid_at_start)
        else:
            self.ctrl.on_session_turn_completed(actor, session_id=sid_at_start)
        self.ctrl.advance_session_turn()


__all__ = ["LongTermNegotiationEnv", "NegotiationEpisodeActor"]
