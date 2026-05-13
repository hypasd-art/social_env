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
import re
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
from .roles import default_display_name_for_role
from .types import NegotiationTimelineParams, Phase

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
        predefined_outcome_rule: dict[str, Any] | None = None,
        agent_display_names: Mapping[str, str] | None = None,
    ) -> None:
        MessengerMixin.__init__(self)
        names = tuple(sorted(agents.keys()))
        if strict_design_v1:
            from .roles import default_agent_resources_bundle, validate_canonical_negotiation_roster

            validate_canonical_negotiation_roster(names)

        self.agents: dict[str, NegotiationEpisodeActor] = dict(agents)
        self.params = params or NegotiationTimelineParams()
        overlay = dict(agent_display_names or {})
        self.agent_display_names: dict[str, str] = {}
        for n in names:
            ag = self.agents.get(n)
            prof = getattr(ag, "profile", None) if ag is not None else None
            nm = ""
            if prof is not None:
                nm = f"{getattr(prof, 'first_name', '')} {getattr(prof, 'last_name', '')}".strip()
            self.agent_display_names[n] = overlay.get(n) or (nm if nm else default_display_name_for_role(n))

        self.ctrl = NegotiationWorldController(
            names, self.params, agent_display_names=self.agent_display_names
        )
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
        self.predefined_outcome_rule: dict[str, Any] = (
            dict(predefined_outcome_rule) if isinstance(predefined_outcome_rule, dict) else {}
        )
        self._contract_status_settlement_applied: bool = False

        for _role, _ag in self.agents.items():
            _bind = getattr(_ag, "bind_episode_display_names", None)
            if callable(_bind):
                _bind(dict(self.agent_display_names))

    def _apply_contract_status_settlement_if_needed(self) -> None:
        """按主合同状态自动结算，并把收益写回 ``SystemState.agent_resources``。"""
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

        from .negotiation_metrics import compute_predefined_rule_settlement_by_contract_status

        metrics = compute_predefined_rule_settlement_by_contract_status(
            env=self,
            predefined_outcome_rule=self.predefined_outcome_rule,
            contract_status=status,
        )

        settlement_by_agent: dict[str, float] = {}
        for agent in self.system_state.agent_keys:
            ind_key = f"negotiation_predefined_rule_individual_profit_{agent}"
            comp_key = f"negotiation_predefined_rule_company_profit_{agent}"
            if ind_key in metrics:
                payout = float(metrics.get(ind_key, 0.0) or 0.0)
            else:
                payout = float(metrics.get(comp_key, 0.0) or 0.0)
            if payout == 0.0:
                continue
            res = self.system_state.agent_resources.setdefault(agent, {})
            cash0 = float(res.get("cash", 0.0) or 0.0)
            res["cash"] = cash0 + payout
            settlement_by_agent[agent] = payout

        self._contract_status_settlement_applied = True
        if settlement_by_agent:
            self.ctrl.record_execution_event(
                "contract_settlement_applied",
                "已按主合同状态与 predefined_outcome_rule 自动结算并更新现金。",
                primary_contract_status=status,
                settlement_by_agent=settlement_by_agent,
                total_settlement=float(sum(settlement_by_agent.values())),
            )

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

    def _rewrite_digest_line_display_names(self, line: str) -> str:
        """``[system]`` digest 中 trust 矩阵的 canonical 键换成人名，便于阅读。"""
        out = line
        for cid in sorted(self.agent_display_names.keys(), key=len, reverse=True):
            disp = self.agent_display_names.get(cid, cid)
            if not disp or disp == cid:
                continue
            out = re.sub(r"\b" + re.escape(cid) + r":", disp + ":", out)
        return out

    def _digest(self, viewer: str) -> str:
        # 类比 ``SocialSystemEnv._before_return_astep``：在环境文本中附上 system_state 摘要。
        base = self._rewrite_digest_line_display_names(self.system_state.digest_line(viewer=viewer))
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
        self._contract_status_settlement_applied = False
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
        self._apply_contract_status_settlement_if_needed()

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
