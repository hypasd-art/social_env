"""与设计文档对齐的 **规则型** negotiation 参与者（评测 / 冒烟 / CI），不依赖 LLM 与 Redis。

与 ``NegotiationEpisodeActor`` / ``BaseAgent.aact`` 接口一致（仅 ``agent_name`` + ``aact``），
可被 ``LongTermNegotiationEnv``、或在测试里与 ``benchmark_evaluators`` 的规则指标并联使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sotopia.messages import AgentAction, Observation


_WORLD_BANNER = re.compile(
    r"\[NegotiationWorld\]\s+day=(?P<d>\d+)/\d+\s+remaining_days=\d+\s+slot=(?P<s>\d+)/\d+",
)


def _sync_day_slot_banner(shared: "NegotiationDummySharedState", last_turn_text: str) -> None:
    m = _WORLD_BANNER.search(last_turn_text)
    if not m:
        return
    dsk = (int(m.group("d")), int(m.group("s")))
    if shared.last_seen_calendar_slot != dsk:
        shared.last_seen_calendar_slot = dsk
        shared.invite_submitted_this_slot = False


def _schedule_pass() -> AgentAction:
    return AgentAction(
        action_type="action",
        argument={"negotiation_op": "sched_pass"},
        to=[],
    )


def _noop() -> AgentAction:
    return AgentAction(action_type="none", argument="", to=[])


def _looks_like_invite_phase(lt: str) -> bool:
    return "Scheduling — Invitation round" in lt


def _looks_like_response_phase(lt: str) -> bool:
    return "Scheduling — Response round" in lt


def _in_active_session(lt: str) -> bool:
    return "Active session" in lt


@dataclass
class NegotiationDummyPolicy:
    """单场景 smoke / 收口用的默认策略旋钮。"""

    invite_requester: str = "firm_a"
    bilateral_participants: tuple[str, ...] = ("firm_a", "firm_b")
    purpose: str = "dummy_episode"
    response_acceptor: str = "firm_b"
    propose_terms: dict[str, object] = field(
        default_factory=lambda: {"price": 80.0, "regulatory_required": 0}
    )
    #: ``minimal``：仅我方发起 propose 一次（旧 smoke）；``toward_accept``：propose→accept→双签力求 ``success``
    mode: str = "toward_accept"
    extra_scheduling_pass_always: bool = True


@dataclass
class NegotiationDummySharedState:
    """跨 agent 共享的极简阶段位（随 day/slot 自动复位 invite 令牌）。"""

    invite_submitted_this_slot: bool = False
    last_seen_calendar_slot: tuple[int, int] | None = None


class NegotiationRuleDummyAgent:
    """规则型 negotiation 参与者。（共享 ``NegotiationDummyPolicy`` + ``NegotiationDummySharedState``。）"""

    __slots__ = ("agent_name", "policy", "shared")

    def __init__(
        self,
        agent_name: str,
        *,
        policy: NegotiationDummyPolicy | None = None,
        shared: NegotiationDummySharedState | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.policy = policy or NegotiationDummyPolicy()
        self.shared = shared if shared is not None else NegotiationDummySharedState()

    async def aact(self, obs: Observation) -> AgentAction:
        lt = obs.last_turn
        _sync_day_slot_banner(self.shared, lt)
        name = self.agent_name

        if _looks_like_invite_phase(lt):
            if name != self.policy.invite_requester:
                return _schedule_pass() if self.policy.extra_scheduling_pass_always else _noop()
            fp = tuple(sorted(frozenset(self.policy.bilateral_participants)))
            if len(fp) < 2:
                return _schedule_pass()
            if self.shared.invite_submitted_this_slot:
                return _schedule_pass()
            self.shared.invite_submitted_this_slot = True
            return AgentAction(
                action_type="action",
                argument={
                    "negotiation_op": "session_request",
                    "proposed_participants": list(fp),
                    "purpose": self.policy.purpose,
                },
                to=[],
            )

        if _looks_like_response_phase(lt):
            if name != self.policy.response_acceptor:
                return _schedule_pass()
            rq = self.policy.invite_requester
            return AgentAction(
                action_type="action",
                argument={
                    "negotiation_op": "session_response",
                    "requester": rq,
                    "accept": True,
                },
                to=[],
            )

        if _in_active_session(lt):
            if self.policy.mode not in ("minimal", "toward_accept"):
                return _noop()
            invited = frozenset(self.policy.bilateral_participants)
            if name not in invited:
                return _noop()

            if self.policy.mode == "minimal":
                if name == self.policy.invite_requester:
                    return AgentAction(
                        action_type="action",
                        argument={
                            "negotiation_op": "formal",
                            "verb": "propose_contract",
                            "terms": dict(self.policy.propose_terms),
                        },
                        to=[],
                    )
                return _noop()

            # toward_accept — digest 末尾含 ``NegotiationWorld`` / ``contracts_visible_to_you``
            if "status='signed'" in lt or 'status="signed"' in lt or "WORLD_TERMINAL: success" in lt:
                return _noop()
            if "status='accepted'" in lt or 'status="accepted"' in lt:
                return AgentAction(
                    action_type="action",
                    argument={"negotiation_op": "formal", "verb": "sign"},
                    to=[],
                )
            if "status='proposed'" in lt or 'status="proposed"' in lt:
                # 双方主体可能均需 ``accept`` 才会进入 ``accepted``；让当前说话人一律尝试 accept（多余 accept 会被控制器静音拒绝）
                return AgentAction(
                    action_type="action",
                    argument={"negotiation_op": "formal", "verb": "accept"},
                    to=[],
                )
            if name == self.policy.invite_requester:
                return AgentAction(
                    action_type="action",
                    argument={
                        "negotiation_op": "formal",
                        "verb": "propose_contract",
                        "terms": dict(self.policy.propose_terms),
                    },
                    to=[],
                )
            return _noop()

        return _noop()


def build_rule_dummy_agents(
    agent_names: tuple[str, ...],
    *,
    policy: NegotiationDummyPolicy | None = None,
) -> dict[str, NegotiationRuleDummyAgent]:
    """按 roster 构造一组共享 policy/state 的规则 agent。"""
    p = policy or NegotiationDummyPolicy()
    shared = NegotiationDummySharedState()
    return {n: NegotiationRuleDummyAgent(n, policy=p, shared=shared) for n in agent_names}


__all__ = [
    "NegotiationDummyPolicy",
    "NegotiationDummySharedState",
    "NegotiationRuleDummyAgent",
    "build_rule_dummy_agents",
]
