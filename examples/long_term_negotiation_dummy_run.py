"""最小可跑脚本：``LongTermNegotiationEnv`` + ``NegotiationRuleDummyAgent``（无需 LLM / Redis）。

与 ``sotopia.envs.benchmark_evaluators`` 相同思路：episode 结束后抽取 **规则型指标**
``compute_negotiation_rule_metrics`` ，便于与 baseline 对比。

在项目根下执行::

    cd social_env && PYTHONPATH=. python examples/long_term_negotiation_dummy_run.py

可选：四方 roster + 收口双签（更接近 design_1 §1.1）::

    cd social_env && PYTHONPATH=. python examples/long_term_negotiation_dummy_run.py --quartet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sotopia.messages import AgentAction, Observation
from sotopia.settings import (
    CANONICAL_NEGOTIATION_ROSTER,
    LongTermNegotiationEnv,
    NegotiationDummyPolicy,
    NegotiationTimelineParams,
    build_rule_dummy_agents,
    compute_negotiation_rule_metrics,
)


class LegacySmokeAgent:
    """保留旧两行主体 smoke（仅 propose，不重试 scheduling）。"""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    async def aact(self, obs: Observation) -> AgentAction:
        to_list: list[str] = []
        lt = obs.last_turn
        name = self.agent_name

        if "Scheduling — Invitation round" in lt:
            if name == "firm_a":
                return AgentAction(
                    action_type="action",
                    argument={
                        "negotiation_op": "session_request",
                        "proposed_participants": ["firm_a", "firm_b"],
                        "purpose": "legacy_smoke",
                    },
                    to=to_list,
                )
            return AgentAction(action_type="action", argument={"negotiation_op": "sched_pass"}, to=to_list)

        if "Scheduling — Response round" in lt:
            if name == "firm_b":
                return AgentAction(
                    action_type="action",
                    argument={
                        "negotiation_op": "session_response",
                        "requester": "firm_a",
                        "accept": True,
                    },
                    to=to_list,
                )
            return AgentAction(action_type="action", argument={"negotiation_op": "sched_pass"}, to=to_list)

        if "Active session" in lt and name == "firm_a":
            return AgentAction(
                action_type="action",
                argument={
                    "negotiation_op": "formal",
                    "verb": "propose_contract",
                    "terms": {"price": 100.0, "regulatory_required": 0},
                },
                to=to_list,
            )

        return AgentAction(action_type="none", argument="", to=to_list)


async def _run_legacy() -> tuple[str, LongTermNegotiationEnv]:
    agents = {
        "firm_a": LegacySmokeAgent("firm_a"),
        "firm_b": LegacySmokeAgent("firm_b"),
    }
    env = LongTermNegotiationEnv(
        agents,
        params=NegotiationTimelineParams(D=3, s_max_per_day=2, max_session_rounds=6),
    )
    outcome = await env.run_episode_async(max_macro_steps=500)
    return outcome, env


async def _run_dummy(*, quartet: bool) -> tuple[str, LongTermNegotiationEnv]:
    policy = NegotiationDummyPolicy(mode="toward_accept", propose_terms={"price": 72.0, "regulatory_required": 0})
    if quartet:
        names = tuple(sorted(CANONICAL_NEGOTIATION_ROSTER))
        agents_map = build_rule_dummy_agents(names, policy=policy)
        env = LongTermNegotiationEnv(
            agents_map,
            params=NegotiationTimelineParams(
                D=5,
                s_max_per_day=2,
                max_session_rounds=32,
                max_total_turns_per_session=48,
            ),
            strict_design_v1=True,
        )
    else:
        names = ("firm_a", "firm_b")
        agents_map = build_rule_dummy_agents(names, policy=policy)
        env = LongTermNegotiationEnv(
            agents_map,
            params=NegotiationTimelineParams(D=6, s_max_per_day=2, max_session_rounds=48, max_total_turns_per_session=64),
        )
    outcome = await env.run_episode_async(max_macro_steps=3500)
    return outcome, env


async def _main(mode: str, quartet: bool) -> None:
    if mode == "legacy":
        outcome, env = await _run_legacy()
    else:
        outcome, env = await _run_dummy(quartet=quartet)
    metrics = compute_negotiation_rule_metrics(env)
    print("terminal:", outcome)
    print("metrics:", json.dumps(metrics, indent=2, sort_keys=True))
    print("n_session_log:", len(env.ctrl.session_log))
    print("n_action_log:", len(env.ctrl.action_log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Long-term negotiation env smoke run")
    ap.add_argument(
        "--mode",
        choices=("dummy", "legacy"),
        default="dummy",
        help="dummy=toward_accept 规则 agents；legacy=内联 Minimal 脚本兼容",
    )
    ap.add_argument("--quartet", action="store_true", help="严格四方 roster（strict_design_v1）")
    args = ap.parse_args()
    try:
        asyncio.run(_main(args.mode, args.quartet))
    except Exception as exc:  # pragma: no cover
        print(exc, file=sys.stderr)
        sys.exit(1)
