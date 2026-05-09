"""``LongTermNegotiationEnv`` 轻量回归：规则 agent + 规则指标（无需 LLM）。"""

from __future__ import annotations

import asyncio
import unittest

from sotopia.settings import (
    LongTermNegotiationEnv,
    NegotiationDummyPolicy,
    NegotiationTimelineParams,
    build_rule_dummy_agents,
    compute_negotiation_rule_metrics,
)


class LongTermNegotiationSmokeTest(unittest.TestCase):
    def test_bilateral_toward_signature_or_progress(self) -> None:
        async def body() -> tuple[str, LongTermNegotiationEnv]:
            pol = NegotiationDummyPolicy(mode="toward_accept")
            ag = build_rule_dummy_agents(("firm_a", "firm_b"), policy=pol)
            env = LongTermNegotiationEnv(
                ag,
                params=NegotiationTimelineParams(
                    D=8,
                    s_max_per_day=2,
                    max_session_rounds=40,
                    max_total_turns_per_session=80,
                ),
            )
            out = await env.run_episode_async(max_macro_steps=4000)
            return out, env

        outcome, env = asyncio.run(body())
        m = compute_negotiation_rule_metrics(env)
        self.assertEqual(outcome, "success", msg=m)
        self.assertEqual(m.get("negotiation_terminal_is_success"), 1.0)
        self.assertGreater(m.get("negotiation_macro_steps_used", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
