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
from sotopia.settings.long_term_negotiation import compute_negotiation_final_state_metrics
from sotopia.settings.long_term_negotiation.negotiation_metrics import (
    build_rule_evaluation_state_record,
)

from tests.negotiation_test_personas import ARI_LYNN, MEI_OKADA, roster_keys


class LongTermNegotiationSmokeTest(unittest.TestCase):
    def test_bilateral_toward_signature_or_progress(self) -> None:
        async def body() -> tuple[str, LongTermNegotiationEnv]:
            pol = NegotiationDummyPolicy(mode="toward_accept")
            ag = build_rule_dummy_agents(roster_keys(ARI_LYNN, MEI_OKADA), policy=pol)
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

        # final intermediate-state score should be merged into rule_metrics.
        self.assertIn("negotiation_final_state_score", m)
        self.assertGreaterEqual(m["negotiation_final_state_score"], 0.0)
        self.assertLessEqual(m["negotiation_final_state_score"], 1.0)
        self.assertGreater(m.get("negotiation_final_state_n_snapshots", 0.0), 0.0)
        # parity: standalone helper produces identical numbers.
        sub = compute_negotiation_final_state_metrics(env)
        for k, v in sub.items():
            self.assertEqual(m[k], v, msg=k)
        # in a successful bilateral run, the success component must be active.
        self.assertEqual(
            m["negotiation_final_state_score_component_terminal_success"], 0.3
        )

        rec = build_rule_evaluation_state_record(env)
        self.assertIn("state_snapshot_for_rule_metrics", rec)
        snap = rec["state_snapshot_for_rule_metrics"]
        self.assertIsNotNone(snap)
        self.assertEqual(snap.get("label"), "after_terminal")
        self.assertIn("agent_resources", snap)
if __name__ == "__main__":
    unittest.main()
