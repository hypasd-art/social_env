"""``firms_only`` lineup（3+ 家公司，无机构位）的结构性回归测试。

只验证:
1. roster / 默认资源 / scenario loader 都对 ``firm_c`` / ``firm_d`` 有效；
2. ``LongTermNegotiationEnv`` 能用 3 家公司 roster 推进 episode 而不崩溃；
3. ``game_metadata.lineup`` 与 ``num_participants`` 的双向序列化（build / parse）一致。

不测「3 家公司全部签署成功」，因为现成的规则 dummy agent 仅覆盖 bilateral 提案 / accept；
3 家公司 success 路径需要 LLM 或专门的多边规则 agent。
"""

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
from sotopia.settings.long_term_negotiation import (
    CANONICAL_NEGOTIATION_ROSTER,
    FIRM_ROLES_ORDER,
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    SESSION_FIRMS_ONLY_ROLE_ORDER,
    default_agent_resources_bundle,
    negotiation_role_order,
)
from sotopia.settings.long_term_negotiation.llm_evaluation import (
    default_negotiation_roster,
)
from sotopia.settings.long_term_negotiation.scenario_loader import (
    build_negotiation_game_metadata_bundle,
    parsed_scenario_from_game_metadata,
)


class FirmsOnlyRosterTest(unittest.TestCase):
    def test_canonical_roster_includes_extra_firms(self) -> None:
        for r in ("firm_a", "firm_b", "firm_c", "firm_d", "investor", "regulator"):
            self.assertIn(r, CANONICAL_NEGOTIATION_ROSTER, msg=r)
        self.assertEqual(
            FIRM_ROLES_ORDER, ("firm_a", "firm_b", "firm_c", "firm_d")
        )
        self.assertEqual(
            SESSION_FIRMS_ONLY_ROLE_ORDER,
            ("firm_a", "firm_b", "firm_c", "firm_d"),
        )

    def test_default_resources_bundle_has_extra_firms(self) -> None:
        bundle = default_agent_resources_bundle()
        for r in ("firm_c", "firm_d"):
            self.assertIn(r, bundle, msg=r)
            self.assertGreater(float(bundle[r]["cash"]), 0.0)

    def test_default_negotiation_roster_for_firms_only(self) -> None:
        roster3 = default_negotiation_roster(
            num_participants=3, lineup=NEGOTIATION_LINEUP_FIRMS_ONLY
        )
        self.assertEqual(roster3, ("firm_a", "firm_b", "firm_c"))
        roster4 = default_negotiation_roster(
            num_participants=4, lineup=NEGOTIATION_LINEUP_FIRMS_ONLY
        )
        self.assertEqual(roster4, ("firm_a", "firm_b", "firm_c", "firm_d"))

    def test_default_negotiation_roster_with_institutional(self) -> None:
        roster3 = default_negotiation_roster(
            num_participants=3, lineup=NEGOTIATION_LINEUP_WITH_INSTITUTIONAL
        )
        self.assertEqual(roster3, ("firm_a", "firm_b", "investor"))
        self.assertEqual(
            negotiation_role_order(NEGOTIATION_LINEUP_WITH_INSTITUTIONAL),
            ("firm_a", "firm_b", "investor", "regulator"),
        )

    def test_game_metadata_lineup_roundtrip(self) -> None:
        params = NegotiationTimelineParams(
            D=6, s_max_per_day=2, max_session_rounds=24, max_total_turns_per_session=48
        )
        gm = build_negotiation_game_metadata_bundle(
            "ltr_firms_only_smoke",
            quartet=False,
            params=params,
            num_participants=3,
            lineup=NEGOTIATION_LINEUP_FIRMS_ONLY,
        )
        self.assertEqual(gm["lineup"], NEGOTIATION_LINEUP_FIRMS_ONLY)
        self.assertEqual(gm["num_participants"], 3)
        self.assertFalse(gm["strict_design_v1"])

        parsed = parsed_scenario_from_game_metadata("env_pk_dummy", gm=gm)
        self.assertEqual(parsed.lineup, NEGOTIATION_LINEUP_FIRMS_ONLY)
        self.assertEqual(parsed.num_participants, 3)
        self.assertEqual(parsed.roles, ("firm_a", "firm_b", "firm_c"))

        briefs = gm.get("predefined_news_briefs")
        self.assertIsInstance(briefs, list)
        self.assertGreaterEqual(len(briefs), 2)
        self.assertIn("thread_id", briefs[0])
        self.assertIn("complexity", briefs[0])


class FirmsOnlyEnvSmokeTest(unittest.TestCase):
    def test_three_firm_env_runs_without_crash(self) -> None:
        async def body() -> LongTermNegotiationEnv:
            pol = NegotiationDummyPolicy(
                mode="toward_accept",
                bilateral_participants=("firm_a", "firm_b"),
            )
            ag = build_rule_dummy_agents(
                ("firm_a", "firm_b", "firm_c"), policy=pol
            )
            env = LongTermNegotiationEnv(
                ag,
                params=NegotiationTimelineParams(
                    D=4,
                    s_max_per_day=1,
                    max_session_rounds=12,
                    max_total_turns_per_session=24,
                ),
            )
            await env.run_episode_async(max_macro_steps=400)
            return env

        env = asyncio.run(body())
        m = compute_negotiation_rule_metrics(env)
        # firm_c 出现在 system_state.agent_keys 中，资源默认现金 > 0。
        self.assertIn("firm_c", env.system_state.agent_keys)
        self.assertGreater(
            float(env.system_state.agent_resources["firm_c"]["cash"]), 0.0
        )
        # final-state 指标至少能产出一帧（terminal snapshot 由 env.run_episode_async 写入）。
        self.assertGreater(m.get("negotiation_final_state_n_snapshots", 0.0), 0.0)
        self.assertGreaterEqual(m.get("negotiation_final_state_score", -1.0), 0.0)
        self.assertLessEqual(m.get("negotiation_final_state_score", 2.0), 1.0)


if __name__ == "__main__":
    unittest.main()
