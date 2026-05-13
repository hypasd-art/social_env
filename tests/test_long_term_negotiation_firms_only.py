"""``firms_only`` lineup（3+ 名商业参与者，无机构位）的结构性回归测试。

只验证:
1. roster / 默认资源 / scenario loader 对第三、第四名参与者（如 Chen Wei、Dana Rios 对应的 agent）有效；
2. ``LongTermNegotiationEnv`` 能用 3 人 roster 推进 episode 而不崩溃；
3. ``game_metadata.lineup`` 与 ``num_participants`` 的双向序列化（build / parse）一致。

不测「3 方全部签署成功」，因为现成的规则 dummy agent 仅覆盖 bilateral 提案 / accept；
3 方 success 路径需要 LLM 或专门的多边规则 agent。
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
    SESSION_FIRMS_ONLY_ROLE_ORDER,
    default_agent_resources_bundle,
)
from sotopia.settings.long_term_negotiation.llm_evaluation import (
    default_negotiation_roster,
)
from sotopia.settings.long_term_negotiation.scenario_loader import (
    build_negotiation_game_metadata_bundle,
    goal_addon_for_deal_closure_pressure,
    parsed_scenario_from_game_metadata,
)

from tests.negotiation_test_personas import (
    ALL_COMMERCIAL_NAMED_AGENTS,
    ARI_LYNN,
    CHEN_WEI,
    DANA_RIOS,
    MEI_OKADA,
    roster_keys,
)


class FirmsOnlyRosterTest(unittest.TestCase):
    def test_canonical_roster_includes_firm_roles(self) -> None:
        """本仓库长期谈判测试路径只覆盖商业参与者 roster，不在此断言 investor/regulator 用法。"""
        for agent in ALL_COMMERCIAL_NAMED_AGENTS:
            self.assertIn(
                agent.key,
                CANONICAL_NEGOTIATION_ROSTER,
                msg=f"{agent.display_name} ({agent.key})",
            )
        self.assertEqual(FIRM_ROLES_ORDER, roster_keys(*ALL_COMMERCIAL_NAMED_AGENTS))
        self.assertEqual(
            SESSION_FIRMS_ONLY_ROLE_ORDER,
            roster_keys(*ALL_COMMERCIAL_NAMED_AGENTS),
        )

    def test_default_resources_bundle_has_named_late_participants(self) -> None:
        bundle = default_agent_resources_bundle()
        for agent in (CHEN_WEI, DANA_RIOS):
            self.assertIn(agent.key, bundle, msg=agent.display_name)
            self.assertGreater(float(bundle[agent.key]["cash"]), 0.0)

    def test_default_negotiation_roster_for_firms_only(self) -> None:
        roster3 = default_negotiation_roster(
            num_participants=3, lineup=NEGOTIATION_LINEUP_FIRMS_ONLY
        )
        self.assertEqual(roster3, roster_keys(ARI_LYNN, MEI_OKADA, CHEN_WEI))
        roster4 = default_negotiation_roster(
            num_participants=4, lineup=NEGOTIATION_LINEUP_FIRMS_ONLY
        )
        self.assertEqual(
            roster4,
            roster_keys(ARI_LYNN, MEI_OKADA, CHEN_WEI, DANA_RIOS),
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
        self.assertEqual(parsed.roles, roster_keys(ARI_LYNN, MEI_OKADA, CHEN_WEI))

        briefs = gm.get("predefined_news_briefs")
        self.assertIsInstance(briefs, list)
        self.assertGreaterEqual(len(briefs), 2)
        self.assertIn("thread_id", briefs[0])
        self.assertIn("complexity", briefs[0])

        cp = gm.get("deal_closure_pressure")
        self.assertIsInstance(cp, dict)
        self.assertEqual(cp.get("version"), 1)
        pressured = cp.get("pressured_roles")
        self.assertIsInstance(pressured, list)
        self.assertGreaterEqual(len(pressured), 1)
        self.assertLessEqual(len(pressured), 2)
        role_set = set(parsed.roles)
        for r in pressured:
            self.assertIn(r, role_set)
            addon = goal_addon_for_deal_closure_pressure(r, cp)
            self.assertIsInstance(addon, str)
            self.assertGreater(len(addon), 20)
        for r in parsed.roles:
            if r not in pressured:
                self.assertIsNone(goal_addon_for_deal_closure_pressure(r, cp))


class FirmsOnlyEnvSmokeTest(unittest.TestCase):
    def test_three_firm_env_runs_without_crash(self) -> None:
        async def body() -> LongTermNegotiationEnv:
            pol = NegotiationDummyPolicy(
                mode="toward_accept",
                bilateral_participants=roster_keys(ARI_LYNN, MEI_OKADA),
            )
            ag = build_rule_dummy_agents(
                roster_keys(ARI_LYNN, MEI_OKADA, CHEN_WEI), policy=pol
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
        # Chen Wei（第三席）出现在 system_state.agent_keys 中，资源默认现金 > 0。
        self.assertIn(CHEN_WEI.key, env.system_state.agent_keys, msg=CHEN_WEI.display_name)
        self.assertGreater(
            float(env.system_state.agent_resources[CHEN_WEI.key]["cash"]),
            0.0,
            msg=CHEN_WEI.display_name,
        )
        # final-state 指标至少能产出一帧（terminal snapshot 由 env.run_episode_async 写入）。
        self.assertGreater(m.get("negotiation_final_state_n_snapshots", 0.0), 0.0)
        self.assertGreaterEqual(m.get("negotiation_final_state_score", -1.0), 0.0)
        self.assertLessEqual(m.get("negotiation_final_state_score", 2.0), 1.0)


if __name__ == "__main__":
    unittest.main()
