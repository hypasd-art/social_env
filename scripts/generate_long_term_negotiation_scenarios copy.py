#!/usr/bin/env python
"""批量构造 **长期谈判** 可用的 benchmark-style 场景数据（V1+V2）。

参考 ``sotopia/benchmark_v2_data_models.py`` 的增量模型与工厂函数：

- ``AgentProfile`` → ``upgrade_agent_profile`` → ``AgentProfileV2``
- ``EnvironmentProfile`` → ``upgrade_environment_profile`` → ``EnvironmentProfileV2``
- ``make_event_script_from_dict`` → ``EventScript``
- ``Contract`` / ``SystemStateSnapshot`` / ``make_initial_state_snapshot``

与老 ``scripts/generate_from_scratch.py`` 一致：默认 ``SOTOPIA_STORAGE_BACKEND=local``，
落盘 ``~/.sotopia/data/{AgentProfile,EnvironmentProfile,...}/``. 运行时长期谈判流水线
当前仍主要从代码内 ``NegotiationTimelineParams`` / ``roles.py`` 取规则；这里提供的
``EnvironmentProfile.game_metadata.long_term_negotiation`` + V2 行是为 **采样题库 /
实验管理** 准备的侧车数据，可把 ``pk`` / ``timeline`` JSON 回填到评测入口。

示例::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local python scripts/generate_long_term_negotiation_scenarios.py --clean --tag ltr_benchmark_v1

生成结束后会写入 ``~/.sotopia/data/long_term_negotiation_manifest.json``（含各场景的 ``pk`` / ``codename``）。
用它来跑大模型批量评测（须与上文相同 **local** 后端，否则 Redis OM 与生成数据不一致）::

    cd social_env && SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. \\
      python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \\
      --scenario-manifest ~/.sotopia/data/long_term_negotiation_manifest.json \\
      -m gpt-4o-mini -e gpt-4o-mini -r 1 -b 2 \\
      -o runs/negotiation_from_manifest.jsonl

仅跑题库中某一个 ``EnvironmentProfile`` 时::

    SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \\
      --scenario-env-pk <manifest 里 environments[].pk>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")
LOCAL_DATA_DIR = Path(os.path.expanduser("~/.sotopia/data"))

from sotopia.benchmark_v2_data_models import (  # noqa: E402
    Contract,
    make_event_script_from_dict,
    make_initial_state_snapshot,
    upgrade_agent_profile,
    upgrade_environment_profile,
)
from sotopia.database import (  # noqa: E402
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    RelationshipProfile,
)
from sotopia.database.persistent_profile import EnvironmentList, RelationshipType  # noqa: E402

from sotopia.settings.long_term_negotiation.types import NegotiationTimelineParams  # noqa: E402
from sotopia.settings.long_term_negotiation.roles import (  # noqa: E402
    CANONICAL_NEGOTIATION_ROSTER,
    ROLE_SUMMARY_EN,
    default_agent_resources_bundle,
)

# Negotiation roster order used by ``default_negotiation_roster(quartet=True)``
QUARTET_ROSTER_ORDER: tuple[str, ...] = tuple(sorted(CANONICAL_NEGOTIATION_ROSTER))


def wipe_local_data(*, yes: bool) -> None:
    if not yes:
        return
    if LOCAL_DATA_DIR.exists():
        print(f"[clean] deleting {LOCAL_DATA_DIR}")
        shutil.rmtree(LOCAL_DATA_DIR)
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


def timeline_as_metadata(params: NegotiationTimelineParams) -> dict[str, Any]:
    payload = asdict(params)
    payload["external_event_specs"] = list(payload.get("external_event_specs") or ())
    return payload


def bilateral_timeline_presets() -> list[tuple[str, NegotiationTimelineParams]]:
    return [
        (
            "D6",
            NegotiationTimelineParams(
                D=6,
                s_max_per_day=2,
                max_session_rounds=24,
                max_total_turns_per_session=48,
            ),
        ),
        (
            "D8",
            NegotiationTimelineParams(
                D=8,
                s_max_per_day=2,
                max_session_rounds=32,
                max_total_turns_per_session=64,
            ),
        ),
        (
            "D12",
            NegotiationTimelineParams(
                D=12,
                s_max_per_day=3,
                max_session_rounds=36,
                max_total_turns_per_session=96,
            ),
        ),
    ]


NEGOTIATION_SCENARIO_BODY = (
    "Two commercial parties negotiate a multi-day acquisition timetable with staged formal sessions "
    "and drafting moves. Scheduling is capacity-constrained per calendar day; in-session exchanges "
    "mix natural dialogue with structured negotiation JSON actions."
)

NEGOTIATION_SCENARIO_QUARTET = (
    NEGOTIATION_SCENARIO_BODY
    + " Two additional institutional participants — an external financing counterpart and a "
      "approval authority — may join selected sessions contingent on formal moves."
)


def save_negotiation_agents(*, tag: str) -> dict[str, AgentProfile]:
    """四方各一条 ``AgentProfile``；bilateral 组合只引用 firm_a/firm_b 的 pk。"""
    profiles: dict[str, AgentProfile] = {}
    for role in QUARTET_ROSTER_ORDER:
        party, _, rest = role.partition("_")
        fn = party.title()
        ln = rest.upper() if rest else party.upper()
        ap = AgentProfile(
            first_name=fn[:12],
            last_name=(ln + "Exec")[:20],
            age=42,
            occupation="corporate stakeholder",
            gender="unknown",
            gender_pronoun="they/them",
            public_info=ROLE_SUMMARY_EN.get(role, ""),
            personality_and_values="Professional; participates in scripted long-horizon negotiation simulator.",
            decision_making_style="formal-move friendly; follows calendar/session protocol",
            moral_values=["fairness"],
            schwartz_personal_values=["achievement"],
            big_five="Openness: medium; Conscientiousness: high; Extraversion: medium; "
            "Agreeableness: medium; Neuroticism: medium",
            secret="",
            model_id=f"negotiation-{role}-{tag}",
            tag=tag,
        )
        ap.save()
        profiles[role] = ap
    print(f"[save] AgentProfile (negotiation roster) x {len(profiles)}")
    return profiles


def pairwise_strangers(agents: dict[str, AgentProfile], *, tag: str) -> None:
    roles = QUARTET_ROSTER_ORDER
    n = 0
    for i, a in enumerate(roles):
        for b in roles[i + 1 :]:
            r = RelationshipProfile(
                agent_1_id=agents[a].pk,
                agent_2_id=agents[b].pk,
                relationship=RelationshipType.stranger,
                background_story=f"Neutral bench relationship for negotiation scenario '{tag}'.",
                tag=tag,
            )
            r.save()
            n += 1
    print(f"[save] RelationshipProfile x {n}")


def build_environment_profile_legacy(
    *,
    codename: str,
    quartet: bool,
    params: NegotiationTimelineParams,
    tag: str,
) -> EnvironmentProfile:
    timeline_meta = timeline_as_metadata(params)
    body = NEGOTIATION_SCENARIO_QUARTET if quartet else NEGOTIATION_SCENARIO_BODY
    if quartet:
        goals = [
            f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Secure financing & approvals.",
            f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Maximize lawful consideration.",
            f"<strategy_hint>{ROLE_SUMMARY_EN['investor']}</strategy_hint> Structure contingent capital.",
            f"<strategy_hint>{ROLE_SUMMARY_EN['regulator']}</strategy_hint> Enforce procedural thresholds.",
        ]
    else:
        goals = [
            f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Close deal under cash/financing limits.",
            f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Negotiate staged consideration.",
        ]
    gm: dict[str, Any] = {
        "pipeline": "long_term_negotiation",
        "strict_design_v1": quartet,
        "quartet": quartet,
        "timeline": timeline_meta,
        "design_doc": "social_env/design_1.md",
        "codename": codename,
    }
    return EnvironmentProfile(
        codename=codename,
        source="benchmark_v2_synthetic_long_term_negotiation",
        scenario=body,
        agent_goals=goals,
        relationship=RelationshipType.stranger,
        tag=tag,
        game_metadata=gm,
    )


def save_combo(env: EnvironmentProfile, agent_roles: tuple[str, ...], agents: dict[str, AgentProfile]) -> EnvAgentComboStorage:
    aids = [agents[r].pk for r in agent_roles]
    combo = EnvAgentComboStorage(env_id=env.pk, agent_ids=aids)
    combo.save()
    return combo


def negotiation_event_scripts(tag: str) -> list[Any]:
    ev1 = make_event_script_from_dict(
        {
            "name": "ltr_market_rumor_day2",
            "category": "news",
            "visibility": "public",
            "intraday": False,
            "apply_days": [2],
            "description": "Sector rumor increases perceived financing pressure (portfolio narrative bump).",
            "effects": [
                {"op": "delta", "target": "public_opinion.firm_a", "value": -0.5},
                {"op": "delta", "target": "public_opinion.investor", "value": 0.25},
            ],
            "tag": tag,
        }
    )
    ev2 = make_event_script_from_dict(
        {
            "name": "ltr_policy_calendar_day5",
            "category": "policy",
            "visibility": "partial",
            "intraday": False,
            "apply_days": [5],
            "description": "Regulatory filing window tightened (observe-only macro shock hook).",
            "effects": [
                {"op": "delta", "target": "market_state.regulatory_stringency", "value": 0.05},
            ],
            "tag": tag,
        }
    )
    return [ev1, ev2]


def save_negotiation_agent_profiles_v2(
    agents_by_role: dict[str, AgentProfile],
    *,
    tag: str,
) -> dict[str, Any]:
    """每个谈判角色仅存一条 ``AgentProfileV2``（全场景共用）。"""
    bundle = default_agent_resources_bundle()
    v2: dict[str, Any] = {}
    for role in QUARTET_ROSTER_ORDER:
        rep = (
            float(bundle.get(role, {}).get("institutional_credibility", 50.0) or 50.0)
            if role == "regulator"
            else 50.0
        )
        upgraded = upgrade_agent_profile(
            agents_by_role[role],
            initial_resources={k: float(v) for k, v in dict(bundle.get(role, {"cash": 0.0})).items()},
            initial_reputation=rep,
            risk_preference="neutral",
            role_type=role,
        )
        upgraded.tag = tag
        upgraded.save()
        v2[role] = upgraded
    print(f"[save] AgentProfileV2 x {len(v2)}")
    return v2


def persist_scenario_v2(
    legacy_env: EnvironmentProfile,
    *,
    quartet: bool,
    params: NegotiationTimelineParams,
    tag: str,
    event_anchor_pk: str | None,
    v2_by_role: dict[str, Any],
) -> tuple[Any, Contract | None, Any]:
    """落库单个场景的 ``EnvironmentProfileV2``、``SystemStateSnapshot``、可选 ``Contract``。"""
    n_agents = 4 if quartet else 2
    bundle = default_agent_resources_bundle()
    timeline_meta = timeline_as_metadata(params)

    md_init: dict[str, Any] = {
        **timeline_meta,
        "negotiation_logical_resources_by_role": {k: dict(v) for k, v in bundle.items()},
        "market_state": {"interest_rate": 0.042, "regulatory_stringency": 1.0},
        "pipeline": "long_term_negotiation",
    }

    v2_env = upgrade_environment_profile(
        legacy_env,
        scenario_type="negotiation",
        n_agents=n_agents,
        max_days=params.D,
        intra_day_steps=max(1, params.s_max_per_day * 8),
        event_schedule_pk=event_anchor_pk,
        system_state_init=md_init,
    )
    v2_env.tag = tag
    v2_env.save()

    roles: tuple[str, ...] = QUARTET_ROSTER_ORDER if quartet else ("firm_a", "firm_b")

    pks = [v2_by_role[r].pk for r in roles]

    ms = md_init["market_state"]
    snap = make_initial_state_snapshot(
        episode_pk=f"ltr_placeholder_ep_{legacy_env.codename}_{legacy_env.pk}",
        agent_pks=pks,
        initial_resources_per_agent={v2_by_role[r].pk: dict(bundle[r]) for r in roles},
        initial_reputation_per_agent={
            v2_by_role[r].pk: float(v2_by_role[r].initial_reputation) for r in roles
        },
        market_state=dict(ms),
        resource_pool={"water": 100.0},
    )
    snap.save()

    contract: Contract | None = None
    if quartet and len(pks) >= 3:
        contract = Contract(
            episode_pk="",
            proposer_pk=str(pks[0]),
            counterparties=[str(pks[1]), str(pks[2])],
            contract_type="agreement",
            terms={"subject": "m&a_outline", "horizon_calendar_days": params.D},
            penalty={"cash_delta": -10.0},
            proposed_day=0,
            expiry_day=params.D,
            status="proposed",
        )
        contract.save()

    return v2_env, contract, snap


def save_environment_list_for_combos(envs: list[EnvironmentProfile], combos: dict[str, EnvAgentComboStorage]) -> EnvironmentList:
    environments: list[str] = []
    agent_index: list[str] = []
    for env in envs:
        combo = combos[env.codename]
        n = len(combo.agent_ids)
        for idx in range(n):
            environments.append(env.pk)
            agent_index.append(str(idx))
    el = EnvironmentList(
        name="long_term_negotiation_scenarios",
        environments=environments,
        agent_index=agent_index,
    )
    el.save()
    print(f"[save] EnvironmentList x 1 pk={el.pk}, entries={len(environments)}")
    return el


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--tag", default="ltr_negotiation_bench_v1")
    ap.add_argument(
        "--modes",
        default="bilat,quartet",
        help="comma separated: bilat, quartet (default both)",
    )
    args = ap.parse_args()

    print(f"[backend] SOTOPIA_STORAGE_BACKEND={os.environ['SOTOPIA_STORAGE_BACKEND']}")
    print(f"[paths]   {LOCAL_DATA_DIR}")

    wipe_local_data(yes=args.clean)

    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    if not modes.intersection({"bilat", "quartet"}):
        modes = {"bilat", "quartet"}

    agents = save_negotiation_agents(tag=args.tag)
    pairwise_strangers(agents, tag=args.tag)
    v2_agents = save_negotiation_agent_profiles_v2(agents, tag=args.tag)

    events = negotiation_event_scripts(args.tag)
    for ev in events:
        ev.save()
    anchor_pk = events[0].pk if events else None
    print(f"[save] EventScript x {len(events)} anchor_pk={anchor_pk}")

    presets = bilateral_timeline_presets()

    combos_by_codename: dict[str, EnvAgentComboStorage] = {}
    legacy_env_objs: list[EnvironmentProfile] = []

    variant_i = 0
    if "bilat" in modes:
        for label, params in presets:
            codename = f"ltr_neg_bil_{label}_v{variant_i}"
            variant_i += 1
            legacy = build_environment_profile_legacy(
                codename=codename, quartet=False, params=params, tag=args.tag
            )
            legacy.save()
            combo = save_combo(legacy, ("firm_a", "firm_b"), agents)
            combos_by_codename[codename] = combo
            legacy_env_objs.append(legacy)
            persist_scenario_v2(
                legacy,
                quartet=False,
                params=params,
                tag=args.tag,
                event_anchor_pk=anchor_pk,
                v2_by_role=v2_agents,
            )

    if "quartet" in modes:
        for label, params in presets:
            codename = f"ltr_neg_quad_{label}_v{variant_i}"
            variant_i += 1
            legacy = build_environment_profile_legacy(
                codename=codename, quartet=True, params=params, tag=args.tag
            )
            legacy.save()
            combo = save_combo(legacy, QUARTET_ROSTER_ORDER, agents)
            combos_by_codename[codename] = combo
            legacy_env_objs.append(legacy)
            persist_scenario_v2(
                legacy,
                quartet=True,
                params=params,
                tag=args.tag,
                event_anchor_pk=anchor_pk,
                v2_by_role=v2_agents,
            )

    save_environment_list_for_combos(legacy_env_objs, combos_by_codename)

    manifest = {
        "tag": args.tag,
        "agent_roles": QUARTET_ROSTER_ORDER,
        "agent_profile_pks_by_role": {r: agents[r].pk for r in QUARTET_ROSTER_ORDER},
        "agent_profile_v2_pks_by_role": {r: v2_agents[r].pk for r in QUARTET_ROSTER_ORDER},
        "environments": [{"codename": e.codename, "pk": e.pk} for e in legacy_env_objs],
    }
    manifest_path = LOCAL_DATA_DIR / "long_term_negotiation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] manifest {manifest_path}")

    print("\n========== DONE ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
