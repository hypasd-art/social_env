#!/usr/bin/env python
"""完全从零造数据：不依赖 Redis、不依赖 export/*.json。

跑完之后 ``~/.sotopia/data/`` 下会有完整的 V1 + V2 数据闭环：

V1（老 sotopia 流水线）：
    AgentProfile/             N 个手写角色卡
    EnvironmentProfile/       M 个手写场景
    RelationshipProfile/      角色对之间的关系
    EnvAgentComboStorage/     M*K 个 (env, [agent1, agent2]) 组合
    EnvironmentList/          一条名为 "scratch_env_set" 的列表

V2（长周期 benchmark 流水线，可选 --with-v2）：
    AgentProfileV2/           升级后的 V2 角色卡
    EnvironmentProfileV2/     升级后的 V2 场景
    EventScript/              端日触发的外部事件
    Contract/                 合约模板
    SystemStateSnapshot/      day=0 初始状态

用法
----
    # 1) 完全重置 + 造一套小规模数据（推荐第一次跑）
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_from_scratch.py --clean --with-v2

    # 2) 自定规模，不删旧数据（增量）
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_from_scratch.py \\
        --n-agents 12 --n-envs 6 --combos-per-env 4 --tag scratch_v1

    # 3) 让 'sotopia benchmark --task hard' 也能直接用：覆盖 hard 名单 pk
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_from_scratch.py --clean --override-hard-list

跑完之后
--------
    ls ~/.sotopia/data/

    # 用 'scratch' 任务名跑 benchmark（走 else 分支，吃全部 EnvAgentComboStorage）
    SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \\
        --models gpt-4o-mini --partner-model gpt-4o-mini \\
        --evaluator-model gpt-4o-mini --task scratch --tag run0
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")
LOCAL_DATA_DIR = Path(os.path.expanduser("~/.sotopia/data"))

from sotopia.database import (  # noqa: E402
    AgentProfile,
    EnvironmentProfile,
    EnvAgentComboStorage,
    RelationshipProfile,
)
from sotopia.database.persistent_profile import (  # noqa: E402
    EnvironmentList,
    RelationshipType,
)


# ---------------------------------------------------------------------------
# 1) 角色档案库（手写，不调 LLM）
# ---------------------------------------------------------------------------

AGENT_ARCHETYPES: list[dict[str, Any]] = [
    {
        "first_name": "Mia",
        "last_name": "Chen",
        "age": 29,
        "occupation": "software engineer",
        "gender": "Woman",
        "gender_pronoun": "she/her",
        "big_five": "Openness: high; Conscientiousness: high; Extraversion: medium; Agreeableness: medium; Neuroticism: low",
        "moral_values": ["fairness", "loyalty"],
        "schwartz_personal_values": ["achievement", "self-direction"],
        "personality_and_values": "Analytical and pragmatic; values fairness over efficiency in interpersonal trade-offs.",
        "decision_making_style": "data-driven, risk-averse",
        "public_info": "Active open-source contributor; enjoys rock climbing on weekends.",
        "secret": "Recently turned down a much higher-paying job offer to stay close to family.",
        "mbti": "INTJ",
    },
    {
        "first_name": "William",
        "last_name": "Brown",
        "age": 41,
        "occupation": "small business owner",
        "gender": "Man",
        "gender_pronoun": "he/him",
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: high; Agreeableness: low; Neuroticism: medium",
        "moral_values": ["liberty", "loyalty"],
        "schwartz_personal_values": ["power", "security"],
        "personality_and_values": "Driven and competitive; willing to break minor norms to protect family interests.",
        "decision_making_style": "intuitive, opportunistic",
        "public_info": "Owns a chain of three coffee shops downtown.",
        "secret": "One of his suppliers is selling him stock that turned out to be smuggled.",
        "mbti": "ESTP",
    },
    {
        "first_name": "Aiden",
        "last_name": "Patel",
        "age": 23,
        "occupation": "graduate student",
        "gender": "Man",
        "gender_pronoun": "he/him",
        "big_five": "Openness: high; Conscientiousness: medium; Extraversion: low; Agreeableness: high; Neuroticism: high",
        "moral_values": ["care", "fairness"],
        "schwartz_personal_values": ["benevolence", "universalism"],
        "personality_and_values": "Curious and idealistic; struggles to assert self-interest in negotiations.",
        "decision_making_style": "deliberative, conflict-averse",
        "public_info": "Researches climate policy; volunteers at a food bank.",
        "secret": "Hasn't told his advisor he's considering dropping out.",
        "mbti": "INFP",
    },
    {
        "first_name": "Sophia",
        "last_name": "Garcia",
        "age": 35,
        "occupation": "ER nurse",
        "gender": "Woman",
        "gender_pronoun": "she/her",
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: medium; Agreeableness: high; Neuroticism: medium",
        "moral_values": ["care", "fairness"],
        "schwartz_personal_values": ["security", "benevolence"],
        "personality_and_values": "Calm under pressure; deeply protective of those she perceives as vulnerable.",
        "decision_making_style": "checklist-driven, escalates quickly when stakes rise",
        "public_info": "Single mother of two; trains junior nurses on weekends.",
        "secret": "Recently tested positive for a chronic illness she hasn't disclosed at work.",
        "mbti": "ISFJ",
    },
    {
        "first_name": "Noah",
        "last_name": "Kim",
        "age": 47,
        "occupation": "investment banker",
        "gender": "Man",
        "gender_pronoun": "he/him",
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: high; Agreeableness: low; Neuroticism: low",
        "moral_values": ["liberty", "achievement"],
        "schwartz_personal_values": ["power", "achievement"],
        "personality_and_values": "Cool, transactional; treats relationships as portfolios.",
        "decision_making_style": "expected-value calculation, long horizon",
        "public_info": "Sits on the board of two startups; teaches Sunday finance class.",
        "secret": "Has been quietly liquidating his stake in his own firm.",
        "mbti": "ENTJ",
    },
    {
        "first_name": "Ava",
        "last_name": "Robinson",
        "age": 19,
        "occupation": "barista",
        "gender": "Non-binary",
        "gender_pronoun": "they/them",
        "big_five": "Openness: high; Conscientiousness: low; Extraversion: high; Agreeableness: high; Neuroticism: medium",
        "moral_values": ["liberty", "care"],
        "schwartz_personal_values": ["self-direction", "stimulation"],
        "personality_and_values": "Spontaneous and warm; chronically underprepared but well-liked.",
        "decision_making_style": "feeling-driven, present-biased",
        "public_info": "Studies fine arts part-time; runs a queer poetry zine.",
        "secret": "Owes their roommate three months of rent.",
        "mbti": "ENFP",
    },
    {
        "first_name": "Lucas",
        "last_name": "Müller",
        "age": 53,
        "occupation": "city council policy advisor",
        "gender": "Man",
        "gender_pronoun": "he/him",
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: medium; Agreeableness: medium; Neuroticism: low",
        "moral_values": ["fairness", "authority"],
        "schwartz_personal_values": ["tradition", "security"],
        "personality_and_values": "Procedural and patient; insists on transparent rules even when they hurt allies.",
        "decision_making_style": "principled, slow-moving",
        "public_info": "PhD in public administration; chairs the public housing review board.",
        "secret": "Once accepted a small gift from a developer he later regulated.",
        "mbti": "ISTJ",
    },
    {
        "first_name": "Zara",
        "last_name": "Okafor",
        "age": 31,
        "occupation": "freelance journalist",
        "gender": "Woman",
        "gender_pronoun": "she/her",
        "big_five": "Openness: high; Conscientiousness: medium; Extraversion: high; Agreeableness: medium; Neuroticism: medium",
        "moral_values": ["fairness", "liberty"],
        "schwartz_personal_values": ["universalism", "stimulation"],
        "personality_and_values": "Probing and direct; will burn a source for a story she believes in.",
        "decision_making_style": "narrative-driven, willing to defer payoff for impact",
        "public_info": "Investigates municipal corruption; podcast host.",
        "secret": "A current source is also a personal friend.",
        "mbti": "ENTP",
    },
]


# ---------------------------------------------------------------------------
# 2) 场景库（手写，不调 LLM）
# ---------------------------------------------------------------------------

ENVIRONMENT_ARCHETYPES: list[dict[str, Any]] = [
    {
        "codename": "shared_apartment_repair",
        "source": "scratch_manual",
        "scenario": (
            "Two roommates share an apartment. The dishwasher broke last week. "
            "Agent1 paid the full $400 repair bill on their card and now wants Agent2 "
            "to chip in, but the lease names only Agent1 on the appliance warranty."
        ),
        "agent_goals": [
            "<extra_info>You paid the full $400 repair bill out of pocket and feel agent2 should split it 50/50. "
            "You also feel that agent2 has been less considerate about cleanliness recently.</extra_info> "
            "Convince agent2 to pay you back at least $200, ideally without escalating tension.",
            "<extra_info>You think the repair bill was avoidable if agent1 had not loaded the dishwasher with hard plastic. "
            "You're already short on cash this month due to a medical bill.</extra_info> "
            "Pay as little as possible (ideally under $80) without seriously damaging the relationship.",
        ],
        "relationship": int(RelationshipType.acquaintance),
        "age_constraint": "[(18, 60), (18, 60)]",
        "occupation_constraint": None,
    },
    {
        "codename": "side_project_equity",
        "source": "scratch_manual",
        "scenario": (
            "Two coworkers built a small SaaS product on weekends. It's starting to "
            "make money and the question of equity split has come up. Agent1 wrote 70% "
            "of the code; agent2 brought in 100% of the paying customers."
        ),
        "agent_goals": [
            "<extra_info>You've been working nights for 9 months while agent2 just made a few sales calls. "
            "You believe equity should reflect technical contribution.</extra_info> "
            "Push for at least a 65/35 split in your favor.",
            "<extra_info>Without your sales the product would have zero revenue. You also brought a key enterprise client. "
            "</extra_info> "
            "Get an even 50/50 split, or 55/45 in your favor at most.",
        ],
        "relationship": int(RelationshipType.know_by_name),
        "age_constraint": "[(22, 55), (22, 55)]",
        "occupation_constraint": None,
    },
    {
        "codename": "drought_well_sharing",
        "source": "scratch_manual",
        "scenario": (
            "A small village has one shared well. A drought has cut its output to 60% of normal. "
            "Agent1 needs water for crops; agent2 needs water for livestock. "
            "If neither yields, the well will be over-drawn and dry up entirely."
        ),
        "agent_goals": [
            "<extra_info>Without irrigation your crops will fail this season — that's your whole income.</extra_info> "
            "Secure at least 60% of the well output, or a written agreement to that effect.",
            "<extra_info>Your livestock will die without water; selling them at a loss would also ruin your year.</extra_info> "
            "Secure at least 60% of the well output, or a written agreement to that effect.",
        ],
        "relationship": int(RelationshipType.acquaintance),
        "age_constraint": "[(25, 65), (25, 65)]",
        "occupation_constraint": None,
    },
    {
        "codename": "whistleblower_dilemma",
        "source": "scratch_manual",
        "scenario": (
            "A journalist (agent1) has discovered that a public official (agent2) accepted "
            "a small gift from a developer. The story would damage the official's career. "
            "They meet in private at the official's request."
        ),
        "agent_goals": [
            "<extra_info>You believe the public deserves to know. The official has otherwise had a clean record.</extra_info> "
            "Either get the official's on-record acknowledgment, or convince yourself the story isn't worth running.",
            "<extra_info>You returned the gift the next day and immediately disclosed it to ethics counsel — but the journalist may not know that.</extra_info> "
            "Avoid having a damaging story published, but without lying or offering anything improper in exchange.",
        ],
        "relationship": int(RelationshipType.know_by_name),
        "age_constraint": "[(25, 65), (25, 65)]",
        "occupation_constraint": None,
    },
    {
        "codename": "rent_late_negotiation",
        "source": "scratch_manual",
        "scenario": (
            "Agent1 owes agent2 (their roommate) three months of rent — about $2400. "
            "Agent2 has been quiet about it but is now asking for a clear payback plan."
        ),
        "agent_goals": [
            "<extra_info>You make ~$1100/month after taxes from your part-time job. You feel guilty but also overwhelmed.</extra_info> "
            "Agree to a payback plan you can actually afford — ideally under $200/month.",
            "<extra_info>You've been carrying both rents for 3 months and your savings are running thin.</extra_info> "
            "Get a concrete payback plan that finishes within 6 months.",
        ],
        "relationship": int(RelationshipType.friend),
        "age_constraint": "[(18, 50), (18, 50)]",
        "occupation_constraint": None,
    },
    {
        "codename": "policy_grant_priority",
        "source": "scratch_manual",
        "scenario": (
            "A city has $500K of one-time grant money. Agent1 wants it spent on emergency "
            "shelter capacity; agent2 wants it spent on long-term affordable housing construction."
        ),
        "agent_goals": [
            "<extra_info>Three of your patients last month were unhoused and turned away from full shelters.</extra_info> "
            "Secure at least $300K for shelter capacity expansion.",
            "<extra_info>Shelter beds don't fix the underlying shortage; a $500K seed could unlock $4M in matched funds for new construction.</extra_info> "
            "Keep at least $400K earmarked for long-term construction.",
        ],
        "relationship": int(RelationshipType.acquaintance),
        "age_constraint": "[(28, 70), (28, 70)]",
        "occupation_constraint": None,
    },
]


# ---------------------------------------------------------------------------
# 3) 落库工具
# ---------------------------------------------------------------------------


def wipe_local_data(yes: bool = True) -> None:
    if not yes:
        return
    if LOCAL_DATA_DIR.exists():
        print(f"[clean] 删除 {LOCAL_DATA_DIR}")
        shutil.rmtree(LOCAL_DATA_DIR)
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_agents(specs: list[dict[str, Any]], *, tag: str) -> list[Any]:
    objs = []
    for spec in specs:
        a = AgentProfile(**spec, tag=tag)
        a.save()
        objs.append(a)
    print(f"[save] AgentProfile x {len(objs)}")
    return objs


def save_envs(specs: list[dict[str, Any]], *, tag: str) -> list[Any]:
    objs = []
    for spec in specs:
        # tag 字段 EnvironmentProfile 没有；改写在 codename suffix 上即可
        e = EnvironmentProfile(**spec)
        e.save()
        objs.append(e)
    print(f"[save] EnvironmentProfile x {len(objs)}")
    return objs


def save_relationships(agents: list[Any], *, tag: str) -> list[Any]:
    """对所有不同的 (a_i, a_j) 对，造一条 stranger 关系，用于 ConstraintBasedSampler。"""

    objs = []
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            r = RelationshipProfile(
                agent_1_id=a.pk,
                agent_2_id=b.pk,
                relationship=int(RelationshipType.stranger),
                background_story=f"{a.first_name} and {b.first_name} have not met before.",
                tag=tag,
            )
            r.save()
            objs.append(r)
    print(f"[save] RelationshipProfile x {len(objs)}")
    return objs


def save_combos(
    envs: list[Any],
    agents: list[Any],
    *,
    combos_per_env: int,
    seed: int = 42,
) -> list[Any]:
    """对每个 env 随机抽 combos_per_env 对 agent 形成 EnvAgentComboStorage。"""

    rng = random.Random(seed)
    objs = []
    for e in envs:
        used_pairs: set[tuple[str, str]] = set()
        for _ in range(combos_per_env):
            for _try in range(20):
                a, b = rng.sample(agents, 2)
                key = tuple(sorted([a.pk, b.pk]))
                if key not in used_pairs:
                    used_pairs.add(key)
                    break
            combo = EnvAgentComboStorage(env_id=e.pk, agent_ids=[a.pk, b.pk])
            combo.save()
            objs.append(combo)
    print(f"[save] EnvAgentComboStorage x {len(objs)}")
    return objs


def save_environment_list(
    envs: list[Any],
    combos: list[Any],
    *,
    name: str = "scratch_env_set",
    override_hard_pk: bool = False,
) -> Any:
    """为每个 env 至少挑一个 combo（取首个），生成 EnvironmentList。

    - 默认随机 pk
    - override_hard_pk=True 时强制 pk='01HAK34YPB1H1RWXQDASDKHSNS'，
      让 sotopia benchmark --task hard 能直接吃这份名单
    """

    env_to_combo: dict[str, Any] = {}
    for c in combos:
        env_to_combo.setdefault(c.env_id, c)

    environments: list[str] = []
    agent_index: list[str] = []
    for e in envs:
        c = env_to_combo.get(e.pk)
        if c is None:
            continue
        environments.append(e.pk)
        agent_index.append("0")  # 让测试模型扮演 agent_ids[0]
        environments.append(e.pk)
        agent_index.append("1")  # 让测试模型扮演 agent_ids[1]

    kwargs: dict[str, Any] = {
        "name": name,
        "environments": environments,
        "agent_index": agent_index,
    }
    if override_hard_pk:
        kwargs["pk"] = "01HAK34YPB1H1RWXQDASDKHSNS"

    el = EnvironmentList(**kwargs)
    el.save()
    print(
        f"[save] EnvironmentList(name='{name}', pk={el.pk}, "
        f"#env={len(environments)})"
    )
    return el


# ---------------------------------------------------------------------------
# 4) 可选：V2 升级层（复用 generate_v2_seed.py 的逻辑）
# ---------------------------------------------------------------------------


def maybe_build_v2(agents: list[Any], envs: list[Any], *, tag: str) -> None:
    from sotopia.benchmark_v2_data_models import (
        Contract,
        upgrade_agent_profile,
        upgrade_environment_profile,
        make_initial_state_snapshot,
        make_event_script_from_dict,
    )

    v2_agents = [
        upgrade_agent_profile(
            a,
            initial_resources={"cash": 100.0, "energy": 100.0},
            initial_reputation=50.0,
            risk_preference="neutral",
            role_type="buyer",
        )
        for a in agents
    ]
    for v in v2_agents:
        v.tag = tag
        v.save()
    print(f"[save] AgentProfileV2 x {len(v2_agents)}")

    v2_envs = [
        upgrade_environment_profile(
            e,
            scenario_type="generic",
            n_agents=2,
            max_days=2,
            intra_day_steps=4,
            system_state_init={
                "market_state": {"interest_rate": 0.05, "price_index": 100.0},
                "resource_pool": {"water": 100.0},
            },
        )
        for e in envs
    ]
    for v in v2_envs:
        v.save()
    print(f"[save] EnvironmentProfileV2 x {len(v2_envs)}")

    events = [
        make_event_script_from_dict(
            {
                "name": "central_bank_rate_hike",
                "category": "policy",
                "visibility": "public",
                "intraday": False,
                "apply_days": [2],
                "description": "央行宣布加息 50bp",
                "effects": [
                    {"op": "delta", "target": "market_state.interest_rate", "value": 0.005},
                ],
                "tag": tag,
            }
        ),
        make_event_script_from_dict(
            {
                "name": "supply_chain_disruption",
                "category": "market",
                "visibility": "partial",
                "intraday": False,
                "apply_days": [1],
                "description": "供应链中断",
                "effects": [
                    {"op": "delta", "target": "resource_pool.water", "value": -30.0},
                ],
                "tag": tag,
            }
        ),
    ]
    for ev in events:
        ev.save()
    print(f"[save] EventScript x {len(events)}")

    if len(agents) >= 2:
        c = Contract(
            episode_pk="",
            proposer_pk=agents[0].pk,
            counterparties=[agents[1].pk],
            contract_type="loan",
            terms={"amount": 100.0, "rate": 0.05, "maturity_day": 7},
            penalty={"reputation_delta": -10.0, "cash_delta": -20.0},
            proposed_day=0,
            expiry_day=7,
            status="proposed",
        )
        c.save()
        print("[save] Contract x 1")

    if v2_envs:
        snap = make_initial_state_snapshot(
            episode_pk=f"placeholder_for_env_{v2_envs[0].pk}",
            agent_pks=[v2_agents[0].pk, v2_agents[1].pk] if len(v2_agents) >= 2 else [v2_agents[0].pk],
            initial_resources_per_agent={a.pk: dict(a.initial_resources) for a in v2_agents[:2]},
            initial_reputation_per_agent={a.pk: a.initial_reputation for a in v2_agents[:2]},
            market_state=v2_envs[0].system_state_init.get("market_state", {}),
            resource_pool=v2_envs[0].system_state_init.get("resource_pool", {}),
        )
        snap.save()
        print("[save] SystemStateSnapshot x 1")


# ---------------------------------------------------------------------------
# 5) main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="先清空 ~/.sotopia/data/")
    parser.add_argument("--n-agents", type=int, default=len(AGENT_ARCHETYPES))
    parser.add_argument("--n-envs", type=int, default=len(ENVIRONMENT_ARCHETYPES))
    parser.add_argument("--combos-per-env", type=int, default=2)
    parser.add_argument(
        "--list-name", default="scratch_env_set", help="EnvironmentList.name"
    )
    parser.add_argument(
        "--override-hard-list",
        action="store_true",
        help="把 EnvironmentList.pk 设成 01HAK34YPB1H1RWXQDASDKHSNS，让 task=hard 直接命中",
    )
    parser.add_argument("--with-v2", action="store_true", help="同时造 V2 数据")
    parser.add_argument("--tag", default="scratch_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[backend] SOTOPIA_STORAGE_BACKEND={os.environ['SOTOPIA_STORAGE_BACKEND']}")
    print(f"[paths]   {LOCAL_DATA_DIR}")

    if args.clean:
        wipe_local_data(yes=True)

    n_agents = min(args.n_agents, len(AGENT_ARCHETYPES))
    n_envs = min(args.n_envs, len(ENVIRONMENT_ARCHETYPES))
    print(f"[plan] agents={n_agents} envs={n_envs} combos/env={args.combos_per_env}")

    agents = save_agents(AGENT_ARCHETYPES[:n_agents], tag=args.tag)
    envs = save_envs(ENVIRONMENT_ARCHETYPES[:n_envs], tag=args.tag)
    save_relationships(agents, tag=args.tag)
    combos = save_combos(envs, agents, combos_per_env=args.combos_per_env, seed=args.seed)
    save_environment_list(
        envs, combos, name=args.list_name, override_hard_pk=args.override_hard_list
    )

    if args.with_v2:
        print("\n[v2] 升级 V2 数据 ...")
        maybe_build_v2(agents, envs, tag=args.tag)

    print("\n========== DONE ==========")
    for sub in [
        "AgentProfile",
        "EnvironmentProfile",
        "RelationshipProfile",
        "EnvAgentComboStorage",
        "EnvironmentList",
        "AgentProfileV2",
        "EnvironmentProfileV2",
        "EventScript",
        "Contract",
        "SystemStateSnapshot",
    ]:
        d = LOCAL_DATA_DIR / sub
        n = len(list(d.glob("*.json"))) if d.exists() else 0
        print(f"  {sub:<25} {n}")

    print("\n下一步：")
    print(
        "  SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \\"
        "\n    --models gpt-4o-mini --partner-model gpt-4o-mini \\"
        "\n    --evaluator-model gpt-4o-mini --task scratch --tag run0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
