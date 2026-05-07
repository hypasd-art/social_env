#!/usr/bin/env python
"""V2 benchmark 种子数据生成器（无 LLM 版）。

把 ``~/.sotopia/data/`` 里现成的老 ``AgentProfile`` / ``EnvironmentProfile``
升级成 V2，再造若干 ``EventScript`` / ``Contract`` / 初始 ``SystemStateSnapshot``，
全部写回本地 JSON 后端。跑完之后 ``~/.sotopia/data/`` 下会多出 5 个新目录：

    AgentProfileV2/         <- 升级后的 V2 角色卡
    EnvironmentProfileV2/   <- 升级后的 V2 场景卡
    EventScript/            <- 末日洪水 / 央行加息 等外部事件
    Contract/               <- 模板合约（status=proposed）
    SystemStateSnapshot/    <- day=0 初始状态

不依赖 LLM，秒级完成；建议先跑这个把流水线打通，再用
``scripts/generate_v2_with_llm.py`` 用 LLM 扩量。

用法
----
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/generate_v2_seed.py

    # 也可以把数量调大
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/generate_v2_seed.py \\
        --n-agents 20 --n-envs 5 --tag bench_v2_demo

    # dry-run 只打印 plan，不落库
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/generate_v2_seed.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 必须在 import sotopia 之前固定后端，否则 V2 模型会按 Redis 路径 import
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")

from sotopia.benchmark_v2_data_models import (  # noqa: E402
    Contract,
    EffectOp,
    EventScript,
    SystemStateSnapshot,
    make_event_script_from_dict,
    make_initial_state_snapshot,
    upgrade_agent_profile,
    upgrade_environment_profile,
)
from sotopia.database import (  # noqa: E402
    AgentProfile,
    EnvironmentProfile,
)


# ---------------------------------------------------------------------------
# 1) 升级老 profile 成 V2
# ---------------------------------------------------------------------------


def upgrade_n_agents(n: int, *, tag: str) -> list[Any]:
    """选 n 个老 AgentProfile，按职业映射 risk_preference / role_type / 起始资源。"""

    pks = list(AgentProfile.all_pks())
    if not pks:
        raise RuntimeError(
            "本地后端没有 AgentProfile；先跑 scripts/setup_local_data.py "
            "把 export/AgentProfile.json 导进来"
        )

    selected = pks[: min(n, len(pks))]
    print(f"[agents] 找到 {len(pks)} 条老 AgentProfile，本次升级前 {len(selected)} 条")

    upgraded = []
    for pk in selected:
        old = AgentProfile.get(pk)

        occ = (old.occupation or "").lower()
        if any(k in occ for k in ["banker", "trader", "ceo"]):
            role, risk, cash = "investor", "seeking", 500.0
        elif any(k in occ for k in ["doctor", "nurse", "teacher"]):
            role, risk, cash = "regulator", "averse", 200.0
        elif any(k in occ for k in ["student", "intern"]):
            role, risk, cash = "buyer", "neutral", 80.0
        else:
            role, risk, cash = "buyer", "neutral", 100.0

        v2 = upgrade_agent_profile(
            old,
            initial_resources={"cash": cash, "energy": 100.0},
            initial_reputation=50.0 + (hash(pk) % 21 - 10),
            risk_preference=risk,
            role_type=role,
        )
        v2.tag = tag
        upgraded.append(v2)
    return upgraded


def upgrade_n_envs(n: int, *, tag: str) -> list[Any]:
    """选 n 个老 EnvironmentProfile，按 source 字段映射 scenario_type / 时长。"""

    pks = list(EnvironmentProfile.all_pks())
    if not pks:
        raise RuntimeError(
            "本地后端没有 EnvironmentProfile；先跑 scripts/setup_local_data.py"
        )

    selected = pks[: min(n, len(pks))]
    print(f"[envs] 找到 {len(pks)} 条老 EnvironmentProfile，本次升级前 {len(selected)} 条")

    upgraded = []
    for pk in selected:
        old = EnvironmentProfile.get(pk)
        src = (old.source or "").lower()

        if "craigslist" in src or "bargain" in src:
            stype, days, intra = "negotiation", 3, 4
        elif "mutual" in src:
            stype, days, intra = "generic", 1, 6
        else:
            stype, days, intra = "generic", 2, 4

        v2 = upgrade_environment_profile(
            old,
            scenario_type=stype,
            n_agents=2,
            max_days=days,
            intra_day_steps=intra,
            event_schedule_pk=None,
            system_state_init={
                "market_state": {"interest_rate": 0.05, "price_index": 100.0},
                "resource_pool": {"water": 100.0},
            },
        )
        upgraded.append(v2)
    return upgraded


# ---------------------------------------------------------------------------
# 2) 造 EventScript（外部事件 - 末日触发）
# ---------------------------------------------------------------------------


def make_event_scripts(*, tag: str) -> list[Any]:
    specs: list[dict[str, Any]] = [
        {
            "name": "central_bank_rate_hike",
            "category": "policy",
            "visibility": "public",
            "intraday": False,
            "apply_days": [2],  # 第 2 天 end_of_day 触发
            "description": "央行宣布加息 50bp，所有资产估值需重新折现",
            "effects": [
                {"op": "delta", "target": "market_state.interest_rate", "value": 0.005},
                {"op": "delta", "target": "market_state.price_index", "value": -3.0},
                {"op": "broadcast", "target": "all_agents", "value": "央行加息 50bp"},
            ],
            "tag": tag,
        },
        {
            "name": "supply_chain_disruption",
            "category": "market",
            "visibility": "partial",
            "intraday": False,
            "apply_days": [1],
            "description": "供应链中断，物资紧缺，公共资源池骤减",
            "effects": [
                {"op": "delta", "target": "resource_pool.water", "value": -30.0},
                {"op": "broadcast", "target": "all_agents", "value": "供水紧张通告"},
            ],
            "tag": tag,
        },
        {
            "name": "rumor_about_agent",
            "category": "social",
            "visibility": "private",
            "intraday": False,
            "apply_days": [3],
            "description": "市场上出现关于某 agent 的负面传闻",
            "effects": [
                # 注意：占位字符串；运行时 EventEngine 会按 episode 上下文解析
                {"op": "delta", "target": "agent_reputation.<target_pk>", "value": -8.0},
            ],
            "tag": tag,
        },
    ]

    return [make_event_script_from_dict(s) for s in specs]


# ---------------------------------------------------------------------------
# 3) 造合约模板（status=proposed，episode_pk 留空，跑时绑定）
# ---------------------------------------------------------------------------


def make_contract_templates(agent_pks: list[str], *, tag: str) -> list[Any]:
    if len(agent_pks) < 2:
        return []

    a, b = agent_pks[0], agent_pks[1]
    return [
        Contract(
            episode_pk="",  # PoC: 模板，不绑 episode
            proposer_pk=a,
            counterparties=[b],
            contract_type="loan",
            terms={"amount": 100.0, "rate": 0.05, "maturity_day": 7},
            penalty={"reputation_delta": -10.0, "cash_delta": -20.0},
            proposed_day=0,
            expiry_day=7,
            status="proposed",
        ),
        Contract(
            episode_pk="",
            proposer_pk=b,
            counterparties=[a],
            contract_type="trade",
            terms={"item": "water", "amount": 10.0, "price": 5.0, "deliver_day": 2},
            penalty={"reputation_delta": -5.0, "cash_delta": -15.0},
            proposed_day=0,
            expiry_day=3,
            status="proposed",
        ),
    ]


# ---------------------------------------------------------------------------
# 4) 造 day=0 的 SystemStateSnapshot
# ---------------------------------------------------------------------------


def make_state_snapshots(envs: list[Any], agents: list[Any]) -> list[Any]:
    snaps: list[Any] = []
    for env in envs:
        a_subset = [a.pk for a in agents[: env.n_agents]]
        if not a_subset:
            continue
        snap = make_initial_state_snapshot(
            episode_pk=f"placeholder_for_env_{env.pk}",  # PoC: 占位
            agent_pks=a_subset,
            initial_resources_per_agent={
                a.pk: dict(a.initial_resources) for a in agents[: env.n_agents]
            },
            initial_reputation_per_agent={
                a.pk: a.initial_reputation for a in agents[: env.n_agents]
            },
            market_state=env.system_state_init.get("market_state", {}),
            resource_pool=env.system_state_init.get("resource_pool", {}),
        )
        snaps.append(snap)
    return snaps


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--tag", type=str, default="bench_v2_seed_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[backend] SOTOPIA_STORAGE_BACKEND={os.environ['SOTOPIA_STORAGE_BACKEND']}")
    print(f"[plan] 升级 {args.n_agents} agent + {args.n_envs} env，tag={args.tag}")

    agents = upgrade_n_agents(args.n_agents, tag=args.tag)
    envs = upgrade_n_envs(args.n_envs, tag=args.tag)
    events = make_event_scripts(tag=args.tag)
    contracts = make_contract_templates([a.pk for a in agents], tag=args.tag)
    snapshots = make_state_snapshots(envs, agents)

    if args.dry_run:
        print("\n========== DRY RUN ==========")
        print(f"AgentProfileV2:        {len(agents)}  e.g. {agents[0].first_name} ({agents[0].role_type})")
        print(f"EnvironmentProfileV2:  {len(envs)}    e.g. {envs[0].codename or envs[0].pk[:8]} ({envs[0].scenario_type}, {envs[0].max_days}天)")
        print(f"EventScript:           {len(events)}  e.g. {events[0].name}")
        print(f"Contract:              {len(contracts)}")
        print(f"SystemStateSnapshot:   {len(snapshots)}")
        print("（未写盘）")
        return 0

    print("\n[save] 开始落本地后端 (~/.sotopia/data/)")
    for obj in [*agents, *envs, *events, *contracts, *snapshots]:
        obj.save()

    print("\n========== DONE ==========")
    print(f"AgentProfileV2:        {len(agents)}")
    print(f"EnvironmentProfileV2:  {len(envs)}")
    print(f"EventScript:           {len(events)}")
    print(f"Contract:              {len(contracts)}")
    print(f"SystemStateSnapshot:   {len(snapshots)}")
    print("\n[verify] 用 ls 看下:")
    for sub in [
        "AgentProfileV2",
        "EnvironmentProfileV2",
        "EventScript",
        "Contract",
        "SystemStateSnapshot",
    ]:
        d = Path(os.path.expanduser(f"~/.sotopia/data/{sub}"))
        n = len(list(d.glob("*.json"))) if d.exists() else 0
        print(f"  ~/.sotopia/data/{sub:<25} {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
