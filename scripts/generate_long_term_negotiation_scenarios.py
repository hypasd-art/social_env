#!/usr/bin/env python
"""批量构造 **长期谈判** 可用的 benchmark-style 场景数据（V1+V2）。

参考 ``sotopia/benchmark_v2_data_models.py`` 的增量模型与工厂函数：

- ``AgentProfile`` → ``upgrade_agent_profile`` → ``AgentProfileV2``（**每个环境各合成一套**六角色画像并写 combo / V2 快照）
- ``EnvironmentProfile`` → ``upgrade_environment_profile`` → ``EnvironmentProfileV2``
- ``make_event_script_from_dict`` → ``EventScript``
- ``Contract`` / ``SystemStateSnapshot`` / ``make_initial_state_snapshot``

与老 ``scripts/generate_from_scratch.py`` 一致：默认 ``SOTOPIA_STORAGE_BACKEND=local``，
落盘 ``~/.sotopia/data/{AgentProfile,EnvironmentProfile,...}/``. 运行时长期谈判流水线
当前仍主要从代码内 ``NegotiationTimelineParams`` / ``roles.py`` 取规则；这里提供的
``EnvironmentProfile.game_metadata.long_term_negotiation`` + V2 行是为 **采样题库 /
实验管理** 准备的侧车数据，可把 ``pk`` / ``timeline`` JSON 回填到评测入口。

支持的 ``--modes``：

- ``bilat`` / ``tri`` / ``quartet`` —— ``with_institutional`` lineup，按
  ``firm_a, firm_b, investor, regulator`` 顺序取 N=2/3/4。
- ``firms3`` / ``firms4`` —— ``firms_only`` lineup（**3 家及以上公司**），按
  ``firm_a, firm_b, firm_c, firm_d`` 顺序取 N=3/4，机构位 investor / regulator
  不进入世界（融资 / 监管路径成为 no-op，contract 主体由 N 家公司组成）。

示例::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local python scripts/generate_long_term_negotiation_scenarios.py --clean --tag ltr_benchmark_v1

    # 规模：只生成 D6/D8 时间轴，每种 (模式×预设) 重复 2 份；模式含双方 / 三方 / 四方 / 三家公司 / 四家公司
    python scripts/generate_long_term_negotiation_scenarios.py --tag ltr_scale_v1 \\
        --modes bilat,tri,quartet,firms3,firms4 --timeline-labels D6,D8 --replicates 2

    # 仅 3+ 家公司互谈（不含 investor/regulator）
    python scripts/generate_long_term_negotiation_scenarios.py --tag ltr_firms_only \\
        --modes firms3,firms4 --timeline-labels D6,D8 --replicates 1

    # 精确指定每种人数 / 公司数的场景条数（不再用 --modes / --replicates）：
    # 8 条 firms3 + 12 条 firms4 + 5 条 bilat（在 D6,D8 preset 上轮转）
    python scripts/generate_long_term_negotiation_scenarios.py --tag ltr_mix \\
        --mode-counts firms3=8,firms4=12,bilat=5 --timeline-labels D6,D8

    # 要求说明（写入 manifest，便于实验记录）
    python scripts/generate_long_term_negotiation_scenarios.py --requirements "用于论文表2；仅规则评测" --tag ltr_paper
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

from sotopia.settings.long_term_negotiation.types import (  # noqa: E402
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    NegotiationTimelineParams,
    SESSION_FIRMS_ONLY_ROLE_ORDER,
    SESSION_SPEAKER_ROLE_ORDER,
)
from sotopia.settings.long_term_negotiation.roles import (  # noqa: E402
    CANONICAL_NEGOTIATION_ROSTER,
    FIRM_ROLES_ORDER,
    ROLE_SUMMARY_EN,
    default_agent_resources_bundle,
)

# 全部 6 个 canonical 角色（含 firm_c / firm_d / investor / regulator）；按字母排序便于
# 写库 / manifest 时稳定（与 V1 quartet 时仅 4 个不冲突）。
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


def parse_unique_modes(s: str) -> list[str]:
    """``--modes`` 逗号分隔，去重保序。

    合法 token：

    - ``bilat`` / ``tri`` / ``quartet`` —— ``with_institutional`` lineup（含机构位）。
    - ``firms3`` / ``firms4`` —— ``firms_only`` lineup（3 / 4 家公司，无机构位）。
    """
    allow = frozenset({"bilat", "tri", "quartet", "firms3", "firms4"})
    out: list[str] = []
    seen: set[str] = set()
    for part in s.split(","):
        p = part.strip().lower()
        if p not in allow or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out or ["bilat", "quartet"]


def parse_mode_counts(s: str) -> dict[str, int] | None:
    """``--mode-counts MODE=N[,MODE=N...]`` 解析。

    例：``firms3=8,firms4=12`` -> ``{'firms3': 8, 'firms4': 12}``，每个 mode 各生成 N 条
    （在 ``--timeline-labels`` 选定的 preset 上轮转）；返回 ``None`` 表示走 ``--modes`` +
    ``--replicates`` 旧路径。
    """
    spec = (s or "").strip()
    if not spec:
        return None
    allow = frozenset({"bilat", "tri", "quartet", "firms3", "firms4"})
    plan: dict[str, int] = {}
    for raw in spec.split(","):
        chunk = raw.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"invalid --mode-counts segment {chunk!r}; use MODE=COUNT (e.g. firms3=8)"
            )
        key, val = chunk.split("=", 1)
        mode = key.strip().lower()
        if mode not in allow:
            raise ValueError(
                f"unknown mode {mode!r} in --mode-counts; allowed {sorted(allow)}"
            )
        try:
            n = int(val.strip())
        except ValueError as e:
            raise ValueError(f"invalid count for mode {mode!r}: {val!r}") from e
        if n < 0:
            raise ValueError(f"--mode-counts {mode!r} must be >= 0, got {n}")
        # 同 mode 多次出现时累加
        plan[mode] = plan.get(mode, 0) + n
    plan = {m: c for m, c in plan.items() if c > 0}
    if not plan:
        raise ValueError("--mode-counts is non-empty but expanded to zero envs")
    return plan


# (模式 → (lineup, num_participants)) 映射；保持 token 与脚本之外的 manifest 兼容。
_MODE_TO_LINEUP_N: dict[str, tuple[str, int]] = {
    "bilat": (NEGOTIATION_LINEUP_WITH_INSTITUTIONAL, 2),
    "tri": (NEGOTIATION_LINEUP_WITH_INSTITUTIONAL, 3),
    "quartet": (NEGOTIATION_LINEUP_WITH_INSTITUTIONAL, 4),
    "firms3": (NEGOTIATION_LINEUP_FIRMS_ONLY, 3),
    "firms4": (NEGOTIATION_LINEUP_FIRMS_ONLY, 4),
}


def lineup_and_n_for_mode(mode: str) -> tuple[str, int]:
    if mode not in _MODE_TO_LINEUP_N:
        raise ValueError(
            f"unknown --modes token {mode!r}; expected one of {sorted(_MODE_TO_LINEUP_N)}"
        )
    return _MODE_TO_LINEUP_N[mode]


def roles_for_mode(mode: str) -> tuple[str, ...]:
    """模式 -> roster 前缀（按 lineup 顺序取 N 个 canonical 角色）。"""
    lineup, n = lineup_and_n_for_mode(mode)
    if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        return tuple(SESSION_FIRMS_ONLY_ROLE_ORDER[:n])
    return tuple(SESSION_SPEAKER_ROLE_ORDER[:n])


def filter_timeline_presets(
    presets: list[tuple[str, NegotiationTimelineParams]],
    labels: frozenset[str],
) -> list[tuple[str, NegotiationTimelineParams]]:
    if not labels:
        return list(presets)
    return [x for x in presets if x[0] in labels]


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

NEGOTIATION_SCENARIO_TRILATERAL = (
    NEGOTIATION_SCENARIO_BODY
    + " An external financing participant may join selected sessions contingent on formal moves."
)

NEGOTIATION_SCENARIO_FIRMS_ONLY_3 = (
    "Three commercial parties negotiate a multi-day acquisition / consortium structure across staged "
    "formal sessions. The third firm participates either as a joint bidder, partner-investor firm, "
    "or co-seller; financing and regulatory paths remain off-table — all contract principals are firms."
)

NEGOTIATION_SCENARIO_FIRMS_ONLY_4 = (
    "Four commercial parties negotiate a multi-day acquisition / consortium structure across staged "
    "formal sessions. The third and fourth firms enter as additional bidders or consortium members; "
    "financing and regulatory paths remain off-table — all contract principals are firms."
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
    num_participants: int | None = None,
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
) -> EnvironmentProfile:
    """``lineup`` + ``num_participants`` 共同决定 roster 与场景文案。

    - ``with_institutional``：N=2 (firm_a/firm_b)，N=3 (+investor)，N=4 (+regulator)。
    - ``firms_only``：N=3 (firm_a/firm_b/firm_c)，N=4 (+firm_d)；不含机构位。
    """
    n = num_participants if num_participants is not None else (4 if quartet else 2)
    if n < 2 or n > 4:
        raise ValueError(f"num_participants must be 2..4, got {n}")
    gm_quartet = False
    if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        if n < 2:
            raise ValueError("firms_only lineup requires num_participants>=2")
        if n == 2:
            body = NEGOTIATION_SCENARIO_BODY
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Close deal under cash limits.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Negotiate staged consideration.",
            ]
        elif n == 3:
            body = NEGOTIATION_SCENARIO_FIRMS_ONLY_3
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Lead acquirer; align consortium economics.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Target / co-seller; protect downside.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_c']}</strategy_hint> Joint bidder / partner firm; carve scope.",
            ]
        else:
            body = NEGOTIATION_SCENARIO_FIRMS_ONLY_4
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Lead acquirer; coordinate consortium.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Target firm; preserve optionality.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_c']}</strategy_hint> Co-bidder; structure tranches.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_d']}</strategy_hint> Late-entrant bidder; trade speed for clarity.",
            ]
    else:
        if n == 2:
            body = NEGOTIATION_SCENARIO_BODY
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Close deal under cash/financing limits.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Negotiate staged consideration.",
            ]
        elif n == 3:
            body = NEGOTIATION_SCENARIO_TRILATERAL
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Secure financing & approvals.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Maximize lawful consideration.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['investor']}</strategy_hint> Structure contingent capital.",
            ]
        else:
            body = NEGOTIATION_SCENARIO_QUARTET
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Secure financing & approvals.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Maximize lawful consideration.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['investor']}</strategy_hint> Structure contingent capital.",
                f"<strategy_hint>{ROLE_SUMMARY_EN['regulator']}</strategy_hint> Enforce procedural thresholds.",
            ]
            gm_quartet = True
    from sotopia.settings.long_term_negotiation.scenario_loader import (
        build_negotiation_game_metadata_bundle,
    )

    gm_base = build_negotiation_game_metadata_bundle(
        codename,
        gm_quartet,
        params,
        num_participants=n,
        lineup=lineup,
        scenario_text=body,
    )
    return EnvironmentProfile(
        codename=codename,
        source="benchmark_v2_synthetic_long_term_negotiation",
        scenario=body,
        agent_goals=goals,
        relationship=RelationshipType.stranger,
        tag=tag,
        game_metadata=gm_base,
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
    num_participants: int | None = None,
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
) -> tuple[Any, Contract | None, Any]:
    """落库单个场景的 ``EnvironmentProfileV2``、``SystemStateSnapshot``、可选 ``Contract``。

    ``lineup`` 决定 N 名 canonical 角色顺序：``with_institutional`` → ``SESSION_SPEAKER_ROLE_ORDER``；
    ``firms_only`` → ``SESSION_FIRMS_ONLY_ROLE_ORDER``（公司自相谈，无机构位）。
    """
    n_agents = num_participants if num_participants is not None else (4 if quartet else 2)
    if n_agents < 2 or n_agents > 4:
        raise ValueError(f"num_participants must be 2..4, got {n_agents}")
    role_order = (
        SESSION_FIRMS_ONLY_ROLE_ORDER
        if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY
        else SESSION_SPEAKER_ROLE_ORDER
    )
    roles = tuple(role_order[:n_agents])
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
    if len(pks) >= 3:
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
        help=(
            "逗号分隔、去重保序：bilat（2 人，with_institutional）/ tri（3 人，含 investor）/ "
            "quartet（4 人，含 investor + regulator）/ firms3（3 家公司，无机构位）/ "
            "firms4（4 家公司，无机构位）"
        ),
    )
    ap.add_argument(
        "--timeline-labels",
        default="",
        help="逗号分隔，仅保留这些时间轴预设标签（如 D6,D8,D12）；空表示全部",
    )
    ap.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="每种 (模式 × 时间轴预设) 重复写入的份数，用于扩大题库规模（>=1）",
    )
    ap.add_argument(
        "--mode-counts",
        default="",
        help=(
            "按模式精确指定生成条数：MODE=N[,MODE=N...]；例 firms3=8,firms4=12,bilat=5。"
            "传入后忽略 --modes 与 --replicates，每个 mode 在 --timeline-labels 选定的 preset 上"
            "轮转生成 N 条。合法 MODE：bilat/tri/quartet/firms3/firms4。"
        ),
    )
    ap.add_argument(
        "--requirements",
        default="",
        help="自由文本，写入 manifest 的 generation_spec.requirements_notes（实验要求/筛选标准等）",
    )
    args = ap.parse_args()

    print(f"[backend] SOTOPIA_STORAGE_BACKEND={os.environ['SOTOPIA_STORAGE_BACKEND']}")
    print(f"[paths]   {LOCAL_DATA_DIR}")

    wipe_local_data(yes=args.clean)

    explicit_counts = parse_mode_counts(args.mode_counts)
    if explicit_counts is not None:
        modes_sel = list(explicit_counts.keys())
    else:
        modes_sel = parse_unique_modes(args.modes)
    presets_all = bilateral_timeline_presets()
    label_filter = frozenset(
        x.strip() for x in args.timeline_labels.split(",") if x.strip()
    )
    presets_use = filter_timeline_presets(presets_all, label_filter)
    if label_filter:
        unknown = sorted(label_filter - {x[0] for x in presets_all})
        if unknown:
            print(f"[warn] unknown --timeline-labels (ignored): {unknown}")
    if not presets_use:
        print("[err] no timeline presets left after --timeline-labels filter")
        return 1
    replicates = max(1, int(args.replicates))
    if explicit_counts is not None:
        total_envs = sum(explicit_counts.values())
        breakdown = ", ".join(f"{m}={explicit_counts[m]}" for m in modes_sel)
        print(
            f"[plan] --mode-counts active: {breakdown} | "
            f"preset_labels={[p[0] for p in presets_use]} -> environments={total_envs} "
            f"(--replicates {replicates} ignored)"
        )
    else:
        approx_envs = len(modes_sel) * len(presets_use) * replicates
        print(
            f"[plan] modes={modes_sel} preset_labels={[p[0] for p in presets_use]} "
            f"replicates={replicates} -> environments≈{approx_envs}"
        )

    events = negotiation_event_scripts(args.tag)
    for ev in events:
        ev.save()
    anchor_pk = events[0].pk if events else None
    print(f"[save] EventScript x {len(events)} anchor_pk={anchor_pk}")

    combos_by_codename: dict[str, EnvAgentComboStorage] = {}
    legacy_env_objs: list[EnvironmentProfile] = []
    env_modes_by_codename: dict[str, str] = {}
    env_lineup_by_codename: dict[str, str] = {}
    env_agent_pks_by_codename: dict[str, dict[str, str]] = {}
    env_agent_v2_pks_by_codename: dict[str, dict[str, str]] = {}

    _MODE_PREFIX = {
        "bilat": "bil",
        "tri": "tri",
        "quartet": "quad",
        "firms3": "firms3",
        "firms4": "firms4",
    }

    variant_i = 0
    for mode in modes_sel:
        lineup, n_agents = lineup_and_n_for_mode(mode)
        roles = roles_for_mode(mode)
        prefix = _MODE_PREFIX.get(mode, mode)

        if explicit_counts is not None:
            # 每个 mode 生成 explicit_counts[mode] 条，逐条在 presets_use 上轮转。
            jobs = [
                (i, *presets_use[i % len(presets_use)])
                for i in range(explicit_counts[mode])
            ]
        else:
            jobs = [
                (rep * len(presets_use) + j, label, params)
                for rep in range(replicates)
                for j, (label, params) in enumerate(presets_use)
            ]

        for slot, label, params in jobs:
            codename = f"ltr_neg_{prefix}_{label}_v{variant_i}_r{slot}"
            variant_i += 1
            legacy = build_environment_profile_legacy(
                codename=codename,
                quartet=(lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL and n_agents == 4),
                params=params,
                tag=args.tag,
                num_participants=n_agents,
                lineup=lineup,
            )
            legacy.save()
            agent_bind_tag = f"{args.tag}__{codename}"
            agents = save_negotiation_agents(tag=agent_bind_tag)
            pairwise_strangers(agents, tag=agent_bind_tag)
            v2_agents = save_negotiation_agent_profiles_v2(agents, tag=agent_bind_tag)
            combo = save_combo(legacy, roles, agents)
            combos_by_codename[codename] = combo
            legacy_env_objs.append(legacy)
            env_modes_by_codename[codename] = mode
            env_lineup_by_codename[codename] = lineup
            env_agent_pks_by_codename[codename] = {r: agents[r].pk for r in QUARTET_ROSTER_ORDER}
            env_agent_v2_pks_by_codename[codename] = {r: v2_agents[r].pk for r in QUARTET_ROSTER_ORDER}
            persist_scenario_v2(
                legacy,
                quartet=(lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL and n_agents == 4),
                params=params,
                tag=args.tag,
                event_anchor_pk=anchor_pk,
                v2_by_role=v2_agents,
                num_participants=n_agents,
                lineup=lineup,
            )

    save_environment_list_for_combos(legacy_env_objs, combos_by_codename)

    manifest = {
        "tag": args.tag,
        "source": "generate_long_term_negotiation_scenarios.py",
        "agent_roles": QUARTET_ROSTER_ORDER,
        "agent_profiles_binding": "per_environment",
        "environments": [
            {
                "codename": e.codename,
                "pk": e.pk,
                "mode": env_modes_by_codename.get(e.codename),
                "lineup": env_lineup_by_codename.get(e.codename),
                "agent_profile_pks_by_role": env_agent_pks_by_codename.get(e.codename, {}),
                "agent_profile_v2_pks_by_role": env_agent_v2_pks_by_codename.get(e.codename, {}),
            }
            for e in legacy_env_objs
        ],
        "generation_spec": {
            "modes": list(modes_sel),
            "timeline_labels_filter": sorted(label_filter) if label_filter else None,
            "presets_used": [p[0] for p in presets_use],
            "replicates": (None if explicit_counts is not None else replicates),
            "mode_counts_spec": (args.mode_counts.strip() or None),
            "mode_counts_resolved": explicit_counts,
            "environment_count": len(legacy_env_objs),
            "requirements_notes": (args.requirements.strip() or None),
            "agent_profiles_binding": "per_environment",
        },
    }
    manifest_path = LOCAL_DATA_DIR / "long_term_negotiation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] manifest {manifest_path}")

    print("\n========== DONE ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
