#!/usr/bin/env python
"""用大模型生成 **长期谈判题库** EnvironmentProfile + AgentProfile（与 ``agenerate_env_profile`` / ``design_1`` 对齐）。

1. 调 ``agenerate_env_profile`` 生成自然语言 ``scenario`` 与 ``agent_goals``（老 Sotopia 结构）。
2. 合并 ``scenario_loader.build_negotiation_game_metadata_bundle`` —— timeline / lineup 与手写脚本同源，
   便于 ``negotiation-batch --scenario-manifest`` 直接加载评测。
3. **AgentProfile**：**每合成一条环境**即合成并落库一套六角色 ``AgentProfile`` / ``AgentProfileV2``，再写入
   该环境的 ``EnvAgentComboStorage`` 与 V2 快照，与环境一一绑定。公司侧默认走 LLM、机构位静态；
   ``--agent-profiles-all-llm`` 六角色均 LLM；``--legacy-agent-profiles`` 为每环境手写占位画像。
4. 复用手写脚本 ``generate_long_term_negotiation_scenarios.py`` 的 ``EnvAgentComboStorage`` /
   ``persist_scenario_v2``，保持 V2 快照与 benchmark 数据结构一致。

支持的 ``--modes``：

- ``bilat`` / ``tri`` / ``quartet`` —— ``with_institutional`` lineup（含 investor / regulator）。
- ``firms3`` / ``firms4`` —— ``firms_only`` lineup，3 / 4 家公司互谈，不含机构位。

依赖 ``social_env/.env`` 里的 ``OPENAI_API_KEY``（及可选 BASE_URL）。

用法::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local python scripts/generate_long_term_negotiation_llm.py \\
        --model gpt-4o-mini --n 3 --modes bilat --tag ltr_llm_v1

    # 规模：条数 --n；并发 --concurrency；时间轴仅用 D6/D8；模式按列表对每条 profile 轮转（含 firms3/firms4）
    python scripts/generate_long_term_negotiation_llm.py --n 12 --concurrency 4 \\
        --timeline-labels D6,D8 --modes bilat,tri,quartet,firms3,firms4 --tag ltr_llm_scale

    # 仅 3+ 家公司互谈
    python scripts/generate_long_term_negotiation_llm.py --n 6 --modes firms3,firms4 \\
        --timeline-labels D6,D8 --tag ltr_llm_firms_only

    # 精确指定每种人数 / 公司数的生成条数（不按 --modes 轮转）：
    # 8 条 firms3 + 12 条 firms4 + 5 条 bilat + 3 条 quartet
    python scripts/generate_long_term_negotiation_llm.py \\
        --mode-counts firms3=8,firms4=12,bilat=5,quartet=3 \\
        --timeline-labels D6,D8 --concurrency 4 --tag ltr_llm_mix

    # 要求说明写入 manifest（generation_spec）
    python scripts/generate_long_term_negotiation_llm.py --n 2 --requirements "仅用于 smoke；需人工抽检 scenario" --tag ltr_smoke

    # 用更轻的 agent_profile 模型 + 自定义导出文件名
    python scripts/generate_long_term_negotiation_llm.py --n 3 --agent-profile-model gpt-4o-mini \\
        --agent-profile-out long_term_negotiation_llm_agent_profiles.smoke.json

    # 全部六角色（含 investor/regulator）都走 LLM
    python scripts/generate_long_term_negotiation_llm.py --n 3 --agent-profiles-all-llm

    # 想保留旧的硬编码 AgentProfile（不调 LLM 造画像）
    python scripts/generate_long_term_negotiation_llm.py --n 3 --legacy-agent-profiles

    # 不写库、只看 prompt / 配额
    python scripts/generate_long_term_negotiation_llm.py --dry-run --n 5
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from sotopia.database import EnvironmentProfile  # noqa: E402
from sotopia.generation_utils.generate import agenerate_env_profile  # noqa: E402
from sotopia.settings.long_term_negotiation.llm_agent_profile_gen import (  # noqa: E402
    DEFAULT_COMPANY_LLM_ROLES,
    agenerate_negotiation_agent_profiles,
    agent_profile_to_jsonable,
)
from sotopia.settings.long_term_negotiation.scenario_loader import (
    build_negotiation_game_metadata_bundle,
)
from sotopia.settings.long_term_negotiation.types import (  # noqa: E402
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    NegotiationTimelineParams,
    SESSION_FIRMS_ONLY_ROLE_ORDER,
    SESSION_SPEAKER_ROLE_ORDER,
)

LOCAL_DATA_DIR = Path(os.path.expanduser("~/.sotopia/data"))


def _load_handwritten_generator() -> Any:
    p = REPO_ROOT / "scripts" / "generate_long_term_negotiation_scenarios.py"
    spec = importlib.util.spec_from_file_location("ltr_gen_manual", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NEGOTIATION_LLM_PROMPTS: list[str] = [
    "Multi-calendar-day M&A process: acquirer and target negotiate milestones, escrow, financing conditions, "
    "and drafting sessions under slot-limited invitations.",
    "Cross-border carve-out sale: seller and buyer align on transitional services, liabilities, regulator filings, "
    "and phased payments across several business weeks.",
    "Distressed refinancing with competing investors: debtor firm negotiates interest, covenants, and "
    "contingent capital tranches alongside an approval-heavy timeline.",
    "Three-firm consortium acquisition: lead acquirer, co-bidder firm, and target negotiate joint scope, "
    "purchase-price allocation, and post-close governance across multi-day formal sessions.",
    "Four-firm carve-out auction: two bidder firms (firm_a / firm_c) compete for two carve-out segments "
    "from a parent group (firm_b / firm_d), with paced formal moves and stalking-horse dynamics.",
]

_LLM_MODE_TO_LINEUP_N: dict[str, tuple[str, int]] = {
    "bilat": (NEGOTIATION_LINEUP_WITH_INSTITUTIONAL, 2),
    "tri": (NEGOTIATION_LINEUP_WITH_INSTITUTIONAL, 3),
    "quartet": (NEGOTIATION_LINEUP_WITH_INSTITUTIONAL, 4),
    "firms3": (NEGOTIATION_LINEUP_FIRMS_ONLY, 3),
    "firms4": (NEGOTIATION_LINEUP_FIRMS_ONLY, 4),
}


def modes_cycle_from_arg(s: str) -> list[str]:
    """与 ``--modes`` 字符串顺序一致，对每条 LLM profile 轮转（合法 token 同手写脚本）。"""
    allow = frozenset(_LLM_MODE_TO_LINEUP_N)
    return [p.strip().lower() for p in s.split(",") if p.strip().lower() in allow] or ["bilat"]


def parse_mode_counts(spec: str) -> list[str] | None:
    """``--mode-counts MODE=N[,MODE=N...]`` 解析。

    例：``firms3=8,firms4=12,bilat=5`` -> 8 条 firms3 + 12 条 firms4 + 5 条 bilat
    （顺序与逗号片段一致，逐条作为 LLM profile 生成的 mode）。

    返回 ``None`` 表示用户未传该参数；走原有 ``--n`` + ``--modes`` 轮转路径。
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    allow = frozenset(_LLM_MODE_TO_LINEUP_N)
    plan: list[str] = []
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
            raise ValueError(
                f"invalid count for mode {mode!r}: {val!r}"
            ) from e
        if n < 0:
            raise ValueError(f"--mode-counts {mode!r} must be >= 0, got {n}")
        plan.extend([mode] * n)
    if not plan:
        raise ValueError("--mode-counts is non-empty but expanded to zero jobs")
    return plan


def _lineup_n_for_mode(mode: str) -> tuple[str, int]:
    if mode not in _LLM_MODE_TO_LINEUP_N:
        raise ValueError(
            f"unknown --modes token {mode!r}; expected one of {sorted(_LLM_MODE_TO_LINEUP_N)}"
        )
    return _LLM_MODE_TO_LINEUP_N[mode]


def _roles_for_mode(mode: str) -> tuple[str, ...]:
    lineup, n = _lineup_n_for_mode(mode)
    if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        return tuple(SESSION_FIRMS_ONLY_ROLE_ORDER[:n])
    return tuple(SESSION_SPEAKER_ROLE_ORDER[:n])


def filter_presets_lite(
    presets: list[tuple[str, NegotiationTimelineParams]],
    labels: frozenset[str],
) -> list[tuple[str, NegotiationTimelineParams]]:
    if not labels:
        return list(presets)
    return [x for x in presets if x[0] in labels]


def bilateral_timeline_presets_lite() -> list[tuple[str, NegotiationTimelineParams]]:
    return [
        (
            "D1",
            NegotiationTimelineParams(
                D=1,
                s_max_per_day=2,
                max_session_rounds=8,
                max_total_turns_per_session=32,
            ),
        ),
        (
            "D2",
            NegotiationTimelineParams(
                D=2,
                s_max_per_day=2,
                max_session_rounds=12,
                max_total_turns_per_session=32,
            ),
        ),
        (
            "D6",
            NegotiationTimelineParams(
                D=6,
                s_max_per_day=2,
                max_session_rounds=24,
                max_total_turns_per_session=32,
            ),
        ),
        (
            "D8",
            NegotiationTimelineParams(
                D=8,
                s_max_per_day=2,
                max_session_rounds=32,
                max_total_turns_per_session=32,
            ),
        ),
        (
            "D12",
            NegotiationTimelineParams(
                D=12,
                s_max_per_day=3,
                max_session_rounds=36,
                max_total_turns_per_session=32,
            ),
        ),
    ]


def tri_goal_padding(env: EnvironmentProfile) -> None:
    """三方 roster 时把 ``agent_goals`` 补到 3 条（LLM 常只写买卖双方）。"""
    goals = list(env.agent_goals or [])
    tail = (
        "<strategy_hint>External financing stakeholder</strategy_hint> "
        "Provide contingent financing and milestone-based tranches aligned with disclosures."
    )
    if len(goals) < 3 and tail not in goals:
        goals.append(tail)
    env.agent_goals = goals[:3]


def quartet_goal_padding(env: EnvironmentProfile) -> None:
    """LLM 常输出 2 条 goal；四方 roster 时补齐至 4 条 institutional 视角。"""
    goals = list(env.agent_goals or [])
    tails = [
        (
            "<strategy_hint>External financing stakeholder</strategy_hint> "
            "Provide contingent financing and milestone-based tranches aligned with disclosures."
        ),
        (
            "<strategy_hint>Regulatory stakeholder</strategy_hint> "
            "Maintain filing calendars, substantive thresholds, and procedural gatekeeping."
        ),
    ]
    for t in tails:
        if len(goals) >= 4:
            break
        if t not in goals:
            goals.append(t)
    env.agent_goals = goals[:4]


def firms3_goal_padding(env: EnvironmentProfile) -> None:
    """``firms3`` lineup 时把 ``agent_goals`` 补到 3 条（第三方为公司，不是机构）。"""
    goals = list(env.agent_goals or [])
    tail = (
        "<strategy_hint>Third commercial firm (firm_c)</strategy_hint> "
        "Joint bidder / partner firm; align consortium economics and protect downside scope."
    )
    if len(goals) < 3 and tail not in goals:
        goals.append(tail)
    env.agent_goals = goals[:3]


def firms4_goal_padding(env: EnvironmentProfile) -> None:
    """``firms4`` lineup 时把 ``agent_goals`` 补到 4 条公司视角。"""
    goals = list(env.agent_goals or [])
    tails = [
        (
            "<strategy_hint>Third commercial firm (firm_c)</strategy_hint> "
            "Co-bidder; structure tranches and reserve walk-away rights."
        ),
        (
            "<strategy_hint>Fourth commercial firm (firm_d)</strategy_hint> "
            "Late-entrant bidder / consortium member; trade speed-to-close for cleaner reps & warranties."
        ),
    ]
    for t in tails:
        if len(goals) >= 4:
            break
        if t not in goals:
            goals.append(t)
    env.agent_goals = goals[:4]


async def generate_one_llm_profile(
    sem: asyncio.Semaphore,
    inspiration: str,
    model: str,
) -> EnvironmentProfile | BaseException:
    async with sem:
        try:
            return await agenerate_env_profile(model_name=model, inspiration_prompt=inspiration, examples="")
        except BaseException as e:
            return e


async def main_async(args: argparse.Namespace, ltr: Any) -> int:
    explicit_plan = parse_mode_counts(getattr(args, "mode_counts", "") or "")
    pool = list(args.inspiration) if args.inspiration else NEGOTIATION_LLM_PROMPTS
    if explicit_plan is not None:
        # 用户精确指定每种人数/公司数的条数；总条数由 plan 决定，不再用 --n。
        if args.n != 3 and len(explicit_plan) != args.n:
            print(
                f"[warn] --mode-counts fixes total LLM scenarios to {len(explicit_plan)}; "
                f"ignoring --n={args.n}"
            )
        inspirations = [pool[i % len(pool)] for i in range(len(explicit_plan))]
    else:
        inspirations = [pool[i % len(pool)] for i in range(max(1, args.n))]
    presets_all = bilateral_timeline_presets_lite()
    label_filter = frozenset(x.strip() for x in args.timeline_labels.split(",") if x.strip())
    presets = filter_presets_lite(presets_all, label_filter)
    if label_filter:
        unknown = sorted(label_filter - {x[0] for x in presets_all})
        if unknown:
            print(f"[warn] unknown --timeline-labels (ignored): {unknown}")
    if not presets:
        print("[err] no timeline presets after --timeline-labels filter")
        return 1
    modes_cycle = modes_cycle_from_arg(args.modes)
    combos_by_codename: dict[str, Any] = {}
    legacy_env_objs: list[EnvironmentProfile] = []

    def _mode_for_idx(i: int) -> str:
        if explicit_plan is not None:
            return explicit_plan[i]
        return modes_cycle[i % len(modes_cycle)]

    if explicit_plan is not None:
        from collections import Counter

        ctr = Counter(explicit_plan)
        order = list(dict.fromkeys(explicit_plan))
        breakdown = ", ".join(f"{m}={ctr[m]}" for m in order)
        print(
            f"[plan] --mode-counts active: total={len(explicit_plan)} ({breakdown}); "
            f"--modes / --n ignored for routing"
        )

    if args.dry_run:
        print(
            f"[dry-run] inspirations={len(inspirations)} modes_cycle={modes_cycle} "
            f"presets={[p[0] for p in presets]} model={args.model} concurrency={args.concurrency}"
        )
        for i, p in enumerate(inspirations):
            mode = _mode_for_idx(i)
            print(f"  {i}: mode={mode} | {p[:100]}...")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("[err] OPENAI_API_KEY 未设置（检查 social_env/.env）")
        return 1

    if args.clean:
        ltr.wipe_local_data(yes=True)

    print(f"[backend] OPENAI-ish model={args.model} conc={args.concurrency} tag={args.tag}")

    agent_profile_model = args.agent_profile_model or args.model
    llm_roles_for_agents: tuple[str, ...] | None = None
    agent_profile_path: Path | None = LOCAL_DATA_DIR / args.agent_profile_out
    agent_profile_source: str
    if args.legacy_agent_profiles:
        print("[agent_profiles] mode=handwritten per-environment (legacy constants)")
        agent_profile_source = "handwritten"
    else:
        llm_roles_for_agents = (
            tuple(ltr.QUARTET_ROSTER_ORDER) if args.agent_profiles_all_llm else DEFAULT_COMPANY_LLM_ROLES
        )
        print(
            f"[agent_profiles] mode=llm per-environment companies_only={not args.agent_profiles_all_llm} "
            f"model={agent_profile_model} llm_roles={list(llm_roles_for_agents)}"
        )
        agent_profile_source = "llm"

    events = ltr.negotiation_event_scripts(args.tag)
    for ev in events:
        ev.save()
    anchor_pk = events[0].pk if events else None
    print(f"[save] EventScript anchor_pk={anchor_pk}")

    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [generate_one_llm_profile(sem, ins, args.model) for ins in inspirations]
    raw_profiles = await asyncio.gather(*tasks)

    variant_i = 0
    env_modes_by_codename: dict[str, str] = {}
    env_lineup_by_codename: dict[str, str] = {}
    env_agent_pks_by_codename: dict[str, dict[str, str]] = {}
    env_agent_v2_pks_by_codename: dict[str, dict[str, str]] = {}
    per_environment_agent_export_rows: list[dict[str, Any]] = []
    llm_role_set = frozenset(llm_roles_for_agents or ())
    for idx, row in enumerate(raw_profiles):
        inspiration = inspirations[idx]
        if isinstance(row, BaseException):
            print(f"[skip] #{idx} LLM failure: {row}; prompt excerpt={inspiration[:96]!r}")
            continue
        env_llm = row

        mode = _mode_for_idx(idx)
        lineup, n_agents = _lineup_n_for_mode(mode)
        quartet_eff = lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL and n_agents == 4

        label, params_eff = presets[variant_i % len(presets)]
        variant_i += 1

        codename = f"ltr_llm_{args.tag}_{label}_{mode}_i{idx}"
        gm = build_negotiation_game_metadata_bundle(
            codename,
            quartet_eff,
            params_eff,
            num_participants=n_agents,
            lineup=lineup,
            scenario_text=str(getattr(env_llm, "scenario", "") or ""),
        )

        merged_gm = {**(dict(env_llm.game_metadata) if isinstance(env_llm.game_metadata, dict) else {}), **gm}
        env_llm.game_metadata = merged_gm
        env_llm.codename = codename
        env_llm.source = f"llm_long_term_negotiation:{args.model}"
        env_llm.tag = args.tag

        if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
            if n_agents == 4:
                firms4_goal_padding(env_llm)
            elif n_agents == 3:
                firms3_goal_padding(env_llm)
        elif n_agents == 4:
            quartet_goal_padding(env_llm)
        elif n_agents == 3:
            tri_goal_padding(env_llm)

        env_llm.save()
        env_modes_by_codename[codename] = mode
        env_lineup_by_codename[codename] = lineup
        print(
            f"  [save] EnvProfile pk={env_llm.pk[:8]}... codename={codename} "
            f"mode={mode} lineup={lineup} num_participants={n_agents}"
        )

        agent_bind_tag = f"{args.tag}__{codename}"
        if args.legacy_agent_profiles:
            agents = ltr.save_negotiation_agents(tag=agent_bind_tag)
        else:
            assert llm_roles_for_agents is not None
            agents = await agenerate_negotiation_agent_profiles(
                roles=ltr.QUARTET_ROSTER_ORDER,
                model_name=agent_profile_model,
                tag=agent_bind_tag,
                concurrency=max(1, args.concurrency),
                save_to_storage=True,
                llm_roles=llm_roles_for_agents,
            )
        ltr.pairwise_strangers(agents, tag=agent_bind_tag)
        v2_agents = ltr.save_negotiation_agent_profiles_v2(agents, tag=agent_bind_tag)

        roles = _roles_for_mode(mode)
        combo = ltr.save_combo(env_llm, roles, agents)
        combos_by_codename[codename] = combo
        legacy_env_objs.append(env_llm)
        env_agent_pks_by_codename[codename] = {r: agents[r].pk for r in ltr.QUARTET_ROSTER_ORDER}
        env_agent_v2_pks_by_codename[codename] = {r: v2_agents[r].pk for r in ltr.QUARTET_ROSTER_ORDER}

        profile_rows = [
            agent_profile_to_jsonable(
                agents[r],
                role=r,
                profile_source=(
                    "handwritten"
                    if args.legacy_agent_profiles
                    else ("llm" if r in llm_role_set else "static")
                ),
            )
            for r in ltr.QUARTET_ROSTER_ORDER
        ]
        per_environment_agent_export_rows.append(
            {
                "codename": codename,
                "environment_pk": env_llm.pk,
                "agent_bind_tag": agent_bind_tag,
                "agent_profiles": profile_rows,
            }
        )
        print(
            f"  [save] AgentProfile+V2 bound to env codename={codename} "
            f"(roles in combo: {', '.join(roles)})"
        )

        ltr.persist_scenario_v2(
            env_llm,
            quartet=quartet_eff,
            params=params_eff,
            tag=args.tag,
            event_anchor_pk=anchor_pk,
            v2_by_role=v2_agents,
            num_participants=n_agents,
            lineup=lineup,
        )

    ltr.save_environment_list_for_combos(legacy_env_objs, combos_by_codename)

    if agent_profile_path is not None and per_environment_agent_export_rows:
        agent_profile_path.parent.mkdir(parents=True, exist_ok=True)
        agent_profile_path.write_text(
            json.dumps(
                {
                    "tag": args.tag,
                    "binding": "per_environment",
                    "source": "generate_long_term_negotiation_llm.py:agent_profiles",
                    "model": agent_profile_model,
                    "agent_profile_source": agent_profile_source,
                    "llm_roles": list(llm_roles_for_agents) if llm_roles_for_agents is not None else None,
                    "environments": per_environment_agent_export_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[save] per-environment AgentProfile export -> {agent_profile_path}")

    manifest = {
        "tag": args.tag,
        "source": "generate_long_term_negotiation_llm.py",
        "model": args.model,
        "agent_profile_model": agent_profile_model,
        "agent_profile_source": agent_profile_source,
        "agent_profile_export_path": str(agent_profile_path) if agent_profile_path else None,
        "agent_roles": list(ltr.QUARTET_ROSTER_ORDER),
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
            "llm_profiles_requested": (
                len(explicit_plan) if explicit_plan is not None else args.n
            ),
            "concurrency": args.concurrency,
            "modes_cycle": modes_cycle,
            "mode_counts_spec": (getattr(args, "mode_counts", "") or None),
            "mode_counts_resolved": (
                {m: explicit_plan.count(m) for m in dict.fromkeys(explicit_plan)}
                if explicit_plan is not None
                else None
            ),
            "timeline_labels_filter": sorted(label_filter) if label_filter else None,
            "presets_used": [p[0] for p in presets],
            "environments_saved": len(legacy_env_objs),
            "requirements_notes": (args.requirements.strip() or None),
            "agent_profile_source": agent_profile_source,
            "agent_profile_model": agent_profile_model,
            "agent_profile_llm_roles": list(llm_roles_for_agents) if llm_roles_for_agents is not None else None,
            "agent_profiles_binding": "per_environment",
        },
    }
    out_name = args.manifest_name
    manifest_path = LOCAL_DATA_DIR / str(out_name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] manifest {manifest_path}")
    print("\n========== DONE ==========")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-4o-mini", help="agenerate_env_profile model_name（LiteLLM 路由键）")
    ap.add_argument("--n", type=int, default=3, help="生成条数上限（会与 inspiration 列表截断对齐）")
    ap.add_argument(
        "--modes",
        default="bilat",
        help=(
            "逗号分隔、保序轮转：bilat（2 人，with_institutional）/ tri（3 人）/ quartet（4 人）/ "
            "firms3（3 家公司，无机构位）/ firms4（4 家公司，无机构位）；对第 i 条 LLM 结果按列表循环取模式。"
            "若同时指定 --mode-counts，则忽略本参数。"
        ),
    )
    ap.add_argument(
        "--mode-counts",
        default="",
        help=(
            "按模式精确指定生成条数：MODE=N[,MODE=N...]；例 firms3=8,firms4=12,bilat=5。"
            "传入后总条数 = 各 N 之和，--n 与 --modes 仅在未传时生效。合法 MODE 与 --modes 相同。"
        ),
    )
    ap.add_argument(
        "--timeline-labels",
        default="",
        help="逗号分隔，仅使用这些时间轴预设标签（如 D6,D12）；空表示全部",
    )
    ap.add_argument(
        "--requirements",
        default="",
        help="自由文本，写入 manifest generation_spec.requirements_notes",
    )
    ap.add_argument("--tag", default="ltr_llm_benchmark")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument(
        "--inspiration",
        action="append",
        dest="inspiration",
        default=None,
        help="自定义灵感句（可多传）；不传则用内置 negotiation 句式",
    )
    ap.add_argument(
        "--manifest-name",
        default="long_term_negotiation_llm_manifest.json",
        help="manifest 文件名（写入 ~/.sotopia/data）",
    )
    ap.add_argument(
        "--agent-profile-model",
        default=None,
        help="生成 AgentProfile 用的 LLM；不传则复用 --model",
    )
    ap.add_argument(
        "--agent-profile-out",
        default="long_term_negotiation_llm_agent_profiles.json",
        help="按环境汇总的 AgentProfile JSON（binding=per_environment；写入 ~/.sotopia/data）",
    )
    ap.add_argument(
        "--agent-profiles-all-llm",
        action="store_true",
        help="四角色均用 LLM（含 investor/regulator）；默认仅 firm_a/firm_b 两家公司走 LLM",
    )
    ap.add_argument(
        "--legacy-agent-profiles",
        action="store_true",
        help="不调 LLM，沿用 generate_long_term_negotiation_scenarios.save_negotiation_agents 的常量画像",
    )
    args = ap.parse_args()

    load_mod = _load_handwritten_generator()
    return asyncio.run(main_async(args, load_mod))


if __name__ == "__main__":
    sys.exit(main())
