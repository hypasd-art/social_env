#!/usr/bin/env python
"""用大模型生成 **长期谈判题库** EnvironmentProfile + AgentProfile（与 ``agenerate_env_profile`` / ``design_1`` 对齐）。

1. 调 ``agenerate_env_profile`` 生成自然语言 ``scenario`` 与 ``agent_goals``（老 Sotopia 结构）。
2. 合并 ``scenario_loader.build_negotiation_game_metadata_bundle`` —— timeline / lineup 与手写脚本同源，
   ``predefined_outcome_rule`` 默认再混入 ``secrets`` 随机熵（每条环境不同）；``--deterministic-outcome-rule``
   时与手写脚本一样仅由 codename/阵容/场景文前缀决定种子。
3. **AgentProfile**：**每合成一条环境**即合成并落库一套 **参与会话的角色** 对应 ``AgentProfile`` / ``AgentProfileV2``（见下），再写入
   该环境的 ``EnvAgentComboStorage`` 与 V2 快照，与环境一一绑定。仅 **公司角色 firm_a..firm_d** 可走 LLM；
   本脚本生成的场景均为 ``firms_only`` roster，**不包含** investor/regulator 会话位，亦**不会**用大模型扮演二者。
   ``--legacy-agent-profiles`` 为每环境手写占位画像。
4. 复用手写脚本 ``generate_long_term_negotiation_scenarios.py`` 的 ``EnvAgentComboStorage`` /
   ``persist_scenario_v2``，保持 V2 快照与 benchmark 数据结构一致。

支持的 ``--modes``（**仅** ``firms_only``；合法 token：``firms2`` / ``firms3`` / ``firms4``）。

依赖 ``social_env/.env`` 里的 ``OPENAI_API_KEY``（及可选 BASE_URL）。

用法::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local python scripts/generate_long_term_negotiation_llm.py \\
        --model gpt-4o-mini --n 3 --modes firms2 --tag ltr_llm_v1

    # 规模：条数 --n；并发 --concurrency；时间轴仅用 D6/D8；模式按列表对每条 profile 轮转
    python scripts/generate_long_term_negotiation_llm.py --n 12 --concurrency 4 \\
        --timeline-labels D6,D8 --modes firms2,firms3,firms4 --tag ltr_llm_scale

    # 仅 3 / 4 方
    python scripts/generate_long_term_negotiation_llm.py --n 6 --modes firms3,firms4 \\
        --timeline-labels D6,D8 --tag ltr_llm_firms_only

    # 精确指定每种人数的生成条数（不按 --modes 轮转）：
    # 8 条 firms3 + 12 条 firms4 + 4 条 firms2
    python scripts/generate_long_term_negotiation_llm.py \\
        --mode-counts firms3=8,firms4=12,firms2=4 \\
        --timeline-labels D6,D8 --concurrency 4 --tag ltr_llm_mix

    # 要求说明写入 manifest（generation_spec）
    python scripts/generate_long_term_negotiation_llm.py --n 2 --requirements "仅用于 smoke；需人工抽检 scenario" --tag ltr_smoke

    # 用更轻的 agent_profile 模型 + 自定义导出文件名
    python scripts/generate_long_term_negotiation_llm.py --n 3 --agent-profile-model gpt-4o-mini \\
        --agent-profile-out long_term_negotiation_llm_agent_profiles.smoke.json

    # 想保留旧的硬编码 AgentProfile（不调 LLM 造画像）
    python scripts/generate_long_term_negotiation_llm.py --n 3 --legacy-agent-profiles

    # 第二轮起：传上一轮 manifest，先按 scene_type 用大模型总结已有样本，再在每条 inspiration 上强制反重复
    python scripts/generate_long_term_negotiation_llm.py --n 6 --tag ltr_v2 \\
        --diversity-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \\
        --diversity-model gpt-4o-mini --diversity-digest-out ~/.sotopia/data/ltr_diversity_digest.json

    # 不写库、只看 prompt / 配额
    python scripts/generate_long_term_negotiation_llm.py --dry-run --n 5
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import secrets
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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
from sotopia.generation_utils.generate import agenerate, agenerate_env_profile  # noqa: E402
from sotopia.generation_utils.output_parsers import PydanticOutputParser  # noqa: E402
from sotopia.settings.long_term_negotiation.llm_agent_profile_gen import (  # noqa: E402
    DEFAULT_COMPANY_LLM_ROLES,
    agenerate_negotiation_agent_profiles,
    agent_profile_to_jsonable,
)
from sotopia.settings.long_term_negotiation.scenario_loader import (
    DIALOGUE_STYLE_SYNTHESIS_APPEND_EN,
    build_negotiation_game_metadata_bundle,
    environment_pks_from_manifest,
)
from sotopia.settings.long_term_negotiation.types import (  # noqa: E402
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NegotiationTimelineParams,
    SESSION_FIRMS_ONLY_ROLE_ORDER,
    SESSION_SPEAKER_ROLE_ORDER,
)

LOCAL_DATA_DIR = Path(os.path.expanduser("~/.sotopia/data"))


class PriorCorpusSceneDigest(BaseModel):
    """LLM-compressed digest of prior samples in one scene_type (for anti-repetition in the next batch)."""

    task_scenarios_covered: str = Field(
        ...,
        description="Bullet-style: recurring negotiation task shapes already present (not verbatim copy targets).",
    )
    participant_and_object_traits: str = Field(
        ...,
        description="Who trades what; typical actor archetypes and commercial objects already used.",
    )
    overlapping_tropes_to_avoid: str = Field(
        ...,
        description="Concrete tropes, micro-industries, product bundles, or clichés the next sample must NOT reuse.",
    )
    suggested_novel_axes: str = Field(
        ...,
        description="3–6 differentiation directions still underrepresented within this scene_type.",
    )


def _clip(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def load_prior_exemplars_by_scene_from_manifest(
    manifest_path: Path,
    *,
    max_profiles_per_scene: int,
) -> dict[str, list[dict[str, str]]]:
    """从已有 LLM/manifest JSON 读取环境 pk，按 ``scene_type`` 分组，抽取 scenario / goals 片段。"""
    path = manifest_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"diversity manifest not found: {path}")
    pks = environment_pks_from_manifest(path)
    manifest_rows: dict[str, dict[str, Any]] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for row in raw.get("environments") or []:
            if isinstance(row, dict) and row.get("pk"):
                manifest_rows[str(row["pk"])] = row
    except (OSError, json.JSONDecodeError):
        manifest_rows = {}

    by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pk in pks:
        row = manifest_rows.get(pk) or {}
        scene = str(row.get("scene_type") or "").strip()
        env = EnvironmentProfile.get(pk)
        if env is None:
            continue
        gm = dict(env.game_metadata) if isinstance(env.game_metadata, dict) else {}
        if not scene:
            hint = gm.get("scene_type_hint")
            scene = str(hint).strip() if hint else "unknown"
        scenario = _clip(str(getattr(env, "scenario", "") or ""), 1400)
        goals = env.agent_goals or []
        goals_txt = _clip("\n---\n".join(str(g) for g in goals[:6]), 900)
        codename = str(getattr(env, "codename", "") or row.get("codename") or pk[:12])
        if len(by_scene[scene]) >= max(1, max_profiles_per_scene):
            continue
        by_scene[scene].append(
            {
                "codename": codename,
                "pk": pk,
                "scenario_excerpt": scenario,
                "agent_goals_excerpt": goals_txt,
            }
        )
    return dict(by_scene)


def _format_corpus_for_digest(rows: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for i, r in enumerate(rows, start=1):
        parts.append(
            f"### Prior sample {i} (codename={r.get('codename', '')})\n"
            f"scenario_excerpt:\n{r.get('scenario_excerpt', '')}\n\n"
            f"agent_goals_excerpt:\n{r.get('agent_goals_excerpt', '')}\n"
        )
    return "\n".join(parts).strip()


async def agenerate_scene_diversity_digest(
    *,
    scene_type: str,
    corpus_text: str,
    model_name: str,
) -> PriorCorpusSceneDigest | None:
    """调用大模型：按场景类型总结已有语料，并给出显式「勿重复」约束要点。"""
    if not corpus_text.strip():
        return None
    template = """You distill prior synthetic negotiation benchmark drafts for diversity control.

Fixed taxonomy label for this batch: {scene_type}

Corpus (excerpts of scenario + agent goals from earlier generations, possibly truncated):
---
{corpus}
---

Return structured fields that a later author-model will use to AVOID near-duplicates:
- Summarize recurring **task shapes** and **commercial objects** (not copy-paste targets).
- List **specific tropes/settings/product bundles** that the next sample must NOT reuse.
- Suggest **novel axes** still underrepresented within this scene_type.

{format_instructions}
"""
    try:
        return await agenerate(
            model_name=model_name,
            template=template,
            input_values={
                "scene_type": scene_type,
                "corpus": corpus_text,
            },
            output_parser=PydanticOutputParser(pydantic_object=PriorCorpusSceneDigest),
            temperature=0.15,
            structured_output=True,
        )
    except BaseException as exc:  # noqa: BLE001
        print(f"[warn] diversity digest LLM failed scene_type={scene_type!r}: {exc}")
        return None


def _diversity_prompt_block(scene_type: str, digest: PriorCorpusSceneDigest | None) -> str:
    if digest is None:
        return ""
    payload = digest.model_dump()
    return (
        "\n\n[prior_corpus_digest — same scene_type cohort]\n"
        f"scene_type={scene_type}\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n[anti_repetition_mandate — binding]\n"
        "You MUST invent a clearly **novel** scenario within this scene_type. Do NOT reuse the same "
        "industry micro-niche, product bundle, venue naming pattern, or conflict hinge described under "
        "`overlapping_tropes_to_avoid`. Do not lightly paraphrase prior samples; pick a fresh commercial "
        "situation and new concrete specifics. Prefer at least one angle from `suggested_novel_axes`.\n"
    )


def _extract_profile_excerpt(env_profile: EnvironmentProfile, codename: str = "") -> dict[str, str]:
    """从刚生成的 EnvironmentProfile 提取 scenario/goals 摘要，用于批内增量约束。"""
    scenario = _clip(str(getattr(env_profile, "scenario", "") or ""), 800)
    goals = env_profile.agent_goals or []
    goals_txt = _clip("\n---\n".join(str(g) for g in goals[:4]), 500)
    return {
        "codename": codename or getattr(env_profile, "codename", "") or "",
        "scenario_excerpt": scenario,
        "agent_goals_excerpt": goals_txt,
    }


def _batch_corpus_block(scene_type: str, batch_items: list[dict[str, str]]) -> str:
    """将批内已生成的条目格式化为反重复约束块。"""
    if not batch_items:
        return ""
    parts: list[str] = []
    for i, item in enumerate(batch_items, start=1):
        parts.append(
            f"  - [in-batch sample {i}] codename={item.get('codename', '')}\n"
            f"    scenario: {item.get('scenario_excerpt', '')}\n"
            f"    agent_goals: {item.get('agent_goals_excerpt', '')}"
        )
    return (
        "\n\n[in_batch_corpus — same scene_type, already generated this run]\n"
        f"scene_type={scene_type}\n"
        "Already generated in this batch (DO NOT repeat or lightly paraphrase):\n"
        + "\n".join(parts)
        + "\n\n[in_batch_anti_repetition_mandate — binding]\n"
        "You MUST NOT replicate any of the above in-batch samples. Pick a clearly "
        "different commercial situation, industry niche, product bundle, and conflict "
        "structure than ALL listed samples.\n"
    )


async def build_scene_diversity_digests(
    *,
    manifest_path: Path | None,
    scenario_mix: tuple[str, ...],
    model_name: str,
    max_profiles_per_scene: int,
) -> dict[str, PriorCorpusSceneDigest | None]:
    """对 manifest 中出现的各 scene_type 分别调用一次摘要 LLM。"""
    if manifest_path is None:
        return {}
    grouped = load_prior_exemplars_by_scene_from_manifest(
        manifest_path,
        max_profiles_per_scene=max_profiles_per_scene,
    )
    if not grouped:
        print(f"[diversity] no prior exemplars loaded from {manifest_path}")
        return {s: None for s in scenario_mix}
    out: dict[str, PriorCorpusSceneDigest | None] = {}
    # 仅对本次 synthesis 会用到的 scene_type 建摘要（也可包含 grouped 里多出的类型）
    scenes_to_cover = list(dict.fromkeys(list(scenario_mix) + sorted(grouped.keys())))
    for scene in scenes_to_cover:
        rows = grouped.get(scene, [])
        if not rows:
            out[scene] = None
            continue
        corpus = _format_corpus_for_digest(rows)
        print(
            f"[diversity] summarizing scene_type={scene!r} with {len(rows)} prior profile(s); "
            f"model={model_name}"
        )
        out[scene] = await agenerate_scene_diversity_digest(
            scene_type=scene,
            corpus_text=corpus,
            model_name=model_name,
        )
    return out


def apply_diversity_blocks_to_inspirations(
    inspirations: list[str],
    scene_types_by_idx: list[str],
    scenario_mix: tuple[str, ...],
    digests: dict[str, PriorCorpusSceneDigest | None],
) -> list[str]:
    """按每条 inspiration 的 scene_type 注入摘要与反重复约束。"""
    out: list[str] = []
    for i, ins in enumerate(inspirations):
        scene = (
            scene_types_by_idx[i]
            if i < len(scene_types_by_idx)
            else scenario_mix[i % len(scenario_mix)]
        )
        out.append(ins + _diversity_prompt_block(scene, digests.get(scene)))
    return out


def _load_handwritten_generator() -> Any:
    p = REPO_ROOT / "scripts" / "generate_long_term_negotiation_scenarios.py"
    spec = importlib.util.spec_from_file_location("ltr_gen_manual", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 追加到 3+ 人场景的 inspiration 末尾，强制「N 个同类型竞争者 + 1 个选择者」博弈格局。
MULTI_PARTY_COMPETITION_STRUCTURE = """
[competition_structure — MANDATORY for scenarios with >=3 participants]
This scenario MUST contain explicit multi-party rivalry. The following structure is NOT optional:

1) **At least TWO participants are SAME-TYPE competitors** — they offer the same kind of resource
   (same product category, same delivery service, same labor pool, same contract bid). They share
   overlapping customer segments / resource windows / contract opportunities. One competitor's
   gain is directly the other's loss.
2) **At least ONE participant is a CHOOSER** — this agent must explicitly compare offers from the
   same-type competitors and decide whom to contract with. The chooser's goal MUST reference
   comparing multiple counterparties (not just negotiating with one).
3) **Competitor goals MUST conflict directly** — at least one pair of same-type competitors must
   have goals that cannot both be fully satisfied. Examples:
   - Both need the same anchor customer whose budget only covers one supplier
   - Both need the same scarce delivery slot / cold-chain window
   - Both bid for the same fixed-budget contract (winner takes all or most)
   - Both claim the same limited-capacity resource with zero-sum allocation
4) **The chooser's goal must encode SELECTION LOGIC** — the chooser weighs competing offers
   by explicit criteria (price, quality, delivery time, trust/reputation, penalty terms) and
   must engage with ALL same-type competitors before committing to any deal.
5) **Rejection consequences** — the competitor NOT chosen must face real costs (lost revenue,
   idle capacity, spoilage risk, reputational damage) that create genuine pressure.

If the scenario does not satisfy ALL five points above, it is INVALID and must be regenerated.
""".strip()

SCENARIO_PROMPT_BANK: dict[str, list[str]] = {
    "business_coopetition": [
        "Small-business coopetition: several independent shop owners negotiate a joint procurement or distribution deal "
        "while still competing for overlapping customers; each side balances collaboration gains against rivalry risk.",
        "Neighborhood merchants consider a temporary alliance on sourcing and delivery slots, but each participant also "
        "wants favorable terms that preserve their own pricing edge and repeat-customer base.",
        "Independent operators co-negotiate a shared contract to reduce costs, yet compete on service quality and final "
        "customer conversion; allocation, commitments, and fallback clauses are all contested.",
        "Community commerce coopetition: personal vendors discuss pooled purchasing and shared logistics under volatile "
        "demand, while protecting their own margin, reputation, and client retention.",
    ],
    "wet_market_competition": [
        "Morning wet-market procurement: an individual household buyer negotiates with multiple produce stall owners "
        "across several days. Rival sellers quote overlapping baskets; customers choose by total value "
        "(price, freshness, delivery slot, refund promise).",
        "Three-person street-commerce rivalry: a buyer lead, an incumbent stall, and a challenger stall negotiate "
        "overlapping bundles while competing for the same walk-by customers and word-of-mouth reputation.",
        "Weekend farmers' lane: a home cook sources meat, tofu, and herbs from rival stalls; sellers cite rival "
        "prices openly and offer tasting samples as leverage.",
        "Apartment block group order: neighbors coordinate a split delivery of rice, cooking oil, and fruit; one "
        "coordinator negotiates minimum order sizes and per-unit splits with two competing mini-vendors.",
    ],
    "business_outsourcing": [
        "Micro-factory outsourcing: three independent workshop owners negotiate labor allocation, milestone acceptance, "
        "and rework responsibility under tight workforce constraints.",
        "Neighborhood cafe supply outsourcing: independent operators bargain over beans, milk, and pastry batches while "
        "committing delivery SLA windows and quality penalties.",
        "Repair plus parts outsourcing: a landlord compares two freelance fixers quoting labor plus replacement parts; "
        "callback duty, milestone timing, and warranty wording are central.",
        "Community canteen prep outsourcing: buyer-side organizer negotiates with personal vendors for prep labor and "
        "rolling replenishment; workforce availability and acceptance criteria drive risk.",
    ],
    "resource_scheduling_management": [
        "Resource scheduling management: multiple individual operators coordinate scarce labor, delivery windows, and "
        "equipment slots over several days; agreements must resolve time conflicts and execution priorities.",
        "Cross-vendor schedule orchestration: independent sellers negotiate who gets limited cold-chain capacity, loading "
        "times, and courier bandwidth, with penalties for overruns and missed handoffs.",
        "Market-day capacity management: personal merchants bargain over shift assignments, storage access, and dispatch "
        "sequence so that urgent orders can be fulfilled without collapsing baseline operations.",
        "Distributed operations planning among micro-business owners: they must allocate people, vehicles, and inventory "
        "refresh windows under uncertainty, while preserving cashflow and service reliability.",
    ],
    "competitive_bidding": [
        "Local school supplies tender: multiple individual vendors submit bids for backpacks, notebooks, and art kits; "
        "the buyer compares headline price, compliance terms, and delivery risk.",
        "Dorm procurement mini-auction: student treasurer requests bids from competing informal suppliers for snacks and "
        "daily essentials; reserve price pressure and undercut tactics appear each round.",
        "Community event catering tender: independent cooks bid for a fixed budget package; scoring balances technical "
        "fit, execution plan, and final bid spread.",
        "Household bulk purchase auction: several personal shoppers compete with explicit bid revisions; the buyer "
        "tracks total landed cost and hidden fee clauses.",
    ],
}
DEFAULT_SCENARIO_MIX: tuple[str, ...] = (
    "business_coopetition",
    "wet_market_competition",
    "resource_scheduling_management",
)

SCENARIO_PROMPT_GUIDE: dict[str, str] = {
    "business_coopetition": (
        "Generate a long-horizon negotiation scenario for independent small-business operators in a coopetition setting.\n"
        "Competitive core: market share is zero-sum — one operator's gain directly reduces another's customer base.\n"
        "Hard requirements:\n"
        "1) Participants are individuals (small bosses), not corporate departments.\n"
        "2) The same parties both cooperate (joint sourcing / shared logistics / pooled commitments) and aggressively compete\n"
        "   (overlapping customers, margin pressure, reputation race, customer poaching).\n"
        "3) At least one operator faces existential cashflow pressure — failure to secure favorable terms this episode\n"
        "   means real risk of insolvency within days, not quarters.\n"
        "4) Include at least one multi-day trade-off where short-term concession may improve long-term position,\n"
        "   BUT include at least one zero-sum fork where one party's win is another's direct loss.\n"
        "5) Goals MUST be directly conflicting across at least one pair of participants (e.g. both need the same\n"
        "   anchor customer, the same scarce supply batch, or the same delivery slot).\n"
        "6) Include winner-takes-most dynamics: the operator who signs first may lock in advantages (exclusive terms,\n"
        "   volume discounts, preferred slots) that weaken the bargaining position of latecomers.\n"
        "7) Keep language concrete (price, delivery window, quality guarantees, penalties, fallback clauses,\n"
        "   customer retention metrics, competitive threat names)."
    ),
    "wet_market_competition": (
        "Generate a long-horizon individual-trader market scenario under a rule-driven social simulation setup.\n"
        "Competitive core: rival sellers share overlapping SKUs and the same walk-by customer pool — every\n"
        "sale one vendor makes is a sale the other loses. Price wars, reputation attacks, and customer\n"
        "poaching are expected dynamics.\n"
        "Use this conceptual model when writing scenario and goals:\n"
        "- World = (Agents, State, Mechanisms, ExogenousEvents, Utility).\n"
        "- Agents are individual merchants with personality factors: risk_tolerance, honesty, cooperation,\n"
        "  long_term_orientation, aggressiveness (all interpreted in [0,1]).\n"
        "- Trading is shaped by trust/reputation/cashflow pressure, not formal corporate hierarchy.\n"
        "- Include dynamics consistent with these mechanisms:\n"
        "  * Trade formation influenced by price attractiveness, bilateral trust, inventory pressure, personality bias.\n"
        "  * Default risk increases when honesty is low and financial pressure is high.\n"
        "  * Trust updates after successful fulfillment, default, and delay.\n"
        "  * Price and inventory evolve with demand/supply and spoilage realities.\n"
        "  * Rival vendors may spread negative gossip or undercut prices to steal customers — reputation is a weapon.\n"
        "  * Price wars can trigger downward spirals: one undercut forces others to match, squeezing all margins.\n"
        "- Include at least one exogenous shock candidate (price shock / rumor / logistics disruption / policy change).\n"
        "- Hard competitive constraint: at least two sellers MUST be directly competing for the SAME customer segment,\n"
        "  and the buyer's goal MUST weigh fresh rivalry signals (who is cheaper today? who has better delivery? who\n"
        "  can be trusted after last week's default rumor?).\n"
        "- The resulting goals must support deterministic rule-based evaluation (economic + social + stability outcomes),\n"
        "  avoiding vague purely emotional objectives."
    ),
    "resource_scheduling_management": (
        "Generate a long-horizon negotiation scenario centered on resource scheduling management among independent operators.\n"
        "Competitive core: scarce resources (labor shifts, delivery slots, cold-chain capacity, vehicles, loading docks)\n"
        "have fixed total capacity — every slot one operator claims is a slot another operator CANNOT use.\n"
        "This is a zero-sum scheduling contest, not a coordination exercise.\n"
        "Use a dialogue-driven allocation lens with explicit three-stage rule structure:\n"
        "Stage 1 (Demand scoring):\n"
        "- Start from base_demand_{i,k}; amplify by urgency/aggressiveness; include trust-based demand correction.\n"
        "Stage 2 (Priority computation):\n"
        "- Priority depends on demand, cooperation, reputation, trust-network influence, and overlap conflict penalties.\n"
        "- Personality effects must be visible: high aggressiveness can raise short-term priority; low honesty carries penalty risk.\n"
        "Stage 3 (Allocation):\n"
        "- Allocation should be computable by softmax-style share or constrained optimization under finite capacity.\n"
        "- Include layered fallback priorities: survival-critical obligations first, then contractual obligations, then market demand.\n"
        "Hard requirements:\n"
        "1) Scarce resources must be explicit: labor shifts, delivery slots, cold-chain/storage, vehicles, loading docks.\n"
        "2) Conflicts are primarily timing/allocation conflicts with ZERO-SUM stakes — one operator's slot win directly blocks another.\n"
        "3) Include strategic behaviors: preemptive claiming of slots, hoarding of capacity, undercutting bids for prime times,\n"
        "   retaliatory overbooking, and bluffing about urgency to jump the queue.\n"
        "4) Include operational constraints: deadlines, overrun penalties, service reliability, dependency chains.\n"
        "5) At least two operators MUST have overlapping slot requirements, creating direct scheduling rivalry.\n"
        "6) Dialogue must act as control signal (belief update -> urgency shift -> demand/priority change) AND as competitive\n"
        "   maneuvering (signaling higher need, questioning rivals' claimed urgency, leveraging past favors).\n"
        "7) Goals should encode executable scheduling priorities, concession paths, renegotiation triggers, and explicit\n"
        "   competitive tactics against named rivals for the same resource windows."
    ),
    "business_outsourcing": (
        "Generate a long-horizon negotiation scenario centered on outsourcing contracts among independent operators.\n"
        "Competitive core: multiple subcontractors bid for the same work; the buyer squeezes margins by pitting\n"
        "vendors against each other. Rework responsibility becomes a weapon — each party tries to shift blame and\n"
        "cost to others. Milestone acceptance is a battleground, not a formality.\n"
        "Hard requirements:\n"
        "1) At least two vendors compete for the same outsourcing contract; the buyer explicitly compares bids.\n"
        "2) Include milestone-gaming dynamics: vendors may underbid then inflate change orders; buyers may reject\n"
        "   milestones to extract concessions; partial performance creates attribution disputes.\n"
        "3) Rework economics must be explicit: who pays for defects, at what rate, with what delay penalty.\n"
        "4) Subcontractor chain pressure: tier-2 vendors squeezing tier-1 creates cascading margin stress.\n"
        "5) Include at least one SLA breach scenario where penalty clauses trigger and must be renegotiated.\n"
        "6) Goals must create direct cost-shifting conflict: each party wants the other to absorb rework/milestone risk.\n"
        "7) Keep language concrete: acceptance criteria, defect rates, penalty per day, payment milestones, warranty scope."
    ),
    "competitive_bidding": (
        "Generate a long-horizon negotiation scenario centered on competitive bidding among individual vendors.\n"
        "Competitive core: multiple bidders compete for a fixed-budget contract; undercutting is the primary weapon\n"
        "but carries risk of winning unprofitable work. Information asymmetry (who knows the reserve price? whose\n"
        "cost structure is leaner?) drives strategic misrepresentation.\n"
        "Hard requirements:\n"
        "1) At least three bidders compete for the same contract; the buyer has a hard budget cap.\n"
        "2) Include bid-spread dynamics: bidders must choose between aggressive undercutting (win now, thin margin)\n"
        "   and conservative pricing (lose now, preserve margin for future rounds).\n"
        "3) Technical compliance vs. headline price creates a strategic fork: lowest bidder may fail compliance,\n"
        "   highest-compliance bidder may be too expensive — the buyer's scoring weights are partially opaque.\n"
        "4) Include information leakage risk: rumors about rival bids, suspected collusion, or last-round visibility\n"
        "   that changes bidding behavior.\n"
        "5) Include at least one bidder facing a win-at-any-cost pressure (debt due, empty pipeline) that forces\n"
        "   risky undercutting against more stable competitors.\n"
        "6) Goals must create direct price competition: each bidder wants to undercut rivals just enough to win\n"
        "   without destroying their own margin; the buyer wants to extract the lowest compliant price.\n"
        "7) Keep language concrete: bid price, compliance score, reserve price, bid rounds, penalty for withdrawal."
    ),
}

_LLM_MODE_TO_LINEUP_N: dict[str, tuple[str, int]] = {
    "firms2": (NEGOTIATION_LINEUP_FIRMS_ONLY, 2),
    "firms3": (NEGOTIATION_LINEUP_FIRMS_ONLY, 3),
    "firms4": (NEGOTIATION_LINEUP_FIRMS_ONLY, 4),
}


def modes_cycle_from_arg(s: str) -> list[str]:
    """与 ``--modes`` 字符串顺序一致，对每条 LLM profile 轮转（合法 token 同手写脚本）。"""
    allow = frozenset(_LLM_MODE_TO_LINEUP_N)
    return [p.strip().lower() for p in s.split(",") if p.strip().lower() in allow] or ["firms3"]


def parse_mode_counts(spec: str) -> list[str] | None:
    """``--mode-counts MODE=N[,MODE=N...]`` 解析。

    例：``firms3=8,firms4=12,firms2=4`` -> 8 条 firms3 + 12 条 firms4 + 4 条 firms2
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


def _build_mixed_inspirations(
    *,
    count: int,
    scenario_mix: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """按给定场景类型循环生成灵感句，保证多场景覆盖。"""
    prompts: list[str] = []
    scene_tags: list[str] = []
    for i in range(max(1, count)):
        scene = scenario_mix[i % len(scenario_mix)]
        bank = SCENARIO_PROMPT_BANK.get(scene) or SCENARIO_PROMPT_BANK["wet_market_competition"]
        seed = bank[(i // len(scenario_mix)) % len(bank)]
        guide = SCENARIO_PROMPT_GUIDE.get(scene, "")
        if guide:
            prompts.append(
                f"[scene_type={scene}]\n"
                f"{guide}\n\n"
                f"[seed_story_hook]\n{seed}\n\n"
                "Output should remain realistic for small-merchant negotiation benchmark generation."
            )
        else:
            prompts.append(seed)
        scene_tags.append(scene)
    return prompts, scene_tags


def _validate_scene_coverage_capacity(total_jobs: int, scenario_mix: tuple[str, ...]) -> None:
    """硬约束：一次数据合成至少要有足够条数覆盖全部场景类型。"""
    need = len(scenario_mix)
    if total_jobs < need:
        raise ValueError(
            "insufficient jobs to cover all scenario types: "
            f"jobs={total_jobs}, required>={need}, scenario_mix={list(scenario_mix)}. "
            "Increase --n or --mode-counts total."
        )


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


async def _generate_with_incremental_diversity(
    *,
    inspirations: list[str],
    scene_types_by_idx: list[str],
    scenario_mix: tuple[str, ...],
    diversity_digests: dict[str, PriorCorpusSceneDigest | None],
    ds_suffix: str,
    sem: asyncio.Semaphore,
    model: str,
    mode_fn,
) -> list[EnvironmentProfile | BaseException]:
    """按 scene_type 分组，组内串行、组间并发生成，每次生成后更新批内语料。

    返回与 inspirations 原始索引对齐的 results 列表。
    """
    n = len(inspirations)

    # 1. 按 scene_type 分组，记录原始索引
    groups: dict[str, list[tuple[int, str]]] = {}
    for i, ins in enumerate(inspirations):
        scene = (
            scene_types_by_idx[i]
            if i < len(scene_types_by_idx)
            else scenario_mix[i % len(scenario_mix)]
        )
        groups.setdefault(scene, []).append((i, ins))

    # 2. 结果占位
    results: list[EnvironmentProfile | BaseException | None] = [None] * n

    # 3. 每个 scene_type 一个协程，内部串行
    async def _process_scene_type(scene: str, items: list[tuple[int, str]]) -> None:
        batch_corpus: list[dict[str, str]] = []
        for orig_idx, inspiration in items:
            # 拼装 prompt: inspiration + ds_suffix + prior digest + in-batch corpus
            prompt = inspiration + ds_suffix
            digest = diversity_digests.get(scene)
            prompt += _diversity_prompt_block(scene, digest)
            prompt += _batch_corpus_block(scene, batch_corpus)

            raw = await generate_one_llm_profile(sem, prompt, model)
            results[orig_idx] = raw

            if not isinstance(raw, BaseException):
                excerpt = _extract_profile_excerpt(raw)
                if excerpt["scenario_excerpt"] or excerpt["agent_goals_excerpt"]:
                    batch_corpus.append(excerpt)

    # 4. 所有 scene_type 并发
    group_tasks = [
        _process_scene_type(scene, items) for scene, items in groups.items()
    ]
    await asyncio.gather(*group_tasks)

    # 5. 类型收窄
    return [r if r is not None else RuntimeError("unexpected missing result") for r in results]


async def main_async(args: argparse.Namespace, ltr: Any) -> int:
    explicit_plan = parse_mode_counts(getattr(args, "mode_counts", "") or "")
    scene_mix_raw = tuple(
        x.strip()
        for x in (getattr(args, "scenario_mix", "") or "").split(",")
        if x.strip()
    )
    scenario_mix = scene_mix_raw or DEFAULT_SCENARIO_MIX
    bad_scene = [x for x in scenario_mix if x not in SCENARIO_PROMPT_BANK]
    if bad_scene:
        print(
            f"[err] unknown --scenario-mix item(s): {bad_scene}; "
            f"allowed={sorted(SCENARIO_PROMPT_BANK)}"
        )
        return 1
    if len(set(scenario_mix)) < 3:
        print(
            "[err] data synthesis must include three distinct scenario types; "
            f"got scenario_mix={list(scenario_mix)}"
        )
        return 1

    scene_types_by_idx: list[str] = []
    if explicit_plan is not None:
        try:
            _validate_scene_coverage_capacity(len(explicit_plan), scenario_mix)
        except ValueError as e:
            print(f"[err] {e}")
            return 1
        # 用户精确指定每种人数/公司数的条数；总条数由 plan 决定，不再用 --n。
        if args.n != 3 and len(explicit_plan) != args.n:
            print(
                f"[warn] --mode-counts fixes total LLM scenarios to {len(explicit_plan)}; "
                f"ignoring --n={args.n}"
            )
        if args.inspiration:
            pool = list(args.inspiration)
            inspirations = []
            for i in range(len(explicit_plan)):
                scene = scenario_mix[i % len(scenario_mix)]
                seed = pool[i % len(pool)]
                guide = SCENARIO_PROMPT_GUIDE.get(scene, "")
                inspirations.append(
                    f"[scene_type={scene}]\n{guide}\n\n[user_seed_inspiration]\n{seed}"
                    if guide
                    else seed
                )
            scene_types_by_idx = [scenario_mix[i % len(scenario_mix)] for i in range(len(explicit_plan))]
        else:
            inspirations, scene_types_by_idx = _build_mixed_inspirations(
                count=len(explicit_plan),
                scenario_mix=scenario_mix,
            )
    else:
        try:
            _validate_scene_coverage_capacity(max(1, args.n), scenario_mix)
        except ValueError as e:
            print(f"[err] {e}")
            return 1
        if args.inspiration:
            pool = list(args.inspiration)
            inspirations = []
            for i in range(max(1, args.n)):
                scene = scenario_mix[i % len(scenario_mix)]
                seed = pool[i % len(pool)]
                guide = SCENARIO_PROMPT_GUIDE.get(scene, "")
                inspirations.append(
                    f"[scene_type={scene}]\n{guide}\n\n[user_seed_inspiration]\n{seed}"
                    if guide
                    else seed
                )
            scene_types_by_idx = [scenario_mix[i % len(scenario_mix)] for i in range(len(inspirations))]
        else:
            inspirations, scene_types_by_idx = _build_mixed_inspirations(
                count=max(1, args.n),
                scenario_mix=scenario_mix,
            )

    _ds_suffix = "\n\n" + DIALOGUE_STYLE_SYNTHESIS_APPEND_EN + "\n"

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

    # ---- 竞争结构注入：仅对 3+ 人场景追加 MULTI_PARTY_COMPETITION_STRUCTURE ----
    def _mode_n(mode: str) -> int:
        """返回 mode 对应的参与人数（2/3/4）。"""
        try:
            return _LLM_MODE_TO_LINEUP_N[mode][1]
        except KeyError:
            return 2  # 未知 mode 保守按 2 人处理

    for i, ins in enumerate(inspirations):
        mode = _mode_for_idx(i)
        n_parts = _mode_n(mode)
        if n_parts >= 3 and MULTI_PARTY_COMPETITION_STRUCTURE not in ins:
            inspirations[i] = ins + "\n\n" + MULTI_PARTY_COMPETITION_STRUCTURE
    # ------------------------------------------------------------------------

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
            f"scenarios={list(scenario_mix)} presets={[p[0] for p in presets]} "
            f"model={args.model} concurrency={args.concurrency} "
            f"incremental_diversity={bool(args.incremental_diversity)}"
        )
        dm = (getattr(args, "diversity_manifest", None) or "").strip()
        if dm:
            try:
                g = load_prior_exemplars_by_scene_from_manifest(
                    Path(dm).expanduser(),
                    max_profiles_per_scene=max(1, int(getattr(args, "diversity_max_per_scene", 16))),
                )
                counts = {k: len(v) for k, v in sorted(g.items())}
                print(f"[dry-run][diversity] prior manifest={dm!r} counts_per_scene={counts}")
            except Exception as e:  # noqa: BLE001
                print(f"[dry-run][diversity] manifest read failed: {e}")
        for i, raw in enumerate(inspirations):
            mode = _mode_for_idx(i)
            scene = scene_types_by_idx[i] if i < len(scene_types_by_idx) else scenario_mix[i % len(scenario_mix)]
            p = raw + _ds_suffix
            print(f"  {i}: mode={mode} scene={scene} | {p[:100]}...")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("[err] OPENAI_API_KEY 未设置（检查 social_env/.env）")
        return 1

    # 保存原始 inspirations，供增量多样性模式使用
    raw_inspirations = list(inspirations)

    diversity_digests: dict[str, PriorCorpusSceneDigest | None] = {}
    dm = (getattr(args, "diversity_manifest", None) or "").strip()
    if dm:
        div_model = (getattr(args, "diversity_model", None) or "").strip() or args.model
        max_per = max(1, int(getattr(args, "diversity_max_per_scene", 16)))
        try:
            diversity_digests = await build_scene_diversity_digests(
                manifest_path=Path(dm).expanduser(),
                scenario_mix=scenario_mix,
                model_name=div_model,
                max_profiles_per_scene=max_per,
            )
            digest_out = (getattr(args, "diversity_digest_out", None) or "").strip()
            if digest_out:
                outp = Path(digest_out).expanduser()
                outp.parent.mkdir(parents=True, exist_ok=True)
                serial = {k: (v.model_dump() if v else None) for k, v in diversity_digests.items()}
                outp.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[diversity] wrote digest JSON -> {outp}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] diversity pipeline skipped: {e}")
            diversity_digests = {}
        if not args.incremental_diversity:
            # 非增量模式：直接注入所有 inspiration
            inspirations = apply_diversity_blocks_to_inspirations(
                inspirations, scene_types_by_idx, scenario_mix, diversity_digests
            )

    if not args.incremental_diversity:
        inspirations = [ins + _ds_suffix for ins in inspirations]

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
        if getattr(args, "agent_profiles_all_llm", False):
            print(
                "[warn] --agent-profiles-all-llm is ignored: this pipeline only samples LLM "
                "profiles for firm_a..firm_d; investor/regulator are not session roles here."
            )
        llm_roles_for_agents = tuple(DEFAULT_COMPANY_LLM_ROLES)
        print(
            f"[agent_profiles] mode=llm per-environment firm_roles_llm={list(llm_roles_for_agents)} "
            f"model={agent_profile_model}"
        )
        agent_profile_source = "llm"

    events = ltr.negotiation_event_scripts(args.tag)
    for ev in events:
        ev.save()
    anchor_pk = events[0].pk if events else None
    print(f"[save] EventScript anchor_pk={anchor_pk}")

    if args.incremental_diversity:
        print(
            f"[incremental-diversity] grouped by scene_type, sequential within group, "
            f"concurrent across groups; ds_suffix={bool(_ds_suffix)}"
        )
        sem = asyncio.Semaphore(max(1, args.concurrency))
        raw_profiles = await _generate_with_incremental_diversity(
            inspirations=raw_inspirations,
            scene_types_by_idx=scene_types_by_idx,
            scenario_mix=scenario_mix,
            diversity_digests=diversity_digests,
            ds_suffix=_ds_suffix,
            sem=sem,
            model=args.model,
            mode_fn=_mode_for_idx,
        )
    else:
        sem = asyncio.Semaphore(max(1, args.concurrency))
        tasks = [generate_one_llm_profile(sem, ins, args.model) for ins in inspirations]
        raw_profiles = await asyncio.gather(*tasks)

    variant_i = 0
    env_modes_by_codename: dict[str, str] = {}
    env_scene_type_by_codename: dict[str, str] = {}
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
        scene_type = (
            scene_types_by_idx[idx]
            if idx < len(scene_types_by_idx)
            else scenario_mix[idx % len(scenario_mix)]
        )
        lineup, n_agents = _lineup_n_for_mode(mode)

        label, params_eff = presets[variant_i % len(presets)]
        variant_i += 1

        scene_tag = scene_type.replace("_", "")
        codename = f"ltr_llm_{args.tag}_{label}_{mode}_{scene_tag}_i{idx}"
        outcome_rule_entropy: str | None = None
        if not args.deterministic_outcome_rule:
            outcome_rule_entropy = secrets.token_hex(16)
        gm = build_negotiation_game_metadata_bundle(
            codename,
            False,
            params_eff,
            num_participants=n_agents,
            lineup=lineup,
            scenario_text=str(getattr(env_llm, "scenario", "") or ""),
            outcome_rule_entropy=outcome_rule_entropy,
            scene_type_hint=scene_type,
        )
        active_roles = _roles_for_mode(mode)
        gm["social_graph"] = {
            "nodes": [
                {
                    "role": r,
                    "summary": ltr.ROLE_SUMMARY_EN.get(r, r),
                    "background_story": ltr._persona_for_role(r).get("background_story", ""),
                    "personality": ltr._persona_for_role(r).get("personality", ""),
                    "dialogue_voice": ltr._persona_for_role(r).get("dialogue_voice", ""),
                    "core_skills": list(ltr._persona_for_role(r).get("core_skills", [])),
                }
                for r in active_roles
            ],
            "edges": ltr._social_graph_edges(active_roles),
        }
        gm["agent_survival_constraints"] = {
            r: {
                "daily_fixed_cost": float(ltr._persona_for_role(r).get("daily_fixed_cost", 0.0) or 0.0),
                "short_term_debt_due": float(ltr._persona_for_role(r).get("short_term_debt_due", 0.0) or 0.0),
                "achievement_motivation": str(ltr._persona_for_role(r).get("achievement_motivation", "") or ""),
            }
            for r in active_roles
        }

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

        env_llm.save()
        env_modes_by_codename[codename] = mode
        env_scene_type_by_codename[codename] = scene_type
        env_lineup_by_codename[codename] = lineup
        print(
            f"  [save] EnvProfile pk={env_llm.pk[:8]}... codename={codename} "
            f"mode={mode} scene={scene_type} lineup={lineup} num_participants={n_agents}"
        )

        agent_bind_tag = f"{args.tag}__{codename}"
        if args.legacy_agent_profiles:
            agents = ltr.save_negotiation_agents(tag=agent_bind_tag, roles=active_roles)
        else:
            assert llm_roles_for_agents is not None
            llm_roles_active = tuple(r for r in llm_roles_for_agents if r in set(active_roles))
            agents = await agenerate_negotiation_agent_profiles(
                roles=active_roles,
                model_name=agent_profile_model,
                tag=agent_bind_tag,
                concurrency=max(1, args.concurrency),
                save_to_storage=True,
                llm_roles=llm_roles_active,
            )
        ltr.pairwise_strangers(agents, tag=agent_bind_tag, roles=active_roles)
        v2_agents = ltr.save_negotiation_agent_profiles_v2(agents, tag=agent_bind_tag, roles=active_roles)

        roles = _roles_for_mode(mode)
        combo = ltr.save_combo(env_llm, roles, agents)
        combos_by_codename[codename] = combo
        legacy_env_objs.append(env_llm)
        env_agent_pks_by_codename[codename] = {r: agents[r].pk for r in active_roles}
        env_agent_v2_pks_by_codename[codename] = {r: v2_agents[r].pk for r in active_roles}

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
            for r in active_roles
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
            quartet=False,
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
        "agent_roles": list(SESSION_FIRMS_ONLY_ROLE_ORDER),
        "agent_profiles_binding": "per_environment",
        "environments": [
            {
                "codename": e.codename,
                "pk": e.pk,
                "mode": env_modes_by_codename.get(e.codename),
                "scene_type": env_scene_type_by_codename.get(e.codename),
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
            "scenario_mix": list(scenario_mix),
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
            "diversity_manifest": ((getattr(args, "diversity_manifest", None) or "").strip() or None),
            "diversity_digest_model": (
                ((getattr(args, "diversity_model", None) or "").strip() or args.model)
                if (getattr(args, "diversity_manifest", None) or "").strip()
                else None
            ),
            "diversity_max_per_scene": int(getattr(args, "diversity_max_per_scene", 16)),
            "diversity_digest_out": (
                (getattr(args, "diversity_digest_out", None) or "").strip() or None
            ),
            "incremental_diversity": bool(args.incremental_diversity),
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
        default="firms3",
        help=(
            "逗号分隔、保序轮转：仅 firms2 / firms3 / firms4（均为 firms_only）。"
            "对第 i 条 LLM 结果按列表循环取模式。若同时指定 --mode-counts，则忽略本参数。"
        ),
    )
    ap.add_argument(
        "--mode-counts",
        default="",
        help=(
            "按模式精确指定生成条数：MODE=N[,MODE=N...]；例 firms3=8,firms4=12,firms2=4。"
            "传入后总条数 = 各 N 之和，--n 与 --modes 仅在未传时生效。合法 MODE：firms2/firms3/firms4。"
        ),
    )
    ap.add_argument(
        "--scenario-mix",
        default="business_coopetition,wet_market_competition,resource_scheduling_management",
        help=(
            "逗号分隔的场景类型轮转；默认固定三类，保证数据合成覆盖。"
            "合法值：business_coopetition,wet_market_competition,resource_scheduling_management,"
            "business_outsourcing,competitive_bidding"
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
        help="兼容旧 CLI：仅 firm_a..firm_d 可走 LLM；传此参数会告警并忽略（本脚本无 investor/regulator 会话）。",
    )
    ap.add_argument(
        "--legacy-agent-profiles",
        action="store_true",
        help="不调 LLM，沿用 generate_long_term_negotiation_scenarios.save_negotiation_agents 的常量画像",
    )
    ap.add_argument(
        "--deterministic-outcome-rule",
        action="store_true",
        help=(
            "predefined_outcome_rule 的种子不混入额外随机串，仅依赖 codename/阵容/人数/场景文前缀"
            "（与手写 generate_long_term_negotiation_scenarios 一致，便于复现）。默认关闭。"
        ),
    )
    ap.add_argument(
        "--diversity-manifest",
        default="",
        help=(
            "可选：指向上一次写出的 manifest JSON（含 environments[].pk / scene_type）。"
            "将按 scene_type 从库存加载 EnvironmentProfile，调用大模型生成「任务场景+对象特点」摘要，"
            "并注入每条 agenerate_env_profile 的 inspiration，要求避免与语料高度相似的样例。"
        ),
    )
    ap.add_argument(
        "--diversity-model",
        default="",
        help="摘要步骤使用的模型；默认与 --model 相同",
    )
    ap.add_argument(
        "--diversity-max-per-scene",
        type=int,
        default=16,
        help="每个 scene_type 参与摘要的最多历史环境条数（控制 token）",
    )
    ap.add_argument(
        "--diversity-digest-out",
        default="",
        help="可选：将各 scene_type 的结构化摘要写入该路径（JSON）",
    )
    ap.add_argument(
        "--incremental-diversity",
        action="store_true",
        help=(
            "按 scene_type 串行生成，每条生成后提取摘要注入下一条 prompt，"
            "约束批内不重复。不同 scene_type 之间仍并发。"
        ),
    )
    args = ap.parse_args()

    load_mod = _load_handwritten_generator()
    return asyncio.run(main_async(args, load_mod))


if __name__ == "__main__":
    sys.exit(main())
