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

支持的 ``--modes``（**仅** ``firms_only`` lineup；token 一律以 ``firms*`` 表示人数）：

- ``firms2`` / ``firms3`` / ``firms4`` —— 2 / 3 / 4 个商业参与者（``firm_a`` … ``firm_d`` 前缀），
  **无** investor / regulator 机构位。

示例::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local python scripts/generate_long_term_negotiation_scenarios.py --clean --tag ltr_benchmark_v1

    # 规模：只生成 D6/D8 时间轴，每种 (模式×预设) 重复 2 份
    python scripts/generate_long_term_negotiation_scenarios.py --tag ltr_scale_v1 \\
        --modes firms2,firms3,firms4 --timeline-labels D6,D8 --replicates 2

    # 仅 3 / 4 方互谈
    python scripts/generate_long_term_negotiation_scenarios.py --tag ltr_firms_only \\
        --modes firms3,firms4 --timeline-labels D6,D8 --replicates 1

    # 精确指定每种人数场景条数（不再用 --modes / --replicates）：
    # 8 条 firms3 + 12 条 firms4 + 6 条 firms2（在 D6,D8 preset 上轮转）
    python scripts/generate_long_term_negotiation_scenarios.py --tag ltr_mix \\
        --mode-counts firms3=8,firms4=12,firms2=6 --timeline-labels D6,D8

    # 要求说明（写入 manifest，便于实验记录）
    python scripts/generate_long_term_negotiation_scenarios.py --requirements "用于论文表2；仅规则评测" --tag ltr_paper
"""

from __future__ import annotations

import argparse
import hashlib
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
    NegotiationTimelineParams,
    SESSION_FIRMS_ONLY_ROLE_ORDER,
    SESSION_SPEAKER_ROLE_ORDER,
)
from sotopia.settings.long_term_negotiation.roles import (  # noqa: E402
    CANONICAL_NEGOTIATION_ROSTER,
    FIRM_ROLES_ORDER,
    ROLE_PERSONA_EN,
    ROLE_SUMMARY_EN,
    default_agent_resources_bundle,
)

# 全部 6 个 canonical 角色（含 firm_c / firm_d / investor / regulator）；按字母排序便于
# 写库 / manifest 时稳定（与 V1 quartet 时仅 4 个不冲突）。
# 历史兼容常量（不再用于默认数据构造的 active 角色选择）。
QUARTET_ROSTER_ORDER: tuple[str, ...] = tuple(sorted(CANONICAL_NEGOTIATION_ROSTER))
_PERSONA_OVERRIDES: dict[str, dict[str, Any]] = {}

_PERSONALITY_ARCHETYPES: tuple[dict[str, str], ...] = (
    {
        "label": "decisive-competitor",
        "style": "fast anchor-setting and assertive boundary defense",
        "risk": "high-upside selective risk",
        "collab": "cooperates only under clear advantage",
        "extra_skill": "tactical anchoring",
        "motivation": "prove market dominance under direct rivalry",
    },
    {
        "label": "risk-averse-stabilizer",
        "style": "stepwise commitments with downside guardrails",
        "risk": "capital-preserving cautious risk",
        "collab": "prefers predictable long-term partners",
        "extra_skill": "downside scenario planning",
        "motivation": "avoid volatility and preserve steady cashflow",
    },
    {
        "label": "analytical-strategist",
        "style": "data-first comparison and contingent planning",
        "risk": "model-based calibrated risk",
        "collab": "collaborates when metrics align",
        "extra_skill": "multi-offer optimization",
        "motivation": "win by superior decision quality",
    },
    {
        "label": "relational-mediator",
        "style": "trust-building and concession sequencing",
        "risk": "moderate risk with relationship safeguards",
        "collab": "actively builds repeat cooperation",
        "extra_skill": "conflict de-escalation",
        "motivation": "convert one-off deals into durable partnerships",
    },
    {
        "label": "opportunistic-bargainer",
        "style": "timing-window exploitation and rapid repricing",
        "risk": "event-driven opportunistic risk",
        "collab": "temporary alliances around short-lived gains",
        "extra_skill": "timing leverage capture",
        "motivation": "extract short-term edge before rivals react",
    },
    {
        "label": "principled-guardian",
        "style": "rule-consistent terms with strict process legitimacy",
        "risk": "compliance-first low tail-risk preference",
        "collab": "cooperates with transparent and consistent actors",
        "extra_skill": "procedural robustness checks",
        "motivation": "be trusted as the most reliable counterparty",
    },
    {
        "label": "wet-market-hawker",
        "style": "loud, emotional display of 'fair price'; reputation and repeat neighbors beat spreadsheets",
        "risk": "gut-feel and rumor-driven; may mis-remember yesterday's quote then stubbornly defend face",
        "collab": "loyal to regulars; cold to strangers until a small favor is exchanged",
        "extra_skill": "crowd-timing and stall-banter to reset leverage",
        "motivation": "protect today's cash drawer and lane reputation more than optimal game theory",
        "big_five": "Openness: medium; Conscientiousness: low; Extraversion: high; Agreeableness: medium; Neuroticism: high",
    },
    {
        "label": "chatty-auntie-vendor",
        "style": "small talk, gossip-as-signal, bundles extras ('I'll throw in scallions') instead of precise math",
        "risk": "under-prices when flattered; over-prices when slighted",
        "collab": "softens for kindness stories; hardens if ignored mid-sentence",
        "extra_skill": "reading who is in a hurry vs browsing",
        "motivation": "keep the stall feeling like family while still covering rent",
        "big_five": "Openness: high; Conscientiousness: medium; Extraversion: high; Agreeableness: high; Neuroticism: medium",
    },
    {
        "label": "hot-then-cool-stallkeeper",
        "style": "short temper, blunt insults to 'unfair' offers, then sudden apology tea-break reset",
        "risk": "impulse concessions after conflict; inconsistent patience by time of day",
        "collab": "fair-dealing once respect is shown; holds grudges lightly if paid in humor",
        "extra_skill": "dramatic pause and walk-away bluff (sometimes real)",
        "motivation": "avoid feeling cheated even if margin is thin",
        "big_five": "Openness: low; Conscientiousness: low; Extraversion: high; Agreeableness: low; Neuroticism: high",
    },
    {
        "label": "superstitious-round-number",
        "style": "lucky/unlucky digits, round-yuan anchors, avoids 'splitting the difference' on 'bad' numbers",
        "risk": "non-linear jumps in price tied to mood and omens, not marginal cost",
        "collab": "trusts handshakes and witnesses more than written lines",
        "extra_skill": "narrating last week's 'sign' from the weather or foot traffic",
        "motivation": "close with a number that 'sits right' socially, not only economically",
        "big_five": "Openness: medium; Conscientiousness: medium; Extraversion: medium; Agreeableness: medium; Neuroticism: medium",
    },
    {
        "label": "sleepy-morning-seller",
        "style": "low energy early; vague until coffee; sharpens after lunch rush",
        "risk": "forgets a prior verbal promise then negotiates as if fresh",
        "collab": "easy rapport once awake; needs reminders written on cardboard",
        "extra_skill": "muscle-memory packing speed over verbal precision",
        "motivation": "survive the shift without drama; prefers repeat small wins",
        "big_five": "Openness: low; Conscientiousness: medium; Extraversion: low; Agreeableness: high; Neuroticism: medium",
    },
)

# 数据合成层：在角色基底 voice 上再叠一层可复现的「口头习惯」，保证同场多人语言风格可区分。
_SPEECH_LAYER_MODS: tuple[str, ...] = (
    "Micro-habit: often starts turns with a one-breath recap of the other side's last point.",
    "Micro-habit: uses concrete units (kg, minutes, dollars) instead of vague intensifiers.",
    "Micro-habit: occasional self-interrupt when correcting own numbers mid-sentence.",
    "Micro-habit: prefers questions over assertions when probing leverage.",
    "Micro-habit: light regional market metaphor about lanes, queues, or weather—never as proof.",
    "Micro-habit: ends stressful turns with a single blunt constraint line ('Non-starter: …').",
    "Micro-habit: rare dry humor only after a concession, never during threats.",
    "Micro-habit: when disagreeing, labels the disagreement as 'timing' or 'scope' rather than personal.",
    "Micro-habit: counts change aloud or taps the scale when insisting a price is 'already fair'.",
    "Micro-habit: alternates half-loud stall-call with sudden quiet aside to the buyer only.",
    "Micro-habit: cites a neighbor stall's price without naming them ('the next lane did…').",
)

# 与 archetype 正交的「对话签名」：同一场景内多人应可听声辨人（合成时可复现）。
_CONVERSATION_SIGNATURES: tuple[dict[str, str], ...] = (
    {
        "label": "staccato-buyer",
        "register": "plain, time-boxed, low metaphor",
        "pacing": "short clauses; rarely more than two sentences before a check-question",
        "openers": "Often opens with 'Quick check—' or 'Before noon I need…'",
        "avoid": "Corporate slogans, mansplaining, fake enthusiasm",
    },
    {
        "label": "story-warm-vendor",
        "register": "neighbor-warm, anecdote-first",
        "pacing": "slow build, then a single crisp number; may repeat the number once when nervous",
        "openers": "Uses 'Regulars tell me…' / 'Same lane last season…'",
        "avoid": "Cold ultimatums unless inventory forces it; performative slang",
    },
    {
        "label": "rapid-fire-challenger",
        "register": "street-fast, competitive",
        "pacing": "bursts of A/B/C options; intentional silence after a sharp offer",
        "openers": "'Today-only window—' / 'Side-by-side: …'",
        "avoid": "Long policy monologues; apologizing for undercuts",
    },
    {
        "label": "minimalist-premium",
        "register": "polished-sparse, precision adjectives",
        "pacing": "long calm setup, then one closing constraint tied to cold-chain or capacity",
        "openers": "Frames scarcity without shouting; 'Capacity truth: …'",
        "avoid": "Rambling small talk; discount without value framing",
    },
    {
        "label": "procedural-neutral",
        "register": "third-person procedural, stall-rule vocabulary",
        "pacing": "even, de-escalating; terminal firmness only on hard lines",
        "openers": "Summarizes dispute in two sentences then cites applicable clause",
        "avoid": "Picking commercial winners; snark; gossip",
    },
    {
        "label": "skeptical-numbers",
        "register": "clipped, runway/tranche vocabulary, mild default skepticism",
        "pacing": "bullet-like conditions; pauses before 'non-negotiable'",
        "openers": "'Downside case first—' / 'Runway math: …'",
        "avoid": "Emotional pep talks; vague triggers",
    },
    {
        "label": "hedged-analyst",
        "register": "explicit uncertainty bands ('likely', 'if receipts match')",
        "pacing": "medium; labels assumptions and asks for one missing fact before committing",
        "openers": "'Two scenarios—' / 'What breaks this if…'",
        "avoid": "False certainty; talking over others' last point without acknowledgement",
    },
    {
        "label": "blunt-foreman",
        "register": "direct, slightly rough, logistics-first",
        "pacing": "one fact, one risk, one ask per turn when stressed",
        "openers": "'No drama—' / 'Crew window is…'",
        "avoid": "Passive voice pile-up; vague hand-wavy commitments",
    },
)


def _roles_by_lineup(lineup: str, n: int) -> tuple[str, ...]:
    if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        return tuple(SESSION_FIRMS_ONLY_ROLE_ORDER[:n])
    return tuple(SESSION_SPEAKER_ROLE_ORDER[:n])


def _persona_for_role(role: str) -> dict[str, Any]:
    override = _PERSONA_OVERRIDES.get(role)
    if override:
        return dict(override)
    raw = dict(ROLE_PERSONA_EN.get(role, {}))
    if not raw:
        return {
            "background_story": f"{role} participates as an independent market actor.",
            "personality": "balanced",
            "dialogue_voice": (
                "Register: neutral-pragmatic; pacing medium; vary slightly from peers so you are not interchangeable."
            ),
            "core_skills": ["negotiation"],
            "survival_pressure": "Maintain daily cashflow.",
            "daily_fixed_cost": 50.0,
            "short_term_debt_due": 0.0,
            "achievement_motivation": "Sustain long-term participation.",
        }
    return raw


def _stable_hash_int(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def _build_diversified_persona_overrides(*, roles: tuple[str, ...], tag: str) -> dict[str, dict[str, Any]]:
    """为本次数据合成构建稳定、可复现的人格多样化覆盖层。

    目标：
    - 同一批 active roles 尽量分配不同 archetype（避免同质化）。
    - 同一 tag 下可复现（便于实验重跑）。
    - 不破坏原有角色背景，只在人格/动机/技能/压力上做轻量扰动。
    """
    if not roles:
        return {}
    sorted_roles = tuple(sorted(roles))
    offset = _stable_hash_int(f"{tag}|persona_offset") % len(_PERSONALITY_ARCHETYPES)
    out: dict[str, dict[str, Any]] = {}
    for idx, role in enumerate(sorted_roles):
        base = dict(ROLE_PERSONA_EN.get(role, {}))
        if not base:
            base = _persona_for_role(role)
        archetype = _PERSONALITY_ARCHETYPES[(offset + idx) % len(_PERSONALITY_ARCHETYPES)]
        jitter_seed = _stable_hash_int(f"{tag}|{role}|jitter")
        # [-10, +10] 范围内的小扰动，保持资源量级稳定但增加个体差异。
        pct = ((jitter_seed % 21) - 10) / 100.0
        daily_base = float(base.get("daily_fixed_cost", 50.0) or 50.0)
        debt_base = float(base.get("short_term_debt_due", 0.0) or 0.0)
        core_skills = [str(x) for x in list(base.get("core_skills", []) or [])]
        if archetype["extra_skill"] not in core_skills:
            core_skills.append(archetype["extra_skill"])
        tic = _SPEECH_LAYER_MODS[
            (offset + idx + _stable_hash_int(f"{tag}|{role}|speech")) % len(_SPEECH_LAYER_MODS)
        ]
        sig = _CONVERSATION_SIGNATURES[
            (offset * 3 + idx * 5 + _stable_hash_int(f"{tag}|{role}|conv_sig"))
            % len(_CONVERSATION_SIGNATURES)
        ]
        sig_block = (
            f" [conversation_signature label={sig['label']}] "
            f"Register: {sig['register']}. Pacing: {sig['pacing']}. "
            f"Typical openers: {sig['openers']} Avoid in dialogue: {sig['avoid']}."
        )
        base_voice = str(base.get("dialogue_voice", "") or "").strip()
        composed_voice = f"{base_voice} {tic}{sig_block}".strip() if base_voice else f"{tic}{sig_block}".strip()
        out[role] = {
            **base,
            "personality": (
                f"{str(base.get('personality', 'balanced'))}; "
                f"archetype={archetype['label']}; risk_pref={archetype['risk']}; "
                f"collaboration={archetype['collab']}"
            ),
            "dialogue_voice": composed_voice,
            "core_skills": core_skills,
            "survival_pressure": (
                f"{str(base.get('survival_pressure', 'Maintain daily cashflow.'))} "
                f"Behavioral pressure: tends toward {archetype['style']}."
            ),
            "achievement_motivation": (
                f"{str(base.get('achievement_motivation', 'Sustain long-term participation.'))} "
                f"Additional drive: {archetype['motivation']}."
            ),
            "daily_fixed_cost": max(1.0, round(daily_base * (1.0 + pct), 2)),
            "short_term_debt_due": max(0.0, round(debt_base * (1.0 - pct), 2)),
            "persona_archetype": archetype["label"],
        }
        if archetype.get("big_five"):
            out[role]["big_five"] = str(archetype["big_five"])
    return out


def _social_graph_edges(roles: tuple[str, ...]) -> list[dict[str, Any]]:
    """社会图边：测试题库 **保证 firm 两两之间为商业竞争关系**（负向 trust_bias）。

    机构位与 firm 之间为融资/合规张力（非对称利润竞争）；investor↔regulator 为流程协调，
    不与 firm 间 rivalry 混用同一语义。
    """
    edges: list[dict[str, Any]] = []
    firm_competition_templates = (
        (
            "rivals_same_trade_lane",
            -0.48,
            "Compete for overlapping customers and walk-in substitution on the same trading day.",
        ),
        (
            "price_war_history",
            -0.38,
            "Recent undercut cycles and headline discounting create lingering distrust.",
        ),
        (
            "parallel_offer_pressure",
            -0.42,
            "Comparable outside quotes remain credible; scarcity keeps bids aggressive.",
        ),
    )
    n_roles = len(roles)

    def _firm_pair_edge(a: str, b: str, variant: int) -> dict[str, Any]:
        rel, trust_bias, note = firm_competition_templates[variant % len(firm_competition_templates)]
        pa = _persona_for_role(a)
        pb = _persona_for_role(b)
        a_skill = str((pb.get("core_skills") or ["negotiation"])[0])
        b_skill = str((pa.get("core_skills") or ["negotiation"])[0])
        a_view_b = (
            f"{a} treats {b} as a direct commercial competitor for the same customer pool; "
            f"{b} reads as {str(pb.get('personality', 'balanced'))} on {a_skill}."
        )
        b_view_a = (
            f"{b} treats {a} as a direct commercial competitor for the same customer pool; "
            f"{a} reads as {str(pa.get('personality', 'balanced'))} on {b_skill}."
        )
        a_view_b += " Expects reservation prices to move quickly under side-by-side comparison."
        b_view_a += " Expects reservation prices to move quickly under side-by-side comparison."
        return {
            "source": a,
            "target": b,
            "relation": rel,
            "trust_bias": float(trust_bias),
            "history_note": note,
            "source_impression_of_target": a_view_b,
            "target_impression_of_source": b_view_a,
            "competitive_axis": True,
        }

    for i, a in enumerate(roles):
        for j, b in enumerate(roles[i + 1 :], start=i + 1):
            a_f = a.startswith("firm_")
            b_f = b.startswith("firm_")
            if a_f and b_f:
                variant = (i * max(1, n_roles) + j) % len(firm_competition_templates)
                edges.append(_firm_pair_edge(a, b, variant))
                continue
            if not a_f and not b_f:
                pa = _persona_for_role(a)
                pb = _persona_for_role(b)
                rel = "institutional_co_gatekeeping"
                trust_bias = 0.18
                note = (
                    "Investor and regulator align on calendar/process while principals compete commercially; "
                    "this edge is coordination, not symmetric profit rivalry."
                )
                s_it = (
                    f"{a} treats {b} as shaping enforcement cadence that capital conditions reference; "
                    f"signal reads {str(pb.get('personality', 'balanced'))}."
                )
                t_is = (
                    f"{b} treats {a} as shaping risk appetite that bleeds into what principals dare promise; "
                    f"signal reads {str(pa.get('personality', 'balanced'))}."
                )
                edges.append(
                    {
                        "source": a,
                        "target": b,
                        "relation": rel,
                        "trust_bias": float(trust_bias),
                        "history_note": note,
                        "source_impression_of_target": s_it,
                        "target_impression_of_source": t_is,
                        "competitive_axis": False,
                    }
                )
                continue

            # 恰一方为 firm_*：与 investor / regulator 的张力边
            if a_f and not b_f:
                firm_r, inst_r = a, b
                f_prof, i_prof = _persona_for_role(firm_r), _persona_for_role(inst_r)
            else:
                firm_r, inst_r = b, a
                f_prof, i_prof = _persona_for_role(firm_r), _persona_for_role(inst_r)
            if inst_r == "investor":
                rel = "financing_leverage_review"
                trust_bias = -0.16
                note = (
                    "Investor ties drawdowns to how competitive pressure reshapes milestones; "
                    "the firm negotiates under contingent capital risk."
                )
                view_firm_of_inst = (
                    f"{firm_r} sees {inst_r} as tightening liquidity when parallel rivals destabilize pricing; "
                    f"expects {str(i_prof.get('personality', 'balanced'))}-driven covenant scrutiny."
                )
                view_inst_of_firm = (
                    f"{inst_r} reads {firm_r} as exposed to repricing in a rival-heavy lane; "
                    f"tracks {str((f_prof.get('core_skills') or ['execution'])[0])} execution risk."
                )
            elif inst_r == "regulator":
                rel = "compliance_visibility_pressure"
                trust_bias = -0.14
                note = (
                    "Regulator attention rises when multi-party rivalry creates incentives to cut corners "
                    "on disclosures or stall hygiene."
                )
                view_firm_of_inst = (
                    f"{firm_r} treats {inst_r} as binding visibility and calendar risk on competitive tactics; "
                    f"expects {str(i_prof.get('personality', 'balanced'))} enforcement tone."
                )
                view_inst_of_firm = (
                    f"{inst_r} maps {firm_r} as a principal whose rivalry-driven promotions may trigger audits "
                    f"or complaints from parallel vendors."
                )
            else:
                rel = "principal_auxiliary_interface"
                trust_bias = -0.1
                note = "Auxiliary roster tie: keep diligence without collapsing into firm-firm rivalry semantics."
                view_firm_of_inst = f"{firm_r} keeps a cautious dossier on {inst_r}."
                view_inst_of_firm = f"{inst_r} keeps a cautious dossier on {firm_r}."
            if a == firm_r:
                s_it, t_is = view_firm_of_inst, view_inst_of_firm
            else:
                s_it, t_is = view_inst_of_firm, view_firm_of_inst
            edges.append(
                {
                    "source": a,
                    "target": b,
                    "relation": rel,
                    "trust_bias": float(trust_bias),
                    "history_note": note,
                    "source_impression_of_target": s_it,
                    "target_impression_of_source": t_is,
                    "competitive_axis": False,
                }
            )
    return edges


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

    - ``firms2`` / ``firms3`` / ``firms4`` —— ``firms_only`` lineup（2 / 3 / 4 名商业参与者）。
    """
    allow = frozenset({"firms2", "firms3", "firms4"})
    out: list[str] = []
    seen: set[str] = set()
    for part in s.split(","):
        p = part.strip().lower()
        if p not in allow or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out or ["firms3", "firms4"]


def parse_mode_counts(s: str) -> dict[str, int] | None:
    """``--mode-counts MODE=N[,MODE=N...]`` 解析。

    例：``firms3=8,firms4=12`` -> ``{'firms3': 8, 'firms4': 12}``，每个 mode 各生成 N 条
    （在 ``--timeline-labels`` 选定的 preset 上轮转）；返回 ``None`` 表示走 ``--modes`` +
    ``--replicates`` 旧路径。
    """
    spec = (s or "").strip()
    if not spec:
        return None
    allow = frozenset({"firms2", "firms3", "firms4"})
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


# 模式 → (lineup, num_participants)；数据生成仅保留 ``firms_only``（``firms2``/``firms3``/``firms4``）。
_MODE_TO_LINEUP_N: dict[str, tuple[str, int]] = {
    "firms2": (NEGOTIATION_LINEUP_FIRMS_ONLY, 2),
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


# 写入 ``EnvironmentProfile.scenario`` / ``agent_goals`` 与 ``AgentProfile.public_info`` 的社会性叙事骨架
# （协议层 canonical id 仍为 ``firm_*``，见 ``roles.ROLE_SUMMARY_EN``）。
MARKET_COMPETITION_SNIPPET = (
    "Multiple independent sellers—or informal peers offering the same class of good—overlap in what they sell; "
    "buyers compare total value (price, condition, freshness, bundle perks, pickup or delivery slot, after-sales) "
    "and pick one workable plan. Scales range from **personal everyday purchases** (groceries, school lists, "
    "second-hand items) to small cooperative restocks. Reference rival offers in speech when persuading; "
    "use structured JSON moves for binding terms."
)

SCENE_CONTEXT_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "scene_type": "wet_market_competition",
        "headline": "菜市场高频交易",
        "detail": (
            "Fresh produce and daily essentials are traded in small but frequent batches; "
            "information is noisy and customer decisions are highly reactive."
        ),
        "perception_cues": ["foot_traffic", "competitor_freshness", "hawker_noise_level"],
    },
    {
        "scene_type": "personal_retail_compare",
        "headline": "日常个人采购比价",
        "detail": (
            "A single household or student compares two or more informal sellers for everyday goods—snacks, "
            "cleaning supplies, small electronics accessories—where ticket size is small but repeat visits matter."
        ),
        "perception_cues": ["shelf_substitution_risk", "walking_distance_minutes", "return_policy_clarity"],
    },
    {
        "scene_type": "second_hand_peer_sale",
        "headline": "二手闲置个人转让",
        "detail": (
            "Private individuals trade used bikes, furniture, or course books; trust is built through "
            "inspectable condition, honest defects, and agreed handoff time rather than formal procurement."
        ),
        "perception_cues": ["wear_visible", "peer_reference_call", "cash_on_pickup_preference"],
    },
    {
        "scene_type": "neighborhood_group_buy",
        "headline": "邻里拼单散购",
        "detail": (
            "Neighbors pool a modest list (rice, oil, fruit); one coordinator negotiates minimum order, split rules, "
            "and delivery drop-off with competing micro-suppliers."
        ),
        "perception_cues": ["min_order_gap", "split_fairness_per_unit", "elevator_carry_limit"],
    },
    {
        "scene_type": "school_supplies_season",
        "headline": "开学季个人采买",
        "detail": (
            "A caregiver shops small stationery and uniform bits across rival booths; kid preferences and total "
            "out-of-pocket dominate over long-term supplier contracts."
        ),
        "perception_cues": ["size_color_stock", "peer_parent_price_whisper", "last_minute_rush_pressure"],
    },
    {
        "scene_type": "errand_runner_basket",
        "headline": "代买跑腿同一购物单",
        "detail": (
            "Two informal errand runners bid to fulfill the same grocery list with substitution policies and "
            "time windows—closer to personal concierge rivalry than B2B procurement."
        ),
        "perception_cues": ["substitution_policy_strictness", "delivery_window_overlap", "receipt_photo_trust"],
    },
    {
        "scene_type": "business_outsourcing",
        "headline": "商业外包技能竞争",
        "detail": (
            "Independent service providers compete on skill combinations, labor availability, and delivery reliability "
            "for repeated task orders."
        ),
        "perception_cues": ["labor_supply_tightness", "skill_fit_score", "sla_reliability_gap"],
    },
    {
        "scene_type": "commercial_bidding",
        "headline": "商业合作竞价",
        "detail": (
            "Multiple operators submit and revise bids; each side has a different acceptable minimum price and "
            "must balance win-rate against margin sustainability."
        ),
        "perception_cues": ["best_bid_gap", "reserve_price_pressure", "undercut_probability"],
    },
)


def _scene_context_for_codename(codename: str) -> dict[str, Any]:
    idx = abs(hash(codename)) % len(SCENE_CONTEXT_TEMPLATES)
    return dict(SCENE_CONTEXT_TEMPLATES[idx])


def _context_reasoning_suffix(ctx: dict[str, Any]) -> str:
    cues = ", ".join(str(x) for x in (ctx.get("perception_cues") or []))
    return (
        f"Context={ctx.get('headline')}: {ctx.get('detail')} "
        f"Reason explicitly over observable cues: {cues}."
    )

NEGOTIATION_SCENARIO_BODY = (
    "Two **individual** participants — a choosy buyer and a primary counterparty (stall, errand-runner, or peer "
    "seller) — negotiate across several calendar days in settings that can include **wet-market lanes**, "
    "**neighborhood shops**, **second-hand meetups**, **dorm or block small bulk buys**, or **everyday personal "
    "retail** (groceries, school lists, small household items). "
    f"{MARKET_COMPETITION_SNIPPET} "
    "Scheduling is capacity-constrained per day; in-session exchanges mix natural dialogue with structured "
    "negotiation JSON actions. Treat counterparties as sole traders or households, not abstract corporations."
)

NEGOTIATION_SCENARIO_QUARTET = (
    NEGOTIATION_SCENARIO_BODY
    + " Two further **individual** roles may join selected sessions when formally invited: "
      "an informal financier (investor) and a market-hall compliance / stall-rules coordinator (regulator). "
      "They still compete for credibility and time with noisy rival stalls outside the session."
)

NEGOTIATION_SCENARIO_TRILATERAL = (
    NEGOTIATION_SCENARIO_BODY
    + " A third **individual** — an informal financier (investor) — may join selected sessions when contingent "
      "funding is formally requested; rivalry among vendors continues to pressure headline price."
)

NEGOTIATION_SCENARIO_FIRMS_ONLY_3 = (
    "Three **individual** operators in the same decision space: e.g. a careful buyer (firm_a), a familiar seller "
    "(firm_b), and a competing seller (firm_c) — stalls, errand-runners, or peer resellers offering overlapping "
    "goods or substitute baskets for **personal or small-group everyday purchase**. "
    f"{MARKET_COMPETITION_SNIPPET} "
    "They negotiate multi-day bundles that may include joint sourcing, split delivery routes, or price-matching "
    "against unseen rivals. Financing and regulatory institutional paths are off-table — only the three principals "
    "draft and amend contracts."
)

NEGOTIATION_SCENARIO_FIRMS_ONLY_4 = (
    "Four **individual** operators: lead buyer (firm_a), incumbent counterparty (firm_b), plus two competing "
    "sellers (firm_c, firm_d) — same foot-traffic, group-buy list, or second-hand lane; rivalry can be over "
    "groceries, dorm supplies, used gear, or small services, not only wholesale restock. "
    f"{MARKET_COMPETITION_SNIPPET} "
    "Sessions rotate formal proposals as each side tries to become the customer's best composite offer while "
    "keeping stock and cash constraints honest. No investor/regulator roles — pure peer rivalry plus buyer choice."
)


_HANDWRITTEN_ROLE_OCCUPATION: dict[str, str] = {
    "firm_a": "Household / canteen buyer (personal budget)",
    "firm_b": "Wet-market stall operator",
    "firm_c": "Challenger hawker / parallel stall",
    "firm_d": "Late-shift specialty stall",
    "investor": "Informal capital partner",
    "regulator": "Market-hall rules coordinator",
}

_DEFAULT_BIG_FIVE = (
    "Openness: medium; Conscientiousness: high; Extraversion: medium; "
    "Agreeableness: medium; Neuroticism: medium"
)


def save_negotiation_agents(*, tag: str, roles: tuple[str, ...]) -> dict[str, AgentProfile]:
    """按 ``roles`` 生成 ``AgentProfile``（当前默认仅 firms_only 角色，不含 investor/regulator）。"""
    profiles: dict[str, AgentProfile] = {}
    for role in roles:
        party, _, rest = role.partition("_")
        fn = party.title()
        ln = rest.upper() if rest else party.upper()
        persona = _persona_for_role(role)
        bg = str(persona.get("background_story", "") or "")
        voice = str(persona.get("dialogue_voice", "") or "").strip()
        skills = ", ".join(str(x) for x in (persona.get("core_skills") or []))
        pressure = str(persona.get("survival_pressure", "") or "")
        motivation = str(persona.get("achievement_motivation", "") or "")
        ap = AgentProfile(
            first_name=fn[:12],
            last_name=(ln + "Exec")[:20],
            age=42,
            occupation=_HANDWRITTEN_ROLE_OCCUPATION.get(role, "Individual market participant")[:80],
            gender="unknown",
            gender_pronoun="they/them",
            public_info=f"{ROLE_SUMMARY_EN.get(role, '')} Background: {bg}",
            personality_and_values=(
                f"Personality: {persona.get('personality', 'balanced')}. Motivation: {motivation}. "
                f"[dialogue_voice — use in speak turns; differ clearly from other roster members] {voice}"
            ),
            decision_making_style=(
                "May lean on habit, gossip, or gut as well as numbers; still uses protocol-compliant formal moves "
                "when locking commitments. "
                f"Core skills: {skills}. "
                "Natural-language turns: follow [dialogue_voice] in personality (register, pacing, openers, avoid)."
            ),
            moral_values=["fairness"],
            schwartz_personal_values=["achievement"],
            big_five=str(persona.get("big_five") or _DEFAULT_BIG_FIVE)[:240],
            secret=f"Survival pressure: {pressure}",
            model_id=f"negotiation-{role}-{tag}",
            tag=tag,
        )
        ap.save()
        profiles[role] = ap
    print(f"[save] AgentProfile (negotiation roster) x {len(profiles)}")
    return profiles


def pairwise_strangers(agents: dict[str, AgentProfile], *, tag: str, roles: tuple[str, ...]) -> None:
    edges = _social_graph_edges(tuple(roles))
    edge_map = {
        tuple(sorted((str(e["source"]), str(e["target"])))): e for e in edges
    }
    n = 0
    for i, a in enumerate(roles):
        for b in roles[i + 1 :]:
            e = edge_map.get(tuple(sorted((a, b))))
            note = (
                f"{e['relation']}: {e['history_note']} (trust_bias={e['trust_bias']:+.2f})"
                if e
                else "No prior tie recorded."
            )
            impressions = (
                f"Impressions — {a}-> {b}: {e['source_impression_of_target']} | "
                f"{b}-> {a}: {e['target_impression_of_source']}"
                if e
                else f"Impressions — {a} and {b} both start neutral with limited prior data."
            )
            r = RelationshipProfile(
                agent_1_id=agents[a].pk,
                agent_2_id=agents[b].pk,
                relationship=RelationshipType.stranger,
                background_story=f"Social graph seeded for '{tag}'. {a}<->{b}: {note}. {impressions}",
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
    lineup: str = NEGOTIATION_LINEUP_FIRMS_ONLY,
) -> EnvironmentProfile:
    """``lineup`` + ``num_participants`` 共同决定 roster 与场景文案。

    默认 ``lineup=firms_only``。``firms_only``：N=2/3/4 对应 ``firm_a``…``firm_d`` 前缀；
    若显式传入 ``with_institutional``，则仍支持旧 N=2/3/4 机构阵容文案（本仓库数据生成脚本不再使用）。
    """
    n = num_participants if num_participants is not None else (4 if quartet else 2)
    if n < 2 or n > 4:
        raise ValueError(f"num_participants must be 2..4, got {n}")
    gm_quartet = False
    ctx = _scene_context_for_codename(codename)
    ctx_suffix = _context_reasoning_suffix(ctx)

    if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        if n < 2:
            raise ValueError("firms_only lineup requires num_participants>=2")
        if n == 2:
            body = NEGOTIATION_SCENARIO_BODY
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> As a buyer, pick the best total offer among rival stalls while staying within cash limits. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Win the customer's choice with a credible bundle (price + quality + delivery). {ctx_suffix}",
            ]
        elif n == 3:
            body = NEGOTIATION_SCENARIO_FIRMS_ONLY_3
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Buyer-side lead: compare two parallel vendors and choose a bundle customers will actually accept. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Anchor vendor: protect margin while proving reliability under competitor pressure. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_c']}</strategy_hint> Parallel vendor: differentiate on quality/speed and outcompete incumbent terms. {ctx_suffix}",
            ]
        else:
            body = NEGOTIATION_SCENARIO_FIRMS_ONLY_4
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Lead buyer: run a competitive selection among multiple sellers and lock the best feasible package. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Incumbent seller: defend customer trust and avoid losing share to newer entrants. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_c']}</strategy_hint> Challenger seller: offer a compelling alternative without breaking cashflow. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_d']}</strategy_hint> Late entrant: use speed/clarity to win customer choice against established rivals. {ctx_suffix}",
            ]
    else:
        if n == 2:
            body = NEGOTIATION_SCENARIO_BODY
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Buyer side: secure supply from the best offer while keeping financing risk acceptable. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Seller side: convert walk-in comparison into a committed order with fair staged terms. {ctx_suffix}",
            ]
        elif n == 3:
            body = NEGOTIATION_SCENARIO_TRILATERAL
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Present a buyer plan strong enough to beat rival offers and still pass financing checks. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Keep terms attractive versus competitors while preserving downside protection. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['investor']}</strategy_hint> Provide contingent capital only if the negotiated bundle is robust under market competition. {ctx_suffix}",
            ]
        else:
            body = NEGOTIATION_SCENARIO_QUARTET
            goals = [
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_a']}</strategy_hint> Win supplier competition and secure both funding and compliance path for execution. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['firm_b']}</strategy_hint> Close a customer-winning package without conceding all margin to competitive pressure. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['investor']}</strategy_hint> Finance only when projected customer uptake and repayment path remain credible. {ctx_suffix}",
                f"<strategy_hint>{ROLE_SUMMARY_EN['regulator']}</strategy_hint> Enforce rules while keeping a fair competitive field for multiple operators. {ctx_suffix}",
            ]
            gm_quartet = True
    from sotopia.settings.long_term_negotiation.scenario_loader import (
        build_negotiation_game_metadata_bundle,
    )

    active_roles = _roles_by_lineup(lineup, n)
    gm_base = build_negotiation_game_metadata_bundle(
        codename,
        gm_quartet,
        params,
        num_participants=n,
        lineup=lineup,
        scenario_text=body,
    )
    gm_base["environment_scene"] = dict(ctx)
    gm_base["social_graph"] = {
        "nodes": [
            {
                "role": r,
                "summary": ROLE_SUMMARY_EN.get(r, r),
                "background_story": _persona_for_role(r).get("background_story", ""),
                "personality": _persona_for_role(r).get("personality", ""),
                "dialogue_voice": _persona_for_role(r).get("dialogue_voice", ""),
                "core_skills": list(_persona_for_role(r).get("core_skills", [])),
            }
            for r in active_roles
        ],
        "edges": _social_graph_edges(active_roles),
    }
    gm_base["agent_survival_constraints"] = {
        r: {
            "daily_fixed_cost": float(_persona_for_role(r).get("daily_fixed_cost", 0.0) or 0.0),
            "short_term_debt_due": float(_persona_for_role(r).get("short_term_debt_due", 0.0) or 0.0),
            "achievement_motivation": str(_persona_for_role(r).get("achievement_motivation", "") or ""),
        }
        for r in active_roles
    }
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


def negotiation_event_scripts(tag: str, *, max_calendar_days: int | None = None) -> list[Any]:
    """构造长期谈判场景共用的 ``EventScript`` 列表。

    约束：**每个自然日 1..max_calendar_days 至少有一条在日终（``END_OF_DAY``）触发的脚本**。
    已有剧情的日（如 day2 / day5）保留原效果；其余日补 ``broadcast`` 占位脚本（不改变 ``SystemState``）。
    ``max_calendar_days`` 默认取 ``bilateral_timeline_presets()`` 中最大的 ``D``，以覆盖全部预设时间轴。
    """
    from sotopia.events.event_engine import calendar_days_with_end_of_day_scripts

    if max_calendar_days is None:
        max_calendar_days = max(p.D for _, p in bilateral_timeline_presets())
    if max_calendar_days < 1:
        raise ValueError(f"max_calendar_days must be >= 1, got {max_calendar_days}")

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
                {"op": "delta", "target": "public_opinion.firm_b", "value": 0.25},
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
    ev3 = make_event_script_from_dict(
        {
            "name": "ltr_market_micro_dynamics_day3",
            "category": "market",
            "visibility": "public",
            "intraday": False,
            "apply_days": [3],
            "description": "Daily micro-dynamics: foot traffic, freshness pressure, and hawker noise shift.",
            "effects": [
                {"op": "delta", "target": "market_state.foot_traffic", "value": 0.06},
                {"op": "delta", "target": "market_state.competitor_quality_signal", "value": -0.04},
                {"op": "delta", "target": "market_state.hawker_noise_level", "value": 0.08},
            ],
            "tag": tag,
        }
    )
    out: list[Any] = [ev1, ev2, ev3]
    covered = calendar_days_with_end_of_day_scripts(out)
    for d in range(1, int(max_calendar_days) + 1):
        if d in covered:
            continue
        out.append(
            make_event_script_from_dict(
                {
                    "name": f"ltr_daily_eod_placeholder_day{d}",
                    "category": "market",
                    "visibility": "public",
                    "intraday": False,
                    "apply_days": [d],
                    "description": (
                        f"Placeholder end-of-day EventScript for calendar day {d} "
                        "(no state change; satisfies one-script-per-day coverage)."
                    ),
                    "effects": [
                        {
                            "op": "broadcast",
                            "target": "_daily_calendar_placeholder",
                            "value": None,
                        }
                    ],
                    "tag": tag,
                }
            )
        )
    return out


def save_negotiation_agent_profiles_v2(
    agents_by_role: dict[str, AgentProfile],
    *,
    tag: str,
    roles: tuple[str, ...],
) -> dict[str, Any]:
    """按 ``roles`` 生成 ``AgentProfileV2``（与当前场景 active roster 一致）。"""
    bundle = default_agent_resources_bundle()
    v2: dict[str, Any] = {}
    for role in roles:
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
    lineup: str = NEGOTIATION_LINEUP_FIRMS_ONLY,
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

    lgm = legacy_env.game_metadata if isinstance(legacy_env.game_metadata, dict) else {}
    env_ctx = dict(lgm.get("environment_context", {}) or {})
    env_scene = dict(lgm.get("environment_scene", {}) or {})
    physical_params = dict(env_ctx.get("physical_social_parameters", {}) or {})
    md_init: dict[str, Any] = {
        **timeline_meta,
        "negotiation_logical_resources_by_role": {k: dict(v) for k, v in bundle.items()},
        "market_state": {
            "interest_rate": 0.042,
            "regulatory_stringency": 1.0,
            "foot_traffic": float(physical_params.get("foot_traffic", 0.6) or 0.6),
            "competitor_quality_signal": float(physical_params.get("competitor_quality_signal", 0.6) or 0.6),
            "hawker_noise_level": float(physical_params.get("hawker_noise_level", 0.5) or 0.5),
            "labor_supply_tightness": float(physical_params.get("labor_supply_tightness", 0.5) or 0.5),
            "skill_complementarity_index": float(physical_params.get("skill_complementarity_index", 0.5) or 0.5),
            "bid_spread_index": float(physical_params.get("bid_spread_index", 0.5) or 0.5),
        },
        "environment_scene": env_scene,
        "environment_context": env_ctx,
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
        default="firms3,firms4",
        help=(
            "逗号分隔、去重保序：仅 firms2 / firms3 / firms4（均为 firms_only lineup，无机构位）"
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
            "按模式精确指定生成条数：MODE=N[,MODE=N...]；例 firms3=8,firms4=12,firms2=6。"
            "传入后忽略 --modes 与 --replicates，每个 mode 在 --timeline-labels 选定的 preset 上"
            "轮转生成 N 条。合法 MODE：firms2/firms3/firms4。"
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

    active_roles_set = {r for m in modes_sel for r in roles_for_mode(m)}
    active_roles_global = tuple(r for r in SESSION_FIRMS_ONLY_ROLE_ORDER if r in active_roles_set)
    _PERSONA_OVERRIDES.clear()
    _PERSONA_OVERRIDES.update(_build_diversified_persona_overrides(roles=active_roles_global, tag=args.tag))
    if _PERSONA_OVERRIDES:
        diversity_view = {
            r: str(_PERSONA_OVERRIDES.get(r, {}).get("persona_archetype", "unknown")) for r in active_roles_global
        }
        print(f"[persona_diversity] tag={args.tag} archetypes={diversity_view}")
    agents = save_negotiation_agents(tag=args.tag, roles=active_roles_global)
    pairwise_strangers(agents, tag=args.tag, roles=active_roles_global)
    v2_agents = save_negotiation_agent_profiles_v2(agents, tag=args.tag, roles=active_roles_global)

    events = negotiation_event_scripts(args.tag)
    for ev in events:
        ev.save()
    anchor_pk = events[0].pk if events else None
    print(f"[save] EventScript x {len(events)} anchor_pk={anchor_pk}")

    combos_by_codename: dict[str, EnvAgentComboStorage] = {}
    legacy_env_objs: list[EnvironmentProfile] = []
    env_modes_by_codename: dict[str, str] = {}
    env_lineup_by_codename: dict[str, str] = {}

    _MODE_PREFIX = {
        "firms2": "firms2",
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
                quartet=False,
                params=params,
                tag=args.tag,
                num_participants=n_agents,
                lineup=lineup,
            )
            legacy.save()
            combo = save_combo(legacy, roles, agents)
            combos_by_codename[codename] = combo
            legacy_env_objs.append(legacy)
            env_modes_by_codename[codename] = mode
            env_lineup_by_codename[codename] = lineup
            persist_scenario_v2(
                legacy,
                quartet=False,
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
        "agent_roles": active_roles_global,
        "agent_profile_pks_by_role": {r: agents[r].pk for r in active_roles_global},
        "agent_profile_v2_pks_by_role": {r: v2_agents[r].pk for r in active_roles_global},
        "environments": [
            {
                "codename": e.codename,
                "pk": e.pk,
                "mode": env_modes_by_codename.get(e.codename),
                "lineup": env_lineup_by_codename.get(e.codename),
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
            "persona_diversity_enabled": True,
            "persona_archetype_by_role": {
                r: str(_PERSONA_OVERRIDES.get(r, {}).get("persona_archetype", "unknown")) for r in active_roles_global
            },
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
