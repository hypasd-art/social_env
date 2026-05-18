"""LLM 驱动的谈判 **人格画像**：每个角色对应**一个具体的人**（不是公司/机构本身）。

- 角色键 ``firm_a`` / ``firm_b`` / ``firm_c`` / ``firm_d`` 仅用于规则
  世界寻位；落库的 ``AgentProfile`` 必须是**自然人**：人名、年龄、个性、职业、价值观，不出现
  ``Firm`` / ``Corp`` / ``Inc`` / ``Ltd`` / ``Holdings`` / ``Authority`` 等公司化字眼。
- 人格应**多样化**：菜场摊主、话痨摊主、情绪化后和好、迷信吉利数、早起迷糊摊主等与「冷静公司谈判人」
  同等合法；由 ``_PERSONA_ARCHETYPES`` 轮转 + 提示词约束共同引导 LLM。
- 默认对**所有公司角色**（``firm_a`` / ``firm_b`` / ``firm_c`` / ``firm_d``）走 LLM 采样。

事实性资源、规则参数仍由 ``roles.py`` / ``default_agent_resources_bundle`` 提供。

依赖 ``social_env/.env`` 中的 ``OPENAI_API_KEY``（与其它 LLM 工具一致）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, ClassVar, Sequence

from pydantic import BaseModel, Field

from sotopia.database import AgentProfile
from sotopia.generation_utils.generate import agenerate
from sotopia.generation_utils.output_parsers import PydanticOutputParser

from .roles import CANONICAL_NEGOTIATION_ROSTER, FIRM_ROLES_ORDER, ROLE_SUMMARY_EN

# 默认对 **所有公司角色** 都用大模型（含 firm_c / firm_d）。
DEFAULT_COMPANY_LLM_ROLES: tuple[str, ...] = tuple(FIRM_ROLES_ORDER)


class LLMNegotiationAgentDraft(BaseModel):
    """谈判主谈人画像（**自然人**，非公司/机构本体）；``model_id`` / ``tag`` 由调用方注入。

    注：本模型多个字段带 ``Field(default=...)``，OpenAI ``json_schema`` 严格模式要求 root
    所有 property 必须在 ``required`` 里、且每个 object schema 必须 ``additionalProperties: false``，
    部分第三方网关同样如此。``OPENAI_DISABLE_STRICT_JSON_SCHEMA`` 让 ``_build_json_schema_response_format``
    把 ``strict=False`` 下传，避免 ``format_bad_output`` 修复路径触发 400。
    """

    OPENAI_DISABLE_STRICT_JSON_SCHEMA: ClassVar[bool] = True

    first_name: str = Field(
        description=(
            "A believable human given name for the negotiator (max 12 chars). "
            "Must NOT contain corporate words (Firm/Corp/Inc/Ltd/LLC/Holdings/Capital/Authority/Bureau/Agency). "
            "Must NOT echo the role key (e.g. 'Firm', 'Investor', 'Regulator')."
        )
    )
    last_name: str = Field(
        description=(
            "A believable human surname (max 16 chars). "
            "No company suffixes (Inc/Ltd/Corp/Co./LLC/Group/Holdings)."
        )
    )
    age: int = Field(
        default=42,
        description="Adult age in [25, 65]; this is the person's age, not a firm tenure.",
    )
    occupation: str = Field(
        description=(
            "Concrete job for THIS person in plain life terms, e.g. 'wet-market produce stall owner', "
            "'night-market skewer vendor', 'dorm floor bulk-buy treasurer', 'second-hand bike reseller', "
            "'household errand runner', 'community canteen buyer' — NOT required to sound like corporate counsel."
        )
    )
    gender: str = Field(default="unknown")
    gender_pronoun: str = Field(default="they/them")
    public_info: str = Field(
        description=(
            "One paragraph of personal background relevant to deal-side negotiation. "
            "Write about the person (career path, education, communication style); avoid press-release tone "
            "and avoid talking only about the institution they represent."
        )
    )
    big_five: str = Field(
        default="Openness: medium; Conscientiousness: high; Extraversion: medium; "
        "Agreeableness: medium; Neuroticism: medium",
        description="Short Big-Five descriptor for the person (5 traits, low/medium/high).",
    )
    moral_values: list[str] = Field(
        default_factory=lambda: ["fairness"],
        description="2–4 short moral-value tokens for the person (e.g. fairness, harm-avoidance).",
    )
    schwartz_personal_values: list[str] = Field(
        default_factory=lambda: ["achievement"],
        description="2–4 Schwartz portrait values for the person (e.g. achievement, security).",
    )
    personality_and_values: str = Field(
        description="Two sentences about THIS person's personality/values in a deal-making context."
    )
    decision_making_style: str = Field(
        description="One line about the person's decision style; must mention calendar/session protocol awareness."
    )
    secret: str = Field(
        default="",
        description="Short non-public personal preference or private BATNA hint of THIS person; may be empty.",
    )
    risk_preference: str = Field(
        default="neutral",
        description=(
            "Risk attitude of THIS person in deal-making: 'averse' (prefers downside protection, "
            "staged commitments), 'neutral' (weighs expected value), or 'seeking' (chases upside, "
            "tolerates volatility). MUST vary across roles — not all 'neutral'."
        ),
    )
    initial_reputation: float = Field(
        default=50.0,
        description=(
            "Initial reputation score 0-100 for THIS person in the market. High (70-90) for "
            "established incumbents with loyal customers; medium (40-60) for regular traders; "
            "low (15-35) for newcomers, challengers, or those with known defaults. "
            "MUST be differentiated across roles based on market position."
        ),
    )
    resource_modifiers: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Multipliers applied to the default resource bundle for this role's side. "
            "Keys match resource fields (cash, daily_fixed_cost, short_term_debt_due, asset, liability). "
            "Values around 1.0 mean no change; >1.0 increases the resource; <1.0 decreases it. "
            "Example: a cash-strapped challenger might have {'cash': 0.7, 'short_term_debt_due': 1.3}. "
            "An incumbent with deep pockets: {'cash': 1.4, 'daily_fixed_cost': 1.1}. "
            "Omit keys that stay at default. MUST differentiate at least 2 roles per scenario. "
            "If not specified, defaults to empty dict (no modification)."
        ),
    )


_PROMPT_TEMPLATE = """Generate a fictional JSON **agent profile** for a single **human negotiator** taking
part in a long-horizon multi-day negotiation simulator (market lanes, personal retail, group buys, or small stalls;
calendar slots, formal JSON moves).

CRITICAL: the profile describes one **specific person**, NOT a company, fund, regulator, or institution.
- **Diversity matters:** people may be wet-market hawkers, stall aunties/uncles, dorm treasurers, second-hand
  resellers, errand runners, or sharp hobbyists — **not** only calm corporate dealmakers. Impulsive, chatty,
  superstitious-about-numbers, or moody styles are welcome if believable and distinct.
- ``first_name`` / ``last_name`` must be believable human personal names. They MUST NOT contain:
  Firm, Corp, Co., Inc, Ltd, LLC, Group, Holdings, Capital, Authority, Bureau, Agency, Office,
  the role key, or any acronym for the institution.
- All other fields describe the person (career, personality, decision style, values), not the institution.
  The role key below only tells you which side this person plays in the episode.

Role key (context only): ``{role}``
Role-side social slot (do NOT treat as a corporation; do NOT name a company in any output field): ``{role_hint}``
Tag (experiment label): ``{tag}``
Assigned persona archetype (MUST emphasize): ``{diversity_brief}``

Constraints (MUST follow):
- ``first_name`` <= 12 chars; ``last_name`` <= 16 chars; ASCII-friendly.
- ``moral_values`` and ``schwartz_personal_values`` are short token lists (2–4 items each).
- ``decision_making_style`` must mention calendar/session protocol awareness.
- **Conversational differentiation:** In ``public_info`` and especially ``personality_and_values``, specify a **distinct**
  spoken style (default register, pacing, typical openers/fillers, what they avoid) that would sound **different**
  from another negotiator in the same market episode—without copying the archetype label verbatim as a name.
- **Benchmark alignment:** The downstream simulator injects ``[dialogue_voice]`` / DialogueVoice into private goals;
  keep ``decision_making_style`` and ``personality_and_values`` consistent with that voice so multi-day transcripts stay distinguishable.
- No sensitive personal data, no real public figures.
- **Economic differentiation — CRITICAL:** The three fields below MUST NOT all be default across roles.
  At least 2 roles in the same episode must have noticeably different ``risk_preference``, ``initial_reputation``,
  or ``resource_modifiers``.
  * ``risk_preference``: Match to the archetype — a decisive-competitor or opportunistic-bargainer is typically
    "seeking"; a risk-averse-stabilizer or principled-guardian is "averse"; others may be "neutral".
  * ``initial_reputation``: Incumbents with long lane history get 65-85; regular traders get 40-60;
    newcomers/challengers get 20-35.
  * ``resource_modifiers``: Financially pressured roles should have cash < 1.0 (e.g. 0.6-0.85) and
    short_term_debt_due > 1.0 (e.g. 1.15-1.4). Cash-rich roles get cash > 1.0 (e.g. 1.15-1.5).
    The modifiers MUST reflect the persona's market position and survival pressure.

Output ONLY a JSON OBJECT WITH FILLED-IN VALUES (do NOT echo a JSON schema, do NOT include
``description`` / ``type`` / ``properties`` keys, do NOT wrap in markdown). Use exactly these
keys (string / int / float / list[str] / dict as shown):

{{
  "first_name": "...",
  "last_name": "...",
  "age": 42,
  "occupation": "...",
  "gender": "...",
  "gender_pronoun": "...",
  "public_info": "...",
  "big_five": "Openness: ...; Conscientiousness: ...; Extraversion: ...; Agreeableness: ...; Neuroticism: ...",
  "moral_values": ["fairness", "loyalty"],
  "schwartz_personal_values": ["achievement", "security"],
  "personality_and_values": "...",
  "decision_making_style": "...",
  "secret": "",
  "risk_preference": "neutral",
  "initial_reputation": 50.0,
  "resource_modifiers": {{"cash": 1.0}}
}}

Reference field semantics (DO NOT include the schema itself in the output):
{format_instructions}
"""


_PERSONA_ARCHETYPES: tuple[dict[str, str], ...] = (
    {
        "label": "decisive-competitor",
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: high; Agreeableness: low; Neuroticism: medium",
        "values": "achievement, power, self-direction",
        "style": "pushes hard anchors, tolerates conflict, seeks first-mover advantage",
    },
    {
        "label": "risk-averse-stabilizer",
        "big_five": "Openness: low; Conscientiousness: high; Extraversion: low; Agreeableness: high; Neuroticism: medium",
        "values": "security, conformity, benevolence",
        "style": "prefers downside protection, staged commitments, and clear safeguards",
    },
    {
        "label": "analytical-strategist",
        "big_five": "Openness: high; Conscientiousness: high; Extraversion: low; Agreeableness: medium; Neuroticism: low",
        "values": "self-direction, achievement, universalism",
        "style": "optimizes with data and contingencies, avoids emotional framing",
    },
    {
        "label": "relational-mediator",
        "big_five": "Openness: medium; Conscientiousness: medium; Extraversion: high; Agreeableness: high; Neuroticism: low",
        "values": "benevolence, fairness, reciprocity",
        "style": "builds trust, reframes disputes, and trades concessions for relationship durability",
    },
    {
        "label": "opportunistic-bargainer",
        "big_five": "Openness: high; Conscientiousness: medium; Extraversion: medium; Agreeableness: low; Neuroticism: high",
        "values": "stimulation, achievement, hedonism",
        "style": "adapts quickly to leverage windows and exploits timing asymmetries",
    },
    {
        "label": "principled-guardian",
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: medium; Agreeableness: medium; Neuroticism: low",
        "values": "tradition, fairness, security",
        "style": "protects process legitimacy, emphasizes consistency and enforceable commitments",
    },
    {
        "label": "wet-market-hawker",
        "big_five": "Openness: medium; Conscientiousness: low; Extraversion: high; Agreeableness: medium; Neuroticism: high",
        "values": "stimulation, achievement, tradition",
        "style": "loud fair-price rhetoric, crowd timing, reputation over spreadsheets; may mis-remember yesterday's quote",
    },
    {
        "label": "chatty-auntie-vendor",
        "big_five": "Openness: high; Conscientiousness: medium; Extraversion: high; Agreeableness: high; Neuroticism: medium",
        "values": "benevolence, hedonism, conformity",
        "style": "gossip-as-signal, throws in extras instead of precise math; mood shifts prices",
    },
    {
        "label": "hot-then-cool-stallkeeper",
        "big_five": "Openness: low; Conscientiousness: low; Extraversion: high; Agreeableness: low; Neuroticism: high",
        "values": "power, stimulation, security",
        "style": "blunt anger then apology reset; impulse concessions after conflict",
    },
    {
        "label": "superstitious-round-number",
        "big_five": "Openness: medium; Conscientiousness: medium; Extraversion: medium; Agreeableness: medium; Neuroticism: medium",
        "values": "tradition, security, conformity",
        "style": "lucky digits, round anchors, omens from weather or foot traffic",
    },
    {
        "label": "sleepy-morning-seller",
        "big_five": "Openness: low; Conscientiousness: medium; Extraversion: low; Agreeableness: high; Neuroticism: medium",
        "values": "security, benevolence, conformity",
        "style": "vague until coffee; forgets verbal side deals; muscle memory over verbal precision",
    },
)

# 规则 / fallback 用的 named human personas（每个角色一名**具体的人**，避免落到公司化字面）。
DEFAULT_HUMAN_PERSONAS: dict[str, dict[str, Any]] = {
    "firm_a": {
        "first_name": "Riley",
        "last_name": "Carter",
        "age": 41,
        "occupation": "Neighborhood canteen buyer / household budget lead",
        "gender": "nonbinary",
        "gender_pronoun": "they/them",
        "public_info": (
            "Runs a tight morning shopping route; compares three stalls by weight, freshness, and who throws in "
            "scallions without being asked. Skeptical of slick talk, loyal when treated fair."
        ),
        "personality_and_values": (
            "Pragmatic and clock-aware; values straight numbers but will bend for a vendor who saved them last week."
        ),
        "decision_making_style": (
            "Calendar- and session-protocol aware; writes quantities on a phone note; switches to formal JSON moves "
            "when locking a bundle."
        ),
        "moral_values": ["fairness", "loyalty"],
        "schwartz_personal_values": ["achievement", "security"],
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: medium; "
        "Agreeableness: medium; Neuroticism: low",
        "secret": "",
    },
    "firm_b": {
        "first_name": "Jordan",
        "last_name": "Hayes",
        "age": 47,
        "occupation": "Wet-market produce stall owner",
        "gender": "female",
        "gender_pronoun": "she/her",
        "public_info": (
            "Third-generation lane regular; knows which hours the foot traffic peaks and which neighbor undercuts "
            "on leafy greens. Talks fast when nervous, slower when building trust."
        ),
        "personality_and_values": (
            "Warm with repeat faces, sharp with strangers; trades short margin for a customer who helps shout prices."
        ),
        "decision_making_style": (
            "Calendar/session-protocol aware; anchors with round numbers; uses formal moves after informal haggling."
        ),
        "moral_values": ["fairness", "stewardship"],
        "schwartz_personal_values": ["achievement", "tradition"],
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: high; "
        "Agreeableness: medium; Neuroticism: medium",
        "secret": "",
    },
    "firm_c": {
        "first_name": "Avery",
        "last_name": "Singh",
        "age": 38,
        "occupation": "Night-market challenger vendor",
        "gender": "female",
        "gender_pronoun": "she/her",
        "public_info": (
            "Newer stall with flexible sourcing; stacks A/B/C bundles loudly, sometimes overpromises delivery then "
            "negotiates extensions. Reads who is in a hurry versus browsing."
        ),
        "personality_and_values": (
            "Opportunistic but not cruel; respects a buyer who keeps their word on pickup time."
        ),
        "decision_making_style": (
            "Calendar/session-protocol aware; bursts of verbal offers then silence; locks terms with structured moves."
        ),
        "moral_values": ["fairness", "stewardship"],
        "schwartz_personal_values": ["achievement", "self-direction"],
        "big_five": "Openness: high; Conscientiousness: medium; Extraversion: high; "
        "Agreeableness: medium; Neuroticism: medium",
        "secret": "",
    },
    "firm_d": {
        "first_name": "Cameron",
        "last_name": "Doyle",
        "age": 45,
        "occupation": "Weekend flea / specialty stall operator",
        "gender": "male",
        "gender_pronoun": "he/him",
        "public_info": (
            "Late-shift seller with niche stock; calm voice until someone lowballs, then blunt. Prefers cash-on-hand "
            "and witnesses from the lane over long paperwork."
        ),
        "personality_and_values": (
            "Values face and repeat customers; distrusts abstract 'synergies' but will match a fair rival price."
        ),
        "decision_making_style": (
            "Calendar/session-protocol aware; keeps a cardboard sign with non-negotiables; uses formal moves to close."
        ),
        "moral_values": ["fairness", "loyalty"],
        "schwartz_personal_values": ["security", "achievement"],
        "big_five": "Openness: medium; Conscientiousness: high; Extraversion: low; "
        "Agreeableness: medium; Neuroticism: low",
        "secret": "",
    },
}


def _truncate_str(s: str, n: int) -> str:
    s = (s or "").strip()
    return s[:n]


_CORPORATE_NAME_TOKENS: tuple[str, ...] = (
    "firm",
    "corp",
    "co.",
    "co ",
    "inc",
    "ltd",
    "llc",
    "group",
    "holdings",
    "capital",
    "authority",
    "bureau",
    "agency",
    "office",
)


def _looks_corporate(name: str) -> bool:
    s = (name or "").strip().lower()
    if not s:
        return True
    return any(tok in s for tok in _CORPORATE_NAME_TOKENS)


def _archetype_briefs_for_roles(roles: Sequence[str], *, tag: str) -> dict[str, str]:
    role_list = list(roles)
    if not role_list:
        return {}
    h = hashlib.sha1(tag.encode("utf-8")).hexdigest()
    offset = int(h[:8], 16) % len(_PERSONA_ARCHETYPES)
    out: dict[str, str] = {}
    for i, role in enumerate(role_list):
        a = _PERSONA_ARCHETYPES[(offset + i) % len(_PERSONA_ARCHETYPES)]
        out[role] = (
            f"{a['label']}; target_big_five={a['big_five']}; "
            f"target_values={a['values']}; negotiation_style={a['style']}. "
            "Keep this profile clearly distinct from other roles in the same batch."
        )
    return out


_ECON_SECRET_KEY = "__v2_econ__"


def _encode_econ_secret(draft: LLMNegotiationAgentDraft) -> str:
    """将 LLM 生成的经济参数编码到 secret 字段，供 ``save_negotiation_agent_profiles_v2`` 解析。"""
    econ: dict[str, Any] = {
        "risk_preference": str(draft.risk_preference or "neutral"),
        "initial_reputation": float(draft.initial_reputation if draft.initial_reputation is not None else 50.0),
    }
    mods = dict(draft.resource_modifiers or {})
    if mods:
        econ["resource_modifiers"] = {str(k): float(v) for k, v in mods.items() if isinstance(v, (int, float))}
    personal_note = (draft.secret or "").strip()
    payload = {_ECON_SECRET_KEY: econ}
    if personal_note:
        payload["personal"] = personal_note[:240]
    return json.dumps(payload, ensure_ascii=False)


def parse_llm_econ_overrides(agent_profile_secret: str) -> dict[str, Any]:
    """从 AgentProfile.secret 解析 LLM 生成的经济参数。

    返回字典包含:
    - ``risk_preference``: str | None
    - ``initial_reputation``: float | None
    - ``resource_modifiers``: dict[str, float] | None
    解析失败时返回空字典。
    """
    secret = (agent_profile_secret or "").strip()
    if not secret:
        return {}
    try:
        payload = json.loads(secret)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    econ = payload.get(_ECON_SECRET_KEY)
    if not isinstance(econ, dict):
        return {}
    result: dict[str, Any] = {}
    rp = econ.get("risk_preference")
    if isinstance(rp, str) and rp in ("averse", "neutral", "seeking"):
        result["risk_preference"] = rp
    ir = econ.get("initial_reputation")
    if isinstance(ir, (int, float)):
        result["initial_reputation"] = float(max(0.0, min(100.0, ir)))
    rm = econ.get("resource_modifiers")
    if isinstance(rm, dict):
        clean: dict[str, float] = {}
        for k, v in rm.items():
            if isinstance(v, (int, float)):
                clean[str(k)] = float(v)
        if clean:
            result["resource_modifiers"] = clean
    return result


def build_static_negotiation_agent_profile(role: str, *, tag: str) -> AgentProfile:
    """从 ``DEFAULT_HUMAN_PERSONAS`` 装出一个**自然人** ``AgentProfile``。"""
    if role not in DEFAULT_HUMAN_PERSONAS:
        raise ValueError(
            f"role {role!r} has no default human persona; expected one of "
            f"{sorted(DEFAULT_HUMAN_PERSONAS)}"
        )
    persona = DEFAULT_HUMAN_PERSONAS[role]
    public_info = (persona["public_info"] or ROLE_SUMMARY_EN.get(role, "")).strip()
    secret = json.dumps(
        {
            _ECON_SECRET_KEY: {
                "risk_preference": "neutral",
                "initial_reputation": 50.0,
            },
            "personal": str(persona.get("secret", ""))[:240],
        },
        ensure_ascii=False,
    )
    return AgentProfile(
        first_name=str(persona["first_name"])[:12],
        last_name=str(persona["last_name"])[:16],
        age=int(persona["age"]),
        occupation=str(persona["occupation"])[:80],
        gender=str(persona["gender"])[:16],
        gender_pronoun=str(persona["gender_pronoun"])[:32],
        public_info=public_info,
        personality_and_values=str(persona["personality_and_values"])[:600],
        decision_making_style=str(persona["decision_making_style"])[:240],
        moral_values=list(persona["moral_values"]),
        schwartz_personal_values=list(persona["schwartz_personal_values"]),
        big_five=str(persona["big_five"])[:240],
        secret=secret,
        model_id=f"negotiation-{role}-{tag}",
        tag=tag,
    )


def _draft_to_agent_profile(
    role: str,
    draft: LLMNegotiationAgentDraft,
    *,
    model_name: str,
    tag: str,
) -> AgentProfile:
    """LLM 草稿 -> ``AgentProfile``。若名字落到公司化字面，回退到该角色的预置人名。"""
    persona = DEFAULT_HUMAN_PERSONAS.get(role, {})
    fallback_first = str(persona.get("first_name", "Alex"))[:12]
    fallback_last = str(persona.get("last_name", "Stone"))[:16]
    fallback_occ = str(persona.get("occupation", "deal-side professional"))[:80]
    fallback_pub = str(persona.get("public_info") or ROLE_SUMMARY_EN.get(role, ""))

    raw_first = _truncate_str(draft.first_name, 12)
    raw_last = _truncate_str(draft.last_name, 16)
    raw_occ = _truncate_str(draft.occupation, 80)

    first_name = raw_first if raw_first and not _looks_corporate(raw_first) else fallback_first
    last_name = raw_last if raw_last and not _looks_corporate(raw_last) else fallback_last
    occupation = raw_occ if raw_occ and not _looks_corporate(raw_occ) else fallback_occ

    return AgentProfile(
        first_name=first_name,
        last_name=last_name,
        age=int(max(25, min(65, draft.age or persona.get("age", 42)))),
        occupation=occupation,
        gender=_truncate_str(draft.gender, 16) or str(persona.get("gender", "unknown")),
        gender_pronoun=_truncate_str(draft.gender_pronoun, 32)
        or str(persona.get("gender_pronoun", "they/them")),
        public_info=_truncate_str(draft.public_info, 1200) or fallback_pub,
        personality_and_values=_truncate_str(draft.personality_and_values, 600)
        or str(persona.get("personality_and_values", "")),
        decision_making_style=_truncate_str(draft.decision_making_style, 240)
        or str(persona.get("decision_making_style", "")),
        moral_values=list(draft.moral_values or persona.get("moral_values", ["fairness"])),
        schwartz_personal_values=list(
            draft.schwartz_personal_values or persona.get("schwartz_personal_values", ["achievement"])
        ),
        big_five=_truncate_str(draft.big_five, 240)
        or str(
            persona.get(
                "big_five",
                "Openness: medium; Conscientiousness: high; Extraversion: medium; "
                "Agreeableness: medium; Neuroticism: medium",
            )
        ),
        secret=_encode_econ_secret(draft),
        model_id=f"negotiation-{role}-{tag}",
        tag=tag,
    )


async def _agenerate_one_draft(
    role: str,
    *,
    model_name: str,
    tag: str,
    diversity_brief: str,
) -> LLMNegotiationAgentDraft:
    parser = PydanticOutputParser[LLMNegotiationAgentDraft](
        pydantic_object=LLMNegotiationAgentDraft
    )
    return await agenerate(
        model_name=model_name,
        template=_PROMPT_TEMPLATE,
        input_values=dict(
            role=role,
            role_hint=ROLE_SUMMARY_EN.get(role, role),
            tag=tag,
            diversity_brief=diversity_brief,
        ),
        output_parser=parser,
        structured_output=False,
    )


async def agenerate_negotiation_agent_profiles(
    roles: Sequence[str],
    *,
    model_name: str,
    tag: str,
    concurrency: int = 4,
    save_to_storage: bool = True,
    llm_roles: Sequence[str] | None = None,
) -> dict[str, AgentProfile]:
    """对 ``roles`` 装配 ``AgentProfile``：所有公司角色走 LLM。

    ``llm_roles`` 显式传入 ``tuple(sorted(CANONICAL_NEGOTIATION_ROSTER))`` 等价于六角色均 LLM。
    """
    role_list = list(dict.fromkeys(roles))
    unknown = sorted(set(role_list) - CANONICAL_NEGOTIATION_ROSTER)
    if unknown:
        raise ValueError(
            f"unknown negotiation roles {unknown}; canonical roster is "
            f"{sorted(CANONICAL_NEGOTIATION_ROSTER)}"
        )

    llm_set = frozenset(llm_roles if llm_roles is not None else DEFAULT_COMPANY_LLM_ROLES)
    bad_llm = sorted(llm_set - CANONICAL_NEGOTIATION_ROSTER)
    if bad_llm:
        raise ValueError(f"llm_roles must be subset of canonical roster, got extra {bad_llm}")
    orphan_llm = sorted(llm_set - set(role_list))
    if orphan_llm:
        raise ValueError(f"llm_roles {orphan_llm} not contained in roles={role_list}")

    sem = asyncio.Semaphore(max(1, concurrency))
    diversity_briefs = _archetype_briefs_for_roles(role_list, tag=tag)

    async def one_llm(role: str) -> tuple[str, LLMNegotiationAgentDraft | BaseException]:
        async with sem:
            try:
                draft = await _agenerate_one_draft(
                    role,
                    model_name=model_name,
                    tag=tag,
                    diversity_brief=diversity_briefs.get(role, "balanced-generalist"),
                )
                return role, draft
            except BaseException as exc:  # noqa: BLE001 — fallback covers ValidationError / network / 4xx
                return role, exc

    llm_role_list = [r for r in role_list if r in llm_set]
    drafts_list = await asyncio.gather(*(one_llm(r) for r in llm_role_list))
    drafts_by_role: dict[str, LLMNegotiationAgentDraft] = {}
    fallback_roles: list[tuple[str, BaseException]] = []
    for role, payload in drafts_list:
        if isinstance(payload, BaseException):
            fallback_roles.append((role, payload))
        else:
            drafts_by_role[role] = payload

    out: dict[str, AgentProfile] = {}
    for role in role_list:
        if role in drafts_by_role:
            ap = _draft_to_agent_profile(role, drafts_by_role[role], model_name=model_name, tag=tag)
        else:
            # 角色被划入 llm_set 但生成失败：fallback 到静态人设；非 llm 角色也走这条路。
            ap = build_static_negotiation_agent_profile(role, tag=tag)
        out[role] = ap

    # 兜底：若 LLM 输出过于趋同，强制把人格描述拉回各自 archetype，保证同批次人格差异。
    signatures = {((out[r].big_five or "").strip().lower(), (out[r].personality_and_values or "").strip().lower()) for r in role_list if r in out}
    if len(signatures) <= 1 and len(role_list) > 1:
        for role in role_list:
            ap = out.get(role)
            if ap is None:
                continue
            brief = diversity_briefs.get(role, "")
            if brief:
                ap.personality_and_values = (
                    f"{ap.personality_and_values} Archetype cue: {brief[:260]}"
                )[:600]
                if "target_big_five=" in brief:
                    target = brief.split("target_big_five=", 1)[1].split("; target_values=", 1)[0].strip()
                    if target:
                        ap.big_five = target[:240]

    if save_to_storage:
        for role in role_list:
            if role in out:
                out[role].save()

    if fallback_roles:
        # 非致命：用 print 而非 logging.warning，避免在 CLI 静默路径里被吃掉。
        for role, exc in fallback_roles:
            short = str(exc)
            if len(short) > 220:
                short = short[:220] + "…"
            print(f"[agent_profile][warn] role={role!r} LLM draft failed -> static fallback: {short}")
    return out


def agent_profile_to_jsonable(
    ap: AgentProfile,
    *,
    role: str,
    profile_source: str,
) -> dict[str, Any]:
    """序列化 ``AgentProfile``；``profile_source`` 为 ``llm`` 或 ``static``。"""
    return {
        "role": role,
        "profile_source": profile_source,
        "pk": getattr(ap, "pk", "") or "",
        "model_id": getattr(ap, "model_id", "") or "",
        "tag": getattr(ap, "tag", "") or "",
        "first_name": ap.first_name,
        "last_name": ap.last_name,
        "age": ap.age,
        "occupation": ap.occupation,
        "gender": ap.gender,
        "gender_pronoun": ap.gender_pronoun,
        "public_info": ap.public_info,
        "personality_and_values": ap.personality_and_values,
        "decision_making_style": ap.decision_making_style,
        "moral_values": list(ap.moral_values or ()),
        "schwartz_personal_values": list(ap.schwartz_personal_values or ()),
        "big_five": ap.big_five,
        "secret": ap.secret,
    }


__all__ = [
    "DEFAULT_COMPANY_LLM_ROLES",
    "DEFAULT_HUMAN_PERSONAS",
    "LLMNegotiationAgentDraft",
    "LLMNegotiationRelationshipDraft",
    "agenerate_negotiation_agent_profiles",
    "agenerate_negotiation_relationship_profiles",
    "agent_profile_to_jsonable",
    "build_static_negotiation_agent_profile",
    "parse_llm_econ_overrides",
]


# ============================================================
# LLM 生成 pairwise 关系画像
# ============================================================

_RELATIONSHIP_PROMPT_TEMPLATE = """Generate a pairwise relationship profile between two individual negotiators
in a long-horizon multi-day market negotiation.

Each person is a real individual (stall owner, buyer, vendor, etc.) — NOT a corporation.
Below are their profiles from the same episode.

Person A (role={role_a}):
- Name: {name_a}
- Occupation: {occupation_a}
- Personality & values: {personality_a}
- Big Five: {big_five_a}
- Decision style: {decision_style_a}
- Public info: {public_info_a}

Person B (role={role_b}):
- Name: {name_b}
- Occupation: {occupation_b}
- Personality & values: {personality_b}
- Big Five: {big_five_b}
- Decision style: {decision_style_b}
- Public info: {public_info_b}

Scene type: {scene_type}

CRITICAL — the relationship MUST be differentiated and specific:
- **relation tag**: a short label capturing the competitive dynamic (e.g. "price-war-rivals",
  "grudging-coopetition", "quality-vs-price-standoff", "late-entrant-vs-incumbent",
  "mutual-suspicion", "customer-poaching-feud"). DO NOT use generic labels like "stranger" or "neutral".
- **trust_bias**: a float in [-0.6, 0.4] — negative for competitive/distrustful pairs,
  positive only if the profiles suggest genuine warm history. Most firm-firm pairs should be negative
  (competition). Different pairs in the same episode MUST have different trust_bias values.
- **history_note**: one sentence describing the pre-episode competitive dynamic between these two
  specific people, reflecting their occupations and personalities. NOT a generic template.
- **impression_a_of_b**: how A views B — based on A's personality and B's profile. Use personal names.
- **impression_b_of_a**: how B views A — based on B's personality and A's profile. Use personal names.
  Must be DIFFERENT from impression_a_of_b (asymmetric perspectives).

Output ONLY a JSON object:
{{
  "relation": "...",
  "trust_bias": -0.30,
  "history_note": "...",
  "impression_a_of_b": "...",
  "impression_b_of_a": "..."
}}"""


class LLMNegotiationRelationshipDraft(BaseModel):
    """LLM 生成的 pairwise 关系画像。"""

    OPENAI_DISABLE_STRICT_JSON_SCHEMA: ClassVar[bool] = True

    relation: str = Field(
        description="Short label for the competitive dynamic between these two people."
    )
    trust_bias: float = Field(
        default=-0.30,
        description="Float in [-0.6, 0.4] — negative for competitive/distrustful, positive for warm history."
    )
    history_note: str = Field(
        description="One sentence describing pre-episode competitive dynamic between these two specific people."
    )
    impression_a_of_b: str = Field(
        description="How person A views person B, based on A's personality and B's profile."
    )
    impression_b_of_a: str = Field(
        description="How person B views person A, based on B's personality and A's profile. Must differ from impression_a_of_b."
    )


def _relation_draft_to_background_story(
    draft: LLMNegotiationRelationshipDraft,
    *,
    name_a: str,
    name_b: str,
    role_a: str,
    role_b: str,
    tag: str,
) -> str:
    """将 LLM 草稿转为 RelationshipProfile.background_story 格式。"""
    return (
        f"{name_a} and {name_b} are {draft.relation.replace('-', ' ')}. "
        f"{draft.history_note} "
        f"{name_a} sees {name_b} as: {draft.impression_a_of_b} "
        f"{name_b} sees {name_a} as: {draft.impression_b_of_a}"
    )


async def agenerate_negotiation_relationship_profiles(
    agents: dict[str, Any],
    *,
    roles: tuple[str, ...],
    model_name: str,
    tag: str,
    scene_type: str = "",
    concurrency: int = 4,
) -> dict[tuple[str, str], LLMNegotiationRelationshipDraft]:
    """对 roles 中所有 pair 调 LLM 生成差异化 pairwise 关系。

    返回 ``{(role_a, role_b): LLMNegotiationRelationshipDraft}``。
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    def _profile_text(ap: Any) -> dict[str, str]:
        return {
            "name": f"{getattr(ap, 'first_name', '')} {getattr(ap, 'last_name', '')}".strip(),
            "occupation": str(getattr(ap, "occupation", "") or ""),
            "personality": str(getattr(ap, "personality_and_values", "") or ""),
            "big_five": str(getattr(ap, "big_five", "") or ""),
            "decision_style": str(getattr(ap, "decision_making_style", "") or ""),
            "public_info": str(getattr(ap, "public_info", "") or ""),
        }

    async def _one_pair(role_a: str, role_b: str) -> tuple[tuple[str, str], LLMNegotiationRelationshipDraft | BaseException]:
        async with sem:
            ap_a = agents.get(role_a)
            ap_b = agents.get(role_b)
            pa = _profile_text(ap_a) if ap_a is not None else {}
            pb = _profile_text(ap_b) if ap_b is not None else {}
            try:
                parser = PydanticOutputParser[LLMNegotiationRelationshipDraft](
                    pydantic_object=LLMNegotiationRelationshipDraft
                )
                draft = await agenerate(
                    model_name=model_name,
                    template=_RELATIONSHIP_PROMPT_TEMPLATE,
                    input_values=dict(
                        role_a=role_a,
                        role_b=role_b,
                        name_a=pa.get("name", role_a),
                        name_b=pb.get("name", role_b),
                        occupation_a=pa.get("occupation", ""),
                        occupation_b=pb.get("occupation", ""),
                        personality_a=pa.get("personality", ""),
                        personality_b=pb.get("personality", ""),
                        big_five_a=pa.get("big_five", ""),
                        big_five_b=pb.get("big_five", ""),
                        decision_style_a=pa.get("decision_style", ""),
                        decision_style_b=pb.get("decision_style", ""),
                        public_info_a=pa.get("public_info", ""),
                        public_info_b=pb.get("public_info", ""),
                        scene_type=scene_type or "general_negotiation",
                    ),
                    output_parser=parser,
                    structured_output=False,
                )
                return (role_a, role_b), draft
            except BaseException as exc:
                return (role_a, role_b), exc

    pairs = [(roles[i], roles[j]) for i in range(len(roles)) for j in range(i + 1, len(roles))]
    if not pairs:
        return {}

    tasks = [_one_pair(a, b) for a, b in pairs]
    results = await asyncio.gather(*tasks)

    out: dict[tuple[str, str], LLMNegotiationRelationshipDraft] = {}
    for key, payload in results:
        if isinstance(payload, BaseException):
            short = str(payload)
            if len(short) > 200:
                short = short[:200] + "..."
            print(f"[relationship][warn] pair={key} LLM draft failed: {short}")
        else:
            out[key] = payload
    return out
