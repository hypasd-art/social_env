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

用大模型起草「scenario 段落 + 各角色战略目标句」（时间与 ``NegotiationTimelineParams``
仍按代码档位固定写入 ``game_metadata``，保证与设计/评测兼容）::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local python scripts/generate_long_term_negotiation_scenarios.py \\
      --llm --llm-model gpt-4o-mini --tag ltr_llm_seed_v1

依赖 ``social_env/.env`` 里的 ``OPENAI_API_KEY``（及可选 ``OPENAI_API_BASE``）。模型需支持
结构化 JSON（``litellm`` 的 ``json_schema`` / ``response_format``）。``--llm`` 分支使用轻量 ``litellm.acompletion``，
不导入整块 ``generation_utils.generate``（避免首轮导入卡住或极慢）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")
LOCAL_DATA_DIR = Path(os.path.expanduser("~/.sotopia/data"))


def _load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass


_load_repo_dotenv()

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

_TimelineLabel = Literal["D6", "D8", "D12"]


class BilateralLlmRow(BaseModel):
    """单条双边场景：与预设 ``D6/D8/D12`` 档位一一对应。"""

    timeline_label: Literal["D6", "D8", "D12"]
    scenario_body: str = Field(
        min_length=80,
        max_length=6000,
        description="Multi-sentence neutral scenario briefing for simulator agents.",
    )
    goal_firm_a: str = Field(
        min_length=20,
        max_length=2000,
        description="2–4 sentences: acquirer strategic objective (English). REQUIRED key goal_firm_a.",
    )
    goal_firm_b: str = Field(
        min_length=20,
        max_length=2000,
        description="2–4 sentences: target strategic objective (English). REQUIRED key goal_firm_b.",
    )


class QuartetLlmRow(BaseModel):
    timeline_label: Literal["D6", "D8", "D12"]
    scenario_body: str = Field(min_length=80, max_length=6000)
    goal_firm_a: str = Field(min_length=20, max_length=2000, description="Acquirer objectives; REQUIRED.")
    goal_firm_b: str = Field(min_length=20, max_length=2000, description="Target objectives; REQUIRED.")
    goal_investor: str = Field(min_length=20, max_length=2000, description="Investor objectives; REQUIRED.")
    goal_regulator: str = Field(min_length=20, max_length=2000, description="Regulator objectives; REQUIRED.")


class LlmNegotiationScenarioBundle(BaseModel):
    """一次 LLM 调用返回整包；条目数须覆盖所需 ``timeline_label``（无重复）。"""

    bilateral: list[BilateralLlmRow] = Field(default_factory=list)
    quartet: list[QuartetLlmRow] = Field(default_factory=list)


SCENARIO_LL_PROMPT_TEMPLATE = """
You produce **English** negotiation benchmark scenario text for a multi-day social simulator.

World rules (must respect in scenario_body text):
{rules}

Negotiation roster role cards (canonical ids — do NOT rename agents in goals; refer by role meaning only):
{role_cards}

Timeline slots you MUST cover exactly once each in ``bilateral`` (length 3) with labels D6, D8, D12:
longer horizons imply heavier scheduling load and more session rounds in-universe;
 mention calendar days, capacity per day, mix of natural language and structured JSON negotiation actions.

If ``quartet`` is required (non-empty list of length 3), same three labels; scenario_body should note investor + regulator
 may join sessions per design; four goal_* fields per row.

CRITICAL JSON shape — every bilateral array element MUST be an object with EXACTLY these keys (do not omit, do not rename, do not nest):
timeline_label, scenario_body, goal_firm_a, goal_firm_b.

Every quartet array element MUST include EXACTLY:
timeline_label, scenario_body, goal_firm_a, goal_firm_b, goal_investor, goal_regulator.

Put strategic instructions only in goal_* strings; scenario_body describes the situational briefing only.

Output JSON matching the provided schema. No markdown fences. Keep scenario_body self-contained (one paragraph or two short paragraphs per row).
"""


def _simulator_rules_block() -> str:
    return (
        "- Two-party (firm_a acquirer, firm_b target) baseline; quartet adds investor + regulator institutions.\n"
        "- Scheduling is constrained by calendar days D and slots per day; formal contract moves use structured actions.\n"
        "- Stakes: staged M&A-style commercial negotiation; lawful behavior; no real firm names.\n"
        "- Scenario is read by LLM agents: clear constraints beat literary flourish."
    )


def _strategy_goal_line(role: str, gist: str) -> str:
    hint = ROLE_SUMMARY_EN.get(role, role)
    return f"<strategy_hint>{hint}</strategy_hint> {gist.strip()}"


def _sanitized_schema_name(name: str) -> str:
    """与 ``generation_utils.generate._sanitize_schema_name`` 等价的最小拷贝（避免整套 generate 导入）。"""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def _scenario_bundle_response_format() -> dict[str, Any]:
    schema = LlmNegotiationScenarioBundle.model_json_schema()
    title = str(schema.get("title", LlmNegotiationScenarioBundle.__name__))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _sanitized_schema_name(title),
            "schema": schema,
            #: True 可减少模型漏字段（与 Chat Completions json_schema strict 对齐；若网关报错再改 False）。
            "strict": True,
        },
    }


def _scenario_bundle_from_raw_content(raw: str) -> LlmNegotiationScenarioBundle:
    """先标准 JSON parse，失败后尝试 ``json-repair``（仓库已有依赖）。"""
    s = str(raw).strip()
    last: BaseException | None = None
    try:
        return LlmNegotiationScenarioBundle.model_validate_json(s)
    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        last = e
    try:
        import json_repair as json_repair  # type: ignore[import-untyped]

        obj = json_repair.loads(s)
        return LlmNegotiationScenarioBundle.model_validate(obj)
    except Exception as e:
        last = e
    raise ValueError("cannot parse LLM scenario bundle as expected JSON/schema") from last


async def fetch_llm_scenario_bundle(
    *,
    model: str,
    temperature: float,
    need_bilateral: bool,
    need_quartet: bool,
) -> LlmNegotiationScenarioBundle:
    """仅用 ``litellm.acompletion`` + Pydantic schema，避免 ``import generate`` 拉满 gin/redis 等大型依赖栈。"""
    from litellm import acompletion
    from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
    from litellm.utils import supports_response_schema

    if not need_bilateral and not need_quartet:
        return LlmNegotiationScenarioBundle(bilateral=[], quartet=[])

    role_cards = json.dumps(ROLE_SUMMARY_EN, ensure_ascii=False, indent=2)
    rules = _simulator_rules_block()
    user_block = SCENARIO_LL_PROMPT_TEMPLATE.format(rules=rules, role_cards=role_cards).strip()

    bilateral_instr = (
        'Return **bilateral**: exactly THREE items with timeline_label covering "D6","D8","D12" once each.'
        if need_bilateral
        else "Return **bilateral**: an empty JSON array []."
    )
    quartet_instr = (
        'Return **quartet**: exactly THREE items with timeline_label covering "D6","D8","D12" once each.'
        if need_quartet
        else "Return **quartet**: an empty JSON array []."
    )

    tpl = """
{embedded}

{bilateral_instr}
{quartet_instr}

{format_instructions}
"""

    tpl_core = (
        tpl.replace("{embedded}", user_block)
        .replace("{bilateral_instr}", bilateral_instr)
        .replace("{quartet_instr}", quartet_instr)
        .replace("{format_instructions}", "")
        .strip()
        + "\n\nRespond with a single JSON object only (no markdown); it must match the response json_schema.\n"
    )

    model_name_effective = model
    base_url: str | None = None
    api_key: str | None = None
    if model_name_effective.startswith("custom"):
        first, rest = model_name_effective.split("@", 1)
        model_name_effective = first.replace("custom/", "openai/")
        base_url = rest
        api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")

    if base_url is None:
        supported = get_supported_openai_params(model=model_name_effective)
        assert supported is not None
        if "response_format" not in supported:
            raise RuntimeError(
                f"模型 {model_name_effective!r} 的 OpenAI 兼容参数中不支持 response_format，无法做结构化场景生成。",
            )
        if not supports_response_schema(model=model_name_effective):
            raise RuntimeError(
                f"模型 {model_name_effective!r} 不支持 response_schema / json_schema 模式。",
            )

    relax_schema_strict: list[bool] = [False]

    def _completion_kwargs(extra_user_prefix: str) -> dict[str, Any]:
        user_content = extra_user_prefix + tpl_core if extra_user_prefix else tpl_core
        rf = _scenario_bundle_response_format()
        if relax_schema_strict[0]:
            rf = json.loads(json.dumps(rf))
            rf["json_schema"]["strict"] = False
        kw: dict[str, Any] = {
            "model": model_name_effective,
            "messages": [{"role": "user", "content": user_content}],
            "response_format": rf,
            "drop_params": True,
            "base_url": base_url,
            "api_key": api_key,
        }
        if temperature is not None:
            kw["temperature"] = temperature
        return kw

    retries = 3
    last_err: BaseException | None = None

    bundle: LlmNegotiationScenarioBundle | None = None

    for attempt in range(retries):
        prefix = ""
        if attempt > 0:
            prefix = (
                "Your prior JSON omitted required keys (goal_firm_a, goal_firm_b, etc.). "
                "Every bilateral item MUST contain all keys: timeline_label, scenario_body, goal_firm_a, goal_firm_b. "
                "Every quartet item MUST also include goal_investor, goal_regulator. "
                "No shorter field names.\n\n"
            )
        try:
            response = await acompletion(**_completion_kwargs(prefix))
        except Exception as exc:
            # 某些自建网关不支持 strict json_schema：降级为 strict=false 再打一次同一请求。
            if not relax_schema_strict[0] and (
                "json_schema" in str(exc).lower()
                or "response_format" in str(exc).lower()
                or "structured" in str(exc).lower()
            ):
                relax_schema_strict[0] = True
                response = await acompletion(**_completion_kwargs(prefix))
            else:
                raise
        raw = response.choices[0].message.content
        if not raw or not str(raw).strip():
            last_err = ValueError("LLM returned empty content for scenario bundle")
            continue
        try:
            bundle = _scenario_bundle_from_raw_content(str(raw))
        except Exception as parse_exc:
            last_err = parse_exc
            continue

        expected = {"D6", "D8", "D12"}
        try:
            if need_bilateral:
                bil_labels = [r.timeline_label for r in bundle.bilateral]
                if set(bil_labels) != expected or len(bundle.bilateral) != 3:
                    raise ValueError(
                        f"LLM bilateral rows must cover D6/D8/D12 exactly once each; got {bil_labels}",
                    )
            elif bundle.bilateral:
                raise ValueError("unexpected non-empty bilateral from LLM when modes excluded bilat")

            if need_quartet:
                quad_labels = [r.timeline_label for r in bundle.quartet]
                if set(quad_labels) != expected or len(bundle.quartet) != 3:
                    raise ValueError(
                        f"LLM quartet rows must cover D6/D8/D12 exactly once each; got {quad_labels}",
                    )
            elif bundle.quartet:
                raise ValueError("unexpected non-empty quartet from LLM when modes excluded quartet")
        except ValueError as ve:
            last_err = ve
            continue

        return bundle

    assert last_err is not None
    raise RuntimeError(f"LLM scenario bundle invalid after {retries} attempts") from last_err


def _index_bilateral(rows: list[BilateralLlmRow]) -> dict[_TimelineLabel, BilateralLlmRow]:
    return {cast(_TimelineLabel, r.timeline_label): r for r in rows}


def _index_quartet(rows: list[QuartetLlmRow]) -> dict[_TimelineLabel, QuartetLlmRow]:
    return {cast(_TimelineLabel, r.timeline_label): r for r in rows}


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
    scenario_text_override: str | None = None,
    agent_goals_override: list[str] | None = None,
) -> EnvironmentProfile:
    timeline_meta = timeline_as_metadata(params)
    if scenario_text_override is not None and scenario_text_override.strip():
        body = scenario_text_override.strip()
    else:
        body = NEGOTIATION_SCENARIO_QUARTET if quartet else NEGOTIATION_SCENARIO_BODY

    if agent_goals_override is not None:
        goals = list(agent_goals_override)
    elif quartet:
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
    if scenario_text_override is not None and scenario_text_override.strip():
        gm["scenario_provenance"] = "llm_structured_bundle_v1"
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
    ap.add_argument(
        "--llm",
        action="store_true",
        help=(
            "使用大模型（结构化输出）起草各档位的 scenario 段落与战略目标句；时间表仍由脚本固定。"
            "需 OPENAI_API_KEY / 可用的 LiteLLM 模型（默认同 long_term_negotiation_llm_eval_demo）。"
        ),
    )
    ap.add_argument(
        "--llm-model",
        default="",
        help="LightLLM 模型名（空则依次为 Env NEGOTIATION_SCENARIO_GEN_MODEL、NEGOTIATION_AGENT_MODEL、gpt-4o-mini）",
    )
    ap.add_argument("--llm-temperature", type=float, default=0.65)
    args = ap.parse_args()

    print(f"[backend] SOTOPIA_STORAGE_BACKEND={os.environ['SOTOPIA_STORAGE_BACKEND']}")
    print(f"[paths]   {LOCAL_DATA_DIR}")

    wipe_local_data(yes=args.clean)

    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    if not modes.intersection({"bilat", "quartet"}):
        modes = {"bilat", "quartet"}

    need_bi = "bilat" in modes
    need_q = "quartet" in modes
    llm_ix_bil: dict[_TimelineLabel, BilateralLlmRow] = {}
    llm_ix_quad: dict[_TimelineLabel, QuartetLlmRow] = {}
    if args.llm:
        model_n = (
            args.llm_model.strip()
            or os.getenv("NEGOTIATION_SCENARIO_GEN_MODEL", "").strip()
            or os.getenv("NEGOTIATION_AGENT_MODEL", "").strip()
            or "gpt-4o-mini"
        )
        print(f"[llm] generating scenario stubs model={model_n} T={args.llm_temperature} bilat={need_bi} quartet={need_q}")
        bundle = asyncio.run(
            fetch_llm_scenario_bundle(
                model=model_n,
                temperature=args.llm_temperature,
                need_bilateral=need_bi,
                need_quartet=need_q,
            ),
        )
        if need_bi:
            llm_ix_bil = _index_bilateral(bundle.bilateral)
        if need_q:
            llm_ix_quad = _index_quartet(bundle.quartet)

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
            tl = cast(_TimelineLabel, label)
            br = llm_ix_bil.get(tl) if llm_ix_bil else None
            scen_ov = br.scenario_body if br else None
            goals_ov = (
                [_strategy_goal_line("firm_a", br.goal_firm_a), _strategy_goal_line("firm_b", br.goal_firm_b)]
                if br
                else None
            )
            legacy = build_environment_profile_legacy(
                codename=codename,
                quartet=False,
                params=params,
                tag=args.tag,
                scenario_text_override=scen_ov,
                agent_goals_override=goals_ov,
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
            tl = cast(_TimelineLabel, label)
            qr = llm_ix_quad.get(tl) if llm_ix_quad else None
            scen_ov = qr.scenario_body if qr else None
            goals_ov = (
                [
                    _strategy_goal_line("firm_a", qr.goal_firm_a),
                    _strategy_goal_line("firm_b", qr.goal_firm_b),
                    _strategy_goal_line("investor", qr.goal_investor),
                    _strategy_goal_line("regulator", qr.goal_regulator),
                ]
                if qr
                else None
            )
            legacy = build_environment_profile_legacy(
                codename=codename,
                quartet=True,
                params=params,
                tag=args.tag,
                scenario_text_override=scen_ov,
                agent_goals_override=goals_ov,
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
        "llm_scenario_authoring": bool(args.llm),
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
