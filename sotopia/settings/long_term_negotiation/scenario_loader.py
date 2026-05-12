"""从 benchmark 存储层加载「长期谈判」场景（``EnvironmentProfile`` + ``game_metadata``）。

脚本 ``scripts/generate_long_term_negotiation_scenarios.py`` 或 ``scripts/generate_long_term_negotiation_llm.py`` 写入::

    EnvironmentProfile.game_metadata = {
        "pipeline": "long_term_negotiation",
        "quartet": bool,
        "num_participants": int | 缺失,  # 2–4；缺失时按 quartet 推断（False→2，True→4）
        "lineup": "with_institutional" | "firms_only",  # 缺失则 with_institutional
        "strict_design_v1": bool,
        "timeline": NegotiationTimelineParams 的 dict（``dataclasses.asdict`` 形态）
        "codename": str,
        ...
    }

``lineup`` 决定按哪一种顺序取 N 名 canonical 角色：

- ``with_institutional``：``SESSION_SPEAKER_ROLE_ORDER`` 前缀
  （N=2 → firm_a/firm_b；N=3 → +investor；N=4 → +regulator）。
- ``firms_only``：``SESSION_FIRMS_ONLY_ROLE_ORDER`` 前缀
  （N=2 → firm_a/firm_b；N=3 → +firm_c；N=4 → +firm_d）—— 用于 **3+ 家公司**互谈。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from redis_om.model.model import NotFoundError

from .types import (
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    NegotiationTimelineParams,
    SUPPORTED_NEGOTIATION_LINEUPS,
    negotiation_role_order,
)


def _bounded(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _rule_seed(*parts: str) -> int:
    key = "|".join(parts)
    # Deterministic seed: keep the same rule per codename.
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _infer_news_sentiment_signal(scenario_text: str) -> float:
    txt = (scenario_text or "").lower()
    pos_kw = ("growth", "synergy", "expansion", "premium", "strong demand", "scale")
    neg_kw = ("distressed", "debt", "default", "layoff", "antitrust", "litigation", "loss")
    score = 0.0
    score += sum(1.0 for kw in pos_kw if kw in txt)
    score -= sum(1.0 for kw in neg_kw if kw in txt)
    return _bounded(score / 4.0, -1.0, 1.0)


def _build_predefined_outcome_rule(
    *,
    codename: str,
    lineup: str,
    num_participants: int,
    scenario_text: str = "",
) -> dict[str, Any]:
    roles = tuple(negotiation_role_order(lineup)[:num_participants])
    company_roles = tuple(r for r in roles if r.startswith("firm_"))
    seed = _rule_seed(codename, lineup, str(num_participants), scenario_text[:512])
    rng = random.Random(seed)

    # Contract economics: news gives directional hint but not full determinism.
    base_sentiment = _infer_news_sentiment_signal(scenario_text)
    jitter = rng.uniform(-0.35, 0.35)
    news_signal = _bounded(base_sentiment * 0.7 + jitter * 0.3, -1.0, 1.0)
    base_margin = rng.uniform(0.03, 0.12)
    profit_margin_bounds = (-0.25, 0.35)
    contract_value = float(rng.randint(120, 420)) * 1_000_000.0

    # Profit shares are generated on company roles only.
    raw = [max(0.05, rng.random()) for _ in company_roles]
    den = sum(raw) if raw else 1.0
    company_profit_share = {
        role: float(v / den) for role, v in zip(company_roles, raw, strict=False)
    }
    # Personal payoff: each role's individual score takes a slice of its company benefit.
    individual_income_share = {role: float(rng.uniform(0.3, 0.75)) for role in roles}

    return {
        "version": "v1",
        "contract_name": f"predefined_{codename}_main_contract",
        "deterministic_seed": seed,
        "contract_value_if_signed": contract_value,
        "profit_margin_bounds": [profit_margin_bounds[0], profit_margin_bounds[1]],
        "margin_formula": {
            "base_margin": base_margin,
            "news_weight": 0.55,
            "execution_weight": 0.45,
        },
        "news_signal": news_signal,
        "company_profit_share": company_profit_share,
        "individual_income_share": individual_income_share,
        "notes": (
            "Predefined scoring rule generated at data-construction time. "
            "Used by evaluation to compute company and individual payoffs."
        ),
    }


def _build_news_briefs_from_rule(
    *,
    codename: str,
    rule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seed = int(rule.get("deterministic_seed") or _rule_seed(codename, "news"))
    rng = random.Random(seed ^ 0xA5A5_5A5A)
    signal = float(rule.get("news_signal", 0.0) or 0.0)
    label = "cautiously positive" if signal >= 0.15 else ("negative" if signal <= -0.15 else "mixed")
    drift = _bounded(signal + rng.uniform(-0.4, 0.4), -1.0, 1.0)
    return [
        {
            "title": f"{codename}: sector outlook turns {label}",
            "summary": (
                "Analysts update medium-term demand and funding conditions; "
                "companies cite uncertainty on integration and cash-flow timing."
            ),
            "signal_hint": round(signal, 4),
            "correlation_level": "medium",
        },
        {
            "title": f"{codename}: rumor-driven volatility ahead of contract talks",
            "summary": (
                "Short-term price action diverges from fundamentals. "
                "This bulletin is intentionally only partially aligned with contract economics."
            ),
            "signal_hint": round(drift, 4),
            "correlation_level": "partial",
        },
    ]


def build_negotiation_game_metadata_bundle(
    codename: str,
    quartet: bool,
    params: NegotiationTimelineParams,
    *,
    num_participants: int | None = None,
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    design_doc: str = "social_env/design_1.md",
    scenario_text: str = "",
) -> dict[str, Any]:
    """构造与手写生成脚本一致的 ``game_metadata`` 谈判块（可合并进 LLM 生成的 profile）。

    ``lineup`` 默认 ``with_institutional``（与历史 bilat / tri / quartet 完全等价，
    ``num_participants∈{2,3,4}`` 取 ``firm_a/firm_b/(investor)/(regulator)`` 前缀）。

    ``lineup="firms_only"`` 时取 ``firm_a/firm_b/(firm_c)/(firm_d)`` 前缀，3+ 家公司不再
    包含 investor / regulator；strict_design_v1 在 firms_only 模式始终为 ``False``。
    """
    timeline_meta = asdict(params)
    timeline_meta["external_event_specs"] = list(timeline_meta.get("external_event_specs") or ())
    if lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"unknown negotiation lineup {lineup!r}; expected one of "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}"
        )
    n = num_participants if num_participants is not None else (4 if quartet else 2)
    if n < 2 or n > 4:
        raise ValueError(f"num_participants must be 2..4, got {n}")
    strict = (
        lineup == NEGOTIATION_LINEUP_WITH_INSTITUTIONAL and quartet and n == 4
    )
    predefined_rule = _build_predefined_outcome_rule(
        codename=codename,
        lineup=lineup,
        num_participants=n,
        scenario_text=scenario_text,
    )
    return {
        "pipeline": "long_term_negotiation",
        "strict_design_v1": strict,
        "quartet": quartet,
        "num_participants": n,
        "lineup": lineup,
        "timeline": timeline_meta,
        "design_doc": design_doc,
        "codename": codename,
        "predefined_outcome_rule": predefined_rule,
        "predefined_news_briefs": _build_news_briefs_from_rule(codename=codename, rule=predefined_rule),
    }


@dataclass(frozen=True)
class NegotiationStoredScenario:
    """从库里还原的一局谈判配置（不含 LiteLLM 模型名）。"""

    environment_profile_pk: str
    codename: str
    quartet: bool
    #: 实际交互的 canonical 角色数；按 ``lineup`` 取对应顺序的前缀。
    num_participants: int
    strict_design_v1: bool
    params: NegotiationTimelineParams
    #: 角色阵型："with_institutional"（含 investor/regulator）或 "firms_only"（3+ 家公司）。
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL

    @property
    def roles(self) -> tuple[str, ...]:
        """按 ``lineup`` 与 ``num_participants`` 还原 N 名 canonical 角色顺序。"""
        return tuple(negotiation_role_order(self.lineup)[: self.num_participants])


def negotiation_timeline_params_from_saved_dict(payload: Mapping[str, Any]) -> NegotiationTimelineParams:
    """把 ``NegotiationTimelineParams`` 的字典快照还原成实例（未知键静默忽略）。"""
    fm = dict(payload)
    if isinstance(fm.get("external_event_specs"), list):
        fm["external_event_specs"] = tuple(fm["external_event_specs"])

    names = {f.name for f in fields(NegotiationTimelineParams)}
    kw = {k: fm[k] for k in fm if k in names}
    return NegotiationTimelineParams(**kw)


def parsed_scenario_from_game_metadata(env_pk: str, *, gm: Mapping[str, Any]) -> NegotiationStoredScenario:
    if gm.get("pipeline") != "long_term_negotiation":
        raise ValueError(
            f"environment_profile {env_pk}: game_metadata.pipeline must be "
            f"'long_term_negotiation', got {gm.get('pipeline')!r}"
        )
    timeline = gm.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError(f"environment_profile {env_pk}: missing game_metadata.timeline (dict)")
    quartet = bool(gm.get("quartet", False))
    raw_n = gm.get("num_participants", None)
    if raw_n is None:
        num_participants = 4 if quartet else 2
    else:
        num_participants = int(raw_n)
        if num_participants < 2 or num_participants > 4:
            raise ValueError(
                f"environment_profile {env_pk}: game_metadata.num_participants must be 2..4, "
                f"got {num_participants!r}"
            )
    lineup = str(gm.get("lineup") or NEGOTIATION_LINEUP_WITH_INSTITUTIONAL)
    if lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"environment_profile {env_pk}: game_metadata.lineup must be in "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}, got {lineup!r}"
        )
    strict = bool(gm.get("strict_design_v1", quartet and num_participants == 4))
    if num_participants < 4 or lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        strict = False
    params = negotiation_timeline_params_from_saved_dict(timeline)
    codename = str(gm.get("codename") or "")

    return NegotiationStoredScenario(
        environment_profile_pk=str(env_pk),
        codename=codename,
        quartet=quartet,
        num_participants=num_participants,
        strict_design_v1=strict,
        params=params,
        lineup=lineup,
    )


def load_negotiation_scenario_from_environment_profile_pk(pk: str) -> NegotiationStoredScenario:
    """``EnvironmentProfile.get(pk)`` 并解析谈判配置。"""
    from sotopia.database import EnvironmentProfile

    env = EnvironmentProfile.get(pk)
    gm = env.game_metadata
    if gm is None or not isinstance(gm, dict):
        raise ValueError(f"environment_profile pk={pk!r} missing game_metadata mapping")
    parsed = parsed_scenario_from_game_metadata(pk, gm=gm)
    if not parsed.codename and getattr(env, "codename", None):
        return NegotiationStoredScenario(
            environment_profile_pk=parsed.environment_profile_pk,
            codename=str(env.codename or ""),
            quartet=parsed.quartet,
            num_participants=parsed.num_participants,
            strict_design_v1=parsed.strict_design_v1,
            params=parsed.params,
            lineup=parsed.lineup,
        )
    return parsed


def environment_pks_from_manifest(path: Path) -> list[str]:
    """读取谈判场景生成脚本写出的 manifest JSON（``long_term_negotiation_manifest.json``）。"""
    import json as _json

    data = _json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("environments") or []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and row.get("pk"):
            pk = str(row["pk"])
            if pk not in seen:
                seen.add(pk)
                out.append(pk)
    return out


def get_agent_profile_v2(pk: str) -> Any | None:
    """按主键从存储读取 ``AgentProfileV2``（本地：``~/.sotopia/data/AgentProfileV2/<pk>.json``）。

    需 ``SOTOPIA_STORAGE_BACKEND=local``（或已配置的 Redis 后端）。不存在时返回 ``None``。
    """
    from sotopia.benchmark_v2_data_models import AgentProfileV2

    if not (pk or "").strip():
        return None
    try:
        return AgentProfileV2.get(str(pk).strip())
    except NotFoundError:
        return None


def load_agent_profile_v2_by_role_from_manifest_env(
    manifest_path: Path | str,
    environment_profile_pk: str,
) -> dict[str, Any]:
    """从 manifest 中解析某条环境的 ``agent_profile_v2_pks_by_role``，并 ``AgentProfileV2.get`` 加载实体。

    优先使用 ``environments[]`` 里与 ``environment_profile_pk`` 匹配条目的
    ``agent_profile_v2_pks_by_role``（**per_environment** 造数脚本）。若缺失则回退 manifest
    顶层的 ``agent_profile_v2_pks_by_role``（旧版全批共用一套 V2 时）。

    返回 ``{role: AgentProfileV2}``；某 pk 缺失或 get 失败则跳过该 role。
    """
    import json as _json

    from sotopia.benchmark_v2_data_models import AgentProfileV2

    path = Path(manifest_path)
    data = _json.loads(path.read_text(encoding="utf-8"))
    want = str(environment_profile_pk).strip()
    role_to_pk: dict[str, str] = {}

    for row in data.get("environments") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("pk", "")).strip() != want:
            continue
        raw = row.get("agent_profile_v2_pks_by_role")
        if isinstance(raw, dict):
            role_to_pk = {str(k): str(v).strip() for k, v in raw.items() if str(v).strip()}
        break

    if not role_to_pk:
        top = data.get("agent_profile_v2_pks_by_role")
        if isinstance(top, dict):
            role_to_pk = {str(k): str(v).strip() for k, v in top.items() if str(v).strip()}

    out: dict[str, Any] = {}
    for role, pk in role_to_pk.items():
        try:
            ap = AgentProfileV2.get(pk)
        except NotFoundError:
            continue
        if ap is not None:
            out[role] = ap
    return out


def negotiation_placeholder_episode_pk(codename: str, legacy_environment_pk: str) -> str:
    """与 ``persist_scenario_v2`` 里 ``make_initial_state_snapshot`` 的 ``episode_pk`` 一致。

    造数脚本用该占位 episode 把 ``day=0`` 的 ``SystemStateSnapshot`` 落盘；真实评测 episode 另用
    其它 ``episode_pk``，二者不必相同。
    """
    return f"ltr_placeholder_ep_{str(codename).strip()}_{str(legacy_environment_pk).strip()}"


def get_system_state_snapshot(pk: str) -> Any | None:
    """按主键读取 ``SystemStateSnapshot``（本地：``~/.sotopia/data/SystemStateSnapshot/<pk>.json``）。"""
    from sotopia.benchmark_v2_data_models import SystemStateSnapshot

    if not (pk or "").strip():
        return None
    try:
        return SystemStateSnapshot.get(str(pk).strip())
    except NotFoundError:
        return None


def load_initial_system_state_snapshot_for_negotiation_legacy_env(
    *,
    codename: str,
    legacy_environment_pk: str,
) -> Any | None:
    """按占位 ``episode_pk`` 与 ``day==0`` 加载造数写入的**初始** ``SystemStateSnapshot``。

    若本地存在多条 ``day==0`` 同 episode（极少见），按 ``pk`` 字典序取第一条。
    """
    from sotopia.benchmark_v2_data_models import SystemStateSnapshot

    ep = negotiation_placeholder_episode_pk(codename, legacy_environment_pk)
    rows = list(SystemStateSnapshot.find(episode_pk=ep, day=0).all())
    if not rows:
        return None
    rows.sort(key=lambda x: str(getattr(x, "pk", "") or ""))
    return rows[0]


def load_initial_system_state_snapshot_from_manifest_env(
    manifest_path: Path | str,
    environment_profile_pk: str,
) -> Any | None:
    """在 manifest 的 ``environments[]`` 中按 ``pk`` 找到 ``codename``，再加载对应初始快照。"""
    import json as _json

    path = Path(manifest_path)
    data = _json.loads(path.read_text(encoding="utf-8"))
    want = str(environment_profile_pk).strip()
    for row in data.get("environments") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("pk", "")).strip() != want:
            continue
        codename = str(row.get("codename") or "").strip()
        if not codename:
            return None
        return load_initial_system_state_snapshot_for_negotiation_legacy_env(
            codename=codename,
            legacy_environment_pk=want,
        )
    return None


__all__ = [
    "NegotiationStoredScenario",
    "build_negotiation_game_metadata_bundle",
    "environment_pks_from_manifest",
    "get_agent_profile_v2",
    "get_system_state_snapshot",
    "load_agent_profile_v2_by_role_from_manifest_env",
    "load_initial_system_state_snapshot_for_negotiation_legacy_env",
    "load_initial_system_state_snapshot_from_manifest_env",
    "load_negotiation_scenario_from_environment_profile_pk",
    "negotiation_placeholder_episode_pk",
    "negotiation_timeline_params_from_saved_dict",
    "parsed_scenario_from_game_metadata",
]
