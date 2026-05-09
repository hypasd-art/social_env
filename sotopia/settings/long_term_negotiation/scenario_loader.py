"""从 benchmark 存储层加载「长期谈判」场景（``EnvironmentProfile`` + ``game_metadata``）。

脚本 ``scripts/generate_long_term_negotiation_scenarios.py`` 写入::

    EnvironmentProfile.game_metadata = {
        "pipeline": "long_term_negotiation",
        "quartet": bool,
        "strict_design_v1": bool,
        "timeline": NegotiationTimelineParams 的 dict（``dataclasses.asdict`` 形态）
        "codename": str,
        ...
    }
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .types import NegotiationTimelineParams


@dataclass(frozen=True)
class NegotiationStoredScenario:
    """从库里还原的一局谈判配置（不含 LiteLLM 模型名）。"""

    environment_profile_pk: str
    codename: str
    quartet: bool
    strict_design_v1: bool
    params: NegotiationTimelineParams


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
    strict = bool(gm.get("strict_design_v1", quartet))
    params = negotiation_timeline_params_from_saved_dict(timeline)
    codename = str(gm.get("codename") or "")

    return NegotiationStoredScenario(
        environment_profile_pk=str(env_pk),
        codename=codename,
        quartet=quartet,
        strict_design_v1=strict,
        params=params,
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
            strict_design_v1=parsed.strict_design_v1,
            params=parsed.params,
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


__all__ = [
    "NegotiationStoredScenario",
    "environment_pks_from_manifest",
    "load_negotiation_scenario_from_environment_profile_pk",
    "negotiation_timeline_params_from_saved_dict",
    "parsed_scenario_from_game_metadata",
]
