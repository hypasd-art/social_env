"""``design_1.md`` §10 — 各角色可配置状态变量（仅类型/占位；效用函数实现不在本模块）。

``SystemState.agent_resources`` / ``agent_reputation`` 承载可演化的数值禀赋；本模块记录
与评测/提示词相关的 **心理与私有信息层**（utility 描述、threshold、memory 旋钮等）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, TypedDict, Union


@dataclass
class NegotiationAgentMemoryVariables:
    """§10.* ``memory`` — 与 ``EpisodicMemory`` 等对齐的最小旋钮。"""

    max_entries: int = 40
    inject_recent_lines: int = 8


def _memory_from_blob(raw: Any) -> NegotiationAgentMemoryVariables | None:
    if raw is None:
        return None
    if isinstance(raw, NegotiationAgentMemoryVariables):
        return raw
    if not isinstance(raw, dict):
        return NegotiationAgentMemoryVariables()
    return NegotiationAgentMemoryVariables(
        max_entries=int(raw.get("max_entries", 40)),
        inject_recent_lines=int(raw.get("inject_recent_lines", raw.get("memory_inject_lines", 8))),
    )


def _pinfo(raw: dict[str, Any]) -> dict[str, Any]:
    d = dict(raw.get("private_information") or raw.get("private_info") or {})
    # 兼容简写：`"private_information": "…"` 归入 summary
    if not d and isinstance(raw.get("private_information"), str):
        return {"summary": raw["private_information"]}
    return dict(d)


@dataclass
class FirmAStateVariables:
    """§10.1 ``firm_a``。"""

    utility_spec: str | dict[str, Any] | None = None
    threshold: float | None = None
    expected_acquisition_value: float | None = None
    reputation_anchor: float | None = None
    private_information: dict[str, Any] = field(default_factory=dict)
    memory: NegotiationAgentMemoryVariables | None = None


@dataclass
class FirmBStateVariables:
    """§10.2 ``firm_b``。"""

    utility_spec: str | dict[str, Any] | None = None
    threshold: float | None = None
    asset_value: float | None = None
    reputation_anchor: float | None = None
    private_information: dict[str, Any] = field(default_factory=dict)
    memory: NegotiationAgentMemoryVariables | None = None


@dataclass
class InvestorStateVariables:
    """§10.3 ``investor``。"""

    utility_spec: str | dict[str, Any] | None = None
    threshold: float | None = None
    risk_exposure: float | None = None
    reputation_anchor: float | None = None
    private_information: dict[str, Any] = field(default_factory=dict)
    memory: NegotiationAgentMemoryVariables | None = None


@dataclass
class RegulatorStateVariables:
    """§10.4 ``regulator``（无需货币维度；含政策/使命类结构化占位）。"""

    utility_spec: str | dict[str, Any] | None = None
    approval_threshold: float | None = None
    public_mandate: str | dict[str, Any] | None = None
    policy_constraints: dict[str, Any] = field(default_factory=dict)
    institutional_credibility_anchor: float | None = None
    private_information: dict[str, Any] = field(default_factory=dict)
    memory: NegotiationAgentMemoryVariables | None = None


NegotiationPsychState = Union[
    FirmAStateVariables,
    FirmBStateVariables,
    InvestorStateVariables,
    RegulatorStateVariables,
]


def _psych_role_label(st: NegotiationPsychState) -> str:
    if isinstance(st, FirmAStateVariables):
        return "firm_a"
    if isinstance(st, FirmBStateVariables):
        return "firm_b"
    if isinstance(st, InvestorStateVariables):
        return "investor"
    if isinstance(st, RegulatorStateVariables):
        return "regulator"
    return type(st).__name__


class PsychBundleDict(TypedDict, total=False):
    """场景 JSON 中单 agent 的常见键（均可选）。"""

    utility_spec: str | dict[str, Any]
    threshold: float
    approval_threshold: float
    expected_acquisition_value: float
    asset_value: float
    risk_exposure: float
    reputation_anchor: float
    institutional_credibility_anchor: float
    public_mandate: str | dict[str, Any]
    policy_constraints: dict[str, Any]
    private_information: dict[str, Any]
    private_info: dict[str, Any]
    memory: dict[str, Any]


def firm_a_state_from_dict(raw: Mapping[str, Any] | None) -> FirmAStateVariables:
    raw = dict(raw or {})
    raw.pop("role", None)
    return FirmAStateVariables(
        utility_spec=raw.get("utility_spec"),
        threshold=raw.get("threshold"),
        expected_acquisition_value=raw.get("expected_acquisition_value"),
        reputation_anchor=raw.get("reputation_anchor"),
        private_information=_pinfo(raw),
        memory=_memory_from_blob(raw.get("memory")),
    )


def firm_b_state_from_dict(raw: Mapping[str, Any] | None) -> FirmBStateVariables:
    raw = dict(raw or {})
    raw.pop("role", None)
    return FirmBStateVariables(
        utility_spec=raw.get("utility_spec"),
        threshold=raw.get("threshold"),
        asset_value=raw.get("asset_value"),
        reputation_anchor=raw.get("reputation_anchor"),
        private_information=_pinfo(raw),
        memory=_memory_from_blob(raw.get("memory")),
    )


def investor_state_from_dict(raw: Mapping[str, Any] | None) -> InvestorStateVariables:
    raw = dict(raw or {})
    raw.pop("role", None)
    return InvestorStateVariables(
        utility_spec=raw.get("utility_spec"),
        threshold=raw.get("threshold"),
        risk_exposure=raw.get("risk_exposure"),
        reputation_anchor=raw.get("reputation_anchor"),
        private_information=_pinfo(raw),
        memory=_memory_from_blob(raw.get("memory")),
    )


def regulator_state_from_dict(raw: Mapping[str, Any] | None) -> RegulatorStateVariables:
    raw = dict(raw or {})
    raw.pop("role", None)
    return RegulatorStateVariables(
        utility_spec=raw.get("utility_spec"),
        approval_threshold=raw.get("approval_threshold", raw.get("threshold")),
        public_mandate=raw.get("public_mandate"),
        policy_constraints=dict(raw.get("policy_constraints") or {}),
        institutional_credibility_anchor=raw.get("institutional_credibility_anchor"),
        private_information=_pinfo(raw),
        memory=_memory_from_blob(raw.get("memory")),
    )


_PSYCH_BUILDERS: dict[str, Any] = {
    "firm_a": firm_a_state_from_dict,
    "firm_b": firm_b_state_from_dict,
    "investor": investor_state_from_dict,
    "regulator": regulator_state_from_dict,
}


def negotiation_psych_state_from_role(role: str, raw: Mapping[str, Any] | None) -> NegotiationPsychState:
    fn = _PSYCH_BUILDERS.get(role)
    if fn is None:
        raise ValueError(f"Unknown negotiation role for §10 psych state: {role!r}")
    return fn(raw)


def psych_bundle_from_agent_dicts(
    agent_names: tuple[str, ...],
    specs: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, NegotiationPsychState]:
    """由 ``{{ agent_name: {{...}} }}`` 构造每 agent 的 §10 状态（未见则跳过）。"""
    if not specs:
        return {}
    out: dict[str, NegotiationPsychState] = {}
    for name in agent_names:
        blob = specs.get(name)
        if not blob:
            continue
        b = dict(blob)
        role = str(b.get("role", name))
        out[name] = negotiation_psych_state_from_role(role, b)
    return out


def psych_state_to_prompt_addon(
    st: NegotiationPsychState | None,
    *,
    expose_threshold: bool = False,
) -> str:
    """将 **本 agent** 的 §10 变量格式化为可被拼进观测 digest 的短文本。"""
    if st is None:
        return ""
    lines: list[str] = []
    lines.append(f"[agent_state_variables §10 — {_psych_role_label(st)}]")

    fd: Any
    for fd in fields(st):
        fn = fd.name
        val = getattr(st, fn, None)
        if val is None:
            continue
        if fn == "threshold" or fn == "approval_threshold":
            if not expose_threshold:
                lines.append(f"- {fn}=<hidden> (enable params.expose_psych_threshold_in_observation)")
            else:
                lines.append(f"- {fn}={val!r}")
            continue
        if fn == "private_information":
            if val:
                lines.append(f"- private_information={val!r}")
            continue
        if fn == "memory":
            if val is None:
                continue
            lines.append(
                "- memory="
                + repr(
                    {
                        "max_entries": val.max_entries,
                        "inject_recent_lines": val.inject_recent_lines,
                    }
                )
            )
            continue
        if fn == "policy_constraints" and not val:
            continue
        if is_dataclass(val):
            continue
        lines.append(f"- {fn}={val!r}")

    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


__all__ = [
    "FirmAStateVariables",
    "FirmBStateVariables",
    "InvestorStateVariables",
    "NegotiationAgentMemoryVariables",
    "NegotiationPsychState",
    "PsychBundleDict",
    "RegulatorStateVariables",
    "firm_a_state_from_dict",
    "firm_b_state_from_dict",
    "investor_state_from_dict",
    "negotiation_psych_state_from_role",
    "psych_bundle_from_agent_dicts",
    "psych_state_to_prompt_addon",
    "regulator_state_from_dict",
]
