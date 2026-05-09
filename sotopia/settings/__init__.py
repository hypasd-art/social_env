"""按场景拆分的世界与控制逻辑（独立于 ``sotopia.envs``）。

当前实现：**长期商业谈判**（详见 ``social_env/design_1.md`` 等设计文档）。

中文总览与 LLM 评测 / JSONL 调用顺序见同目录 ``README.zh.md``。
"""

from __future__ import annotations

from .long_term_negotiation import (
    CANONICAL_NEGOTIATION_ROSTER,
    LongTermNegotiationEnv,
    NegotiationContract,
    NegotiationDummyPolicy,
    NegotiationTimelineParams,
    NegotiationWorldController,
    Phase,
    ResolvedSession,
    ROLE_SUMMARY_EN,
    SessionInviteRecord,
    build_rule_dummy_agents,
    compute_negotiation_rule_metrics,
    default_agent_resources_bundle,
    parse_agent_action_payload,
    remaining_days,
    validate_canonical_negotiation_roster,
)

__all__ = [
    "LongTermNegotiationEnv",
    "NegotiationWorldController",
    "NegotiationTimelineParams",
    "NegotiationContract",
    "NegotiationDummyPolicy",
    "build_rule_dummy_agents",
    "compute_negotiation_rule_metrics",
    "Phase",
    "ResolvedSession",
    "SessionInviteRecord",
    "parse_agent_action_payload",
    "remaining_days",
    "CANONICAL_NEGOTIATION_ROSTER",
    "ROLE_SUMMARY_EN",
    "default_agent_resources_bundle",
    "validate_canonical_negotiation_roster",
]
