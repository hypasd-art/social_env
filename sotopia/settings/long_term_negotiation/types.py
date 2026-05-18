"""长期商业谈判世界的类型（对应 ``design_1.md``）；不修改 ``message_classes``。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# `firms_only` lineup 的 speaker 顺序。
SESSION_FIRMS_ONLY_ROLE_ORDER: tuple[str, ...] = ("firm_a", "firm_b", "firm_c", "firm_d")
SESSION_SPEAKER_ROLE_ORDER = SESSION_FIRMS_ONLY_ROLE_ORDER  # 别名，兼容脚本直接导入

#: 受支持的 lineup 字符串（写入 ``EnvironmentProfile.game_metadata.lineup``）。
NEGOTIATION_LINEUP_FIRMS_ONLY: str = "firms_only"
SUPPORTED_NEGOTIATION_LINEUPS: frozenset[str] = frozenset({NEGOTIATION_LINEUP_FIRMS_ONLY})


def negotiation_role_order(lineup: str) -> tuple[str, ...]:
    """按 ``lineup`` 返回 canonical 角色顺序（取前 N 即可拿到 N 人 roster）。"""
    if lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
        return SESSION_FIRMS_ONLY_ROLE_ORDER
    raise ValueError(
        f"unknown negotiation lineup {lineup!r}; expected one of "
        f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}"
    )


class Phase(str, Enum):
    INIT = "init"
    SCHEDULE_INVITE = "schedule_invite"
    SCHEDULE_RESPONSE = "schedule_response"
    SESSION = "session"
    POST_SESSION = "post_session"
    END_OF_DAY = "end_of_day"
    TERMINATED = "terminated"


class WorldTerminalKind(str, Enum):
    """``design_1.md`` §9 — world-level 终止状态（与 ``NegotiationWorldController.terminal`` 取值对齐）。"""

    SUCCESS = "success"
    FAILURE = "failure"
    TERMINATED_BY_AGENT = "terminated_by_agent"
    TIMEOUT = "timeout"


@dataclass
class NegotiationTimelineParams:
    """与时间、容量相关的可配置参数（与设计文档符号对齐）。"""

    D: int = 10

    s_max_per_day: int = 3

    max_session_rounds: int = 16

    #: §1.2 / §1.1 — 校验融资可承诺额度、监管硬违规等；评测可关。
    enforce_formal_budget_checks: bool = True

    #: §2.1 — 每名 agent 每自然日至多发出的 formal(JSON) 次数；``None`` 表示不限制。
    max_formal_actions_per_agent_per_day: int | None = None

    #: §2.1 / §6.3 — ``M_max``：每名 agent 在每 session 内 message（speak / non-verbal）条数上限；``None`` 不限制。
    max_natural_turns_per_agent_per_session: int | None = None

    #: §6.4 — ``H_max``：每名 agent 在每 session 内 formal 次数上限（与 ``F_max`` 同时生效时取更紧者）；``None`` 不限制。
    max_formal_actions_per_agent_per_session: int | None = None

    #: §6.5 — 每名 agent 在每 session 内 session_control（leave / terminate_session）次数上限；``None`` 不限制。
    max_session_control_actions_per_agent_per_session: int | None = None

    #: §4.3 — ``T_s``：本会话允许的总会话回合（macro turn）上限；``None`` 则取
    #: ``max_session_rounds * |P_s|``（与同字段旧语义相容）。
    max_total_turns_per_session: int | None = None

    #: §4.3 — ``K_s``：每名 participant 在本 session 内的 macro turn 上限；``None`` 表示不按人均上限结束。
    max_turns_per_participant_per_session: int | None = None

    #: §8 — 外部事件配置（每项为 dict，参见 ``external_events.external_event_from_dict``）。
    external_event_specs: tuple[dict[str, Any], ...] = ()

    #: §9.3 ``failure`` — 若连续若干个**自然日**内未发生会话成立或合同/融资监管侧结构性进展，
    #: 控制器在当日 ``end_day_tick`` 末尾终止世界；``None`` 关闭此判定（默认）。
    failure_stagnation_calendar_days: int | None = None

    #: §10 — 是否在观测 digest 中暴露 ``threshold``（默认隐藏）。
    expose_psych_threshold_in_observation: bool = False


@dataclass
class SessionRuntimeMeta:
    """§4.1 — session 元数据（与 ``ResolvedSession`` 并行挂载）。"""

    session_id: str
    day: int
    slot: int
    participants_start: tuple[str, ...]
    t_start_global_turn: int
    t_end_global_turn: int | None = None
    status: str = "active"  # active | closed


@dataclass
class SessionInviteRecord:
    requester: str
    proposed_participants: frozenset[str]
    purpose: str
    slot: int
    day: int


@dataclass
class ResolvedSession:
    session_id: str
    day: int
    slot: int
    participants: tuple[str, ...]


# 合同主体（principal）= 所有公司角色。
PRINCIPAL_PARTY_ROLES: frozenset[str] = frozenset(
    {"firm_a", "firm_b", "firm_c", "firm_d"}
)


@dataclass
class NegotiationContract:
    """§5 — 控制器维护的全局合同账本（独立于 ``benchmark_v2_data_models.Contract``）。

    ``status ∈ proposed | amended | accepted | rejected | signed | superseded | failed`` （可扩充）。
    """

    contract_id: str
    parent_id: str | None
    status: str
    #: §5 条款（Setting 1 可含 valuation / payment / closing / compliance / penalty 等键）。
    terms: dict[str, Any]
    created_by: str = ""
    #: ``{ day, slot_id, session_id, turn_id }`` — §5 ``created_at``。
    created_at: dict[str, Any] = field(default_factory=dict)
    parties: set[str] = field(default_factory=set)
    acceptances: dict[str, bool | None] = field(default_factory=dict)
    visibility: set[str] = field(default_factory=set)
    signatures: dict[str, bool] = field(default_factory=dict)
    financing: dict[str, Any] = field(default_factory=lambda: {"required": 0, "status": "not_required", "actor": None})
    regulatory: dict[str, Any] = field(default_factory=lambda: {"required": 0, "status": "not_required", "actor": None})
    history: list[dict[str, Any]] = field(default_factory=list)
    created_day: int = 0
    created_slot: int = 0
