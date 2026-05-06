"""
长周期多智能体社会系统 Benchmark v2 - 数据构造层（独立模块）
===============================================================

本文件集中存放 BENCHMARK_DESIGN_zh.md §3 所描述的全部数据相关代码，
**不修改任何现有代码**，可以独立 import / 实例化 / 落库验证。

包含三类内容：

1. Profile 增量字段（不重写 ``AgentProfile`` / ``EnvironmentProfile``，
   而是用并列的 V2 子类承载新字段，老脚本完全不受影响）。
2. 三张新表：``EventScript`` / ``Contract`` / ``SystemStateSnapshot``。
3. ``EpisodeLogV2``：在原 ``EpisodeLog`` 字段之上加 ``schema_version=2``
   与三条 pk 引用（state / events / contracts）+ ``final_metrics``。

设计原则
--------

- **后端无关**：完全沿用 ``is_local_backend()`` 双分支模式，本地 JSON
  与 Redis JsonModel 都能用。
- **pk workaround**：所有 JsonModel 子类都重写 ``__init__`` 显式塞
  ``pk=""``，避免 ``redis-om`` 元类把 ``pk`` 替换成 ``ExpressionProxy``
  导致 pydantic 校验失败（与 ``persistent_profile.py`` 中既有写法一致）。
- **可单文件 import**：``from sotopia.benchmark_v2_data_models import *``
  即可拿到所有新模型，方便先做 PoC 与离线脚本，再逐步并入主流程。

未集成到主流程的部分（事件触发、状态更新、合约生命周期等运行时逻辑）
属于 BENCHMARK_DESIGN_zh.md §B/C/D 的范畴，不在本文件覆盖范围内。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field as PydField
from redis_om import JsonModel
from redis_om.model.model import Field

from sotopia.database.base_models import patch_model_for_local_storage
from sotopia.database.persistent_profile import (
    BaseAgentProfile,
    BaseEnvironmentProfile,
)
from sotopia.database.storage_backend import is_local_backend


# ============================================================================
# 1) Profile 增量字段（§3.1）
#    - V2 子类只 *新增* 字段，旧字段全部继承
#    - 老 AgentProfile / EnvironmentProfile 与老脚本完全不受影响
# ============================================================================


class BaseAgentProfileV2(BaseAgentProfile):
    """在原 BaseAgentProfile 上叠加 benchmark v2 所需的可选字段。

    - ``initial_resources``：智能体起始资源池（dict 形态，便于异质资源）
    - ``initial_reputation``：起始声誉/信用分（0~100）
    - ``risk_preference``：风险偏好（averse/neutral/seeking）
    - ``role_type``：场景内角色（如 buyer/seller/regulator/...）

    所有字段都给了默认值，老数据可零改造转成 V2。
    """

    initial_resources: dict[str, float] = Field(
        default_factory=dict,
        description="起始资源池，例如 {'cash': 100.0, 'water': 20.0}",
    )
    initial_reputation: float = Field(
        default=50.0,
        description="0~100 的初始声誉/信用分",
    )
    risk_preference: Literal["averse", "neutral", "seeking"] = Field(
        default="neutral",
        description="风险偏好，影响策略层先验",
    )
    role_type: str = Field(
        default="",
        index=True,
        description="场景内角色（如 buyer/seller/investor/regulator）",
    )


if TYPE_CHECKING:

    class AgentProfileV2(BaseAgentProfileV2, JsonModel):
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

elif is_local_backend():

    class AgentProfileV2(BaseAgentProfileV2):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

else:

    class AgentProfileV2(BaseAgentProfileV2, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


class BaseEnvironmentProfileV2(BaseEnvironmentProfile):
    """在原 BaseEnvironmentProfile 上叠加 benchmark v2 所需的可选字段。"""

    scenario_type: Literal[
        "negotiation", "investment", "commons", "generic"
    ] = Field(
        default="generic",
        index=True,
        description="场景大类，决定动作集合与评测器选择",
    )
    n_agents: int = Field(
        default=2,
        description="参与该场景的 agent 数量；2 即对齐老 sotopia 行为",
    )
    max_days: int = Field(
        default=1,
        description="时间长度（天）。老场景默认 1 天向后兼容",
    )
    intra_day_steps: int = Field(
        default=1,
        description="每天内的 agent 交互步数",
    )
    event_schedule_pk: Optional[str] = Field(
        default=None,
        index=True,
        description="关联的 EventScript.pk；为空则不触发外部事件",
    )
    system_state_init: dict[str, Any] = Field(
        default_factory=dict,
        description="系统初始状态（市场参数、资源池等），交由 SystemState 解释",
    )


if TYPE_CHECKING:

    class EnvironmentProfileV2(BaseEnvironmentProfileV2, JsonModel):
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

elif is_local_backend():

    class EnvironmentProfileV2(BaseEnvironmentProfileV2):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

else:

    class EnvironmentProfileV2(BaseEnvironmentProfileV2, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


# ============================================================================
# 2) 新增三张表（§3.2）
#    - EventScript / Contract / SystemStateSnapshot
#    - 都以独立 JsonModel 形式存在；老 sotopia 不感知
# ============================================================================


# ---------- 2.1 EventScript ----------

EventVisibility = Literal["public", "partial", "private"]
EventCategory = Literal["news", "market", "policy", "weather", "social"]


class EffectOp(BaseModel):
    """单条 effect 的最小描述单元。

    用类似 DSL 的方式表达：``op`` + ``target`` + ``value``。
    例：``EffectOp(op='delta', target='trust_matrix:A->B', value=-0.2)``
    """

    op: Literal["set", "delta", "disable_action", "broadcast"] = PydField(
        description="set=覆盖；delta=增量；disable_action=禁用动作；broadcast=纯广播"
    )
    target: str = PydField(
        description="作用对象路径，例如 'market_state.interest_rate' "
        "或 'agent_resources.<agent_pk>.cash'",
    )
    value: Any = PydField(default=None, description="目标值或增量")


class BaseEventScript(BaseModel):
    """事件脚本：一份脚本对应一类外部事件，内部可有多条 effect。

    注意 ``apply_days`` 与 ``intraday`` 共同决定何时触发：
    - ``intraday=False`` 且 ``apply_days`` 非空 → 仅在指定 day 的 ``end_of_day`` 触发
    - ``intraday=False`` 且 ``apply_days`` 为空 → 按 ``step``（自定义解析）触发
    - ``intraday=True`` → 允许日内触发，但仍需 config.events.intraday_enabled
    """

    pk: str | None = Field(default_factory=lambda: "")
    name: str = Field(index=True)
    category: EventCategory = Field(index=True)
    visibility: EventVisibility = Field(default="public", index=True)
    intraday: bool = Field(default=False)
    apply_days: list[int] = Field(default_factory=list)
    step: Optional[int] = Field(
        default=None,
        description="可选；当 apply_days 为空时用 step 字段表达触发时刻",
    )
    description: str = Field(default="")
    effects: list[EffectOp] = Field(default_factory=list)
    tag: str = Field(default="", index=True)


if TYPE_CHECKING:

    class EventScript(BaseEventScript, JsonModel):
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

elif is_local_backend():

    class EventScript(BaseEventScript):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

else:

    class EventScript(BaseEventScript, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


# ---------- 2.2 Contract ----------

ContractStatus = Literal[
    "proposed", "active", "fulfilled", "breached", "expired", "cancelled"
]


class BaseContract(BaseModel):
    """合约 / 协议对象：覆盖谈判 / 借贷 / 配额三类场景的最小公共字段。"""

    pk: str | None = Field(default_factory=lambda: "")
    episode_pk: str = Field(index=True, description="所属 EpisodeLogV2.pk")
    proposer_pk: str = Field(index=True)
    counterparties: list[str] = Field(
        default_factory=list,
        description="对手方 agent_pk 列表；多方合约用",
    )
    contract_type: Literal["trade", "loan", "quota", "agreement"] = Field(
        index=True,
    )
    terms: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化条款，举例：{'amount': 100, 'rate': 0.05, 'maturity_day': 30}",
    )
    penalty: dict[str, Any] = Field(
        default_factory=dict,
        description="违约惩罚，举例：{'reputation_delta': -10, 'cash_delta': -50}",
    )
    proposed_day: int = Field(default=0, index=True)
    expiry_day: Optional[int] = Field(default=None, index=True)
    status: ContractStatus = Field(default="proposed", index=True)
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="状态迁移轨迹，元素如 {'day': 5, 'from': 'active', 'to': 'breached'}",
    )


if TYPE_CHECKING:

    class Contract(BaseContract, JsonModel):
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

elif is_local_backend():

    class Contract(BaseContract):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

else:

    class Contract(BaseContract, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


# ---------- 2.3 SystemStateSnapshot ----------


class BaseSystemStateSnapshot(BaseModel):
    """每个 ``end_of_day`` 写一条；day 粒度的状态快照。"""

    pk: str | None = Field(default_factory=lambda: "")
    episode_pk: str = Field(index=True)
    day: int = Field(index=True)

    # 关系/信任：信任矩阵以邻接稀疏 dict 表达，避免与 agent 数 N x N 强耦合
    trust_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    public_opinion: dict[str, float] = Field(
        default_factory=dict,
        description="agent_pk -> 公众舆论分（可来自 LLM 评估或规则）",
    )

    # 市场/资源
    market_state: dict[str, float] = Field(
        default_factory=dict,
        description="例：{'interest_rate': 0.05, 'price_index': 100.0}",
    )
    resource_pool: dict[str, float] = Field(
        default_factory=dict,
        description="公共资源池，例：{'water': 80.0}",
    )
    agent_resources: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="agent_pk -> 个人资源 dict",
    )
    agent_reputation: dict[str, float] = Field(default_factory=dict)

    # 元数据
    triggered_event_pks: list[str] = Field(default_factory=list)
    contracts_changed_pks: list[str] = Field(default_factory=list)


if TYPE_CHECKING:

    class SystemStateSnapshot(BaseSystemStateSnapshot, JsonModel):
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

elif is_local_backend():

    class SystemStateSnapshot(BaseSystemStateSnapshot):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

else:

    class SystemStateSnapshot(BaseSystemStateSnapshot, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


# ============================================================================
# 3) EpisodeLogV2（§3.3）
#    - 不修改老 EpisodeLog
#    - 用 schema_version=2 标记 + 增加 state/events/contracts/metrics 引用
# ============================================================================


class BaseEpisodeLogV2(BaseModel):
    """v2 episode 日志：旧字段不全部复制，仅保留 *主表所需* 元数据；
    详细对话与奖励仍可走老 EpisodeLog（用 ``legacy_episode_pk`` 关联），
    或在 PoC 阶段直接把 messages/rewards 也写进来。

    与老 EpisodeLog 的兼容策略：
    - 如果是老格式数据 → 仍写 EpisodeLog，``schema_version`` 隐式为 1
    - 如果是 v2 benchmark 数据 → 写 EpisodeLogV2，``schema_version=2``
    """

    pk: str | None = Field(default_factory=lambda: "")
    schema_version: Literal[2] = Field(default=2, index=True)

    environment_pk: str = Field(index=True)
    agent_pks: list[str] = Field(index=True)
    tag: str = Field(default="", index=True)
    models: list[str] = Field(default_factory=list, index=True)

    legacy_episode_pk: Optional[str] = Field(
        default=None,
        index=True,
        description="可选；指向老 EpisodeLog.pk，用于双写过渡期",
    )

    # 引用而非内嵌，保持单条 JSON 体积可控
    state_trajectory_pks: list[str] = Field(
        default_factory=list,
        description="按 day 顺序的 SystemStateSnapshot.pk 列表",
    )
    events_log_pks: list[str] = Field(
        default_factory=list,
        description="本 episode 实际触发过的 EventScript.pk（含同一脚本多次触发的多条记录的 pk）",
    )
    contracts_pks: list[str] = Field(
        default_factory=list,
        description="本 episode 内涉及到的 Contract.pk",
    )

    # PoC 阶段也允许直接内嵌；正式实验建议走 trajectory_pks
    messages: list[list[tuple[str, str, str]]] = Field(default_factory=list)
    rewards: list[float] = Field(default_factory=list)
    final_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="四层评测最终值，例：{'welfare': 0.8, 'gini': 0.31, 'default_rate': 0.1}",
    )
    reasoning: str = Field(default="")


if TYPE_CHECKING:

    class EpisodeLogV2(BaseEpisodeLogV2, JsonModel):
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

elif is_local_backend():

    class EpisodeLogV2(BaseEpisodeLogV2):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)

else:

    class EpisodeLogV2(BaseEpisodeLogV2, JsonModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            if "pk" not in kwargs:
                kwargs["pk"] = ""
            super().__init__(**kwargs)


# 为本地后端打补丁，与既有模型保持一致
AgentProfileV2 = patch_model_for_local_storage(AgentProfileV2)  # type: ignore[misc]
EnvironmentProfileV2 = patch_model_for_local_storage(EnvironmentProfileV2)  # type: ignore[misc]
EventScript = patch_model_for_local_storage(EventScript)  # type: ignore[misc]
Contract = patch_model_for_local_storage(Contract)  # type: ignore[misc]
SystemStateSnapshot = patch_model_for_local_storage(SystemStateSnapshot)  # type: ignore[misc]
EpisodeLogV2 = patch_model_for_local_storage(EpisodeLogV2)  # type: ignore[misc]


# ============================================================================
# 4) 工厂函数 / 升级映射（构造层最常用的入口）
#    - 这一节是真正“数据构造”的便利层；上层脚本只用调这几个函数即可
# ============================================================================


def upgrade_agent_profile(
    legacy_profile: BaseAgentProfile,
    *,
    initial_resources: Optional[dict[str, float]] = None,
    initial_reputation: float = 50.0,
    risk_preference: Literal["averse", "neutral", "seeking"] = "neutral",
    role_type: str = "",
) -> "AgentProfileV2":
    """把老 AgentProfile 升级成 V2，老字段全部沿用，新字段走默认值或入参。

    刻意不持久化（不调 save），调用方决定是否落库，方便离线脚本批量处理。
    """

    payload = legacy_profile.model_dump()
    payload.pop("pk", None)
    payload["initial_resources"] = initial_resources or {}
    payload["initial_reputation"] = initial_reputation
    payload["risk_preference"] = risk_preference
    payload["role_type"] = role_type
    return AgentProfileV2(**payload)


def upgrade_environment_profile(
    legacy_profile: BaseEnvironmentProfile,
    *,
    scenario_type: Literal[
        "negotiation", "investment", "commons", "generic"
    ] = "generic",
    n_agents: int = 2,
    max_days: int = 1,
    intra_day_steps: int = 1,
    event_schedule_pk: Optional[str] = None,
    system_state_init: Optional[dict[str, Any]] = None,
) -> "EnvironmentProfileV2":
    """把老 EnvironmentProfile 升级成 V2。"""

    payload = legacy_profile.model_dump()
    payload.pop("pk", None)
    payload["scenario_type"] = scenario_type
    payload["n_agents"] = n_agents
    payload["max_days"] = max_days
    payload["intra_day_steps"] = intra_day_steps
    payload["event_schedule_pk"] = event_schedule_pk
    payload["system_state_init"] = system_state_init or {}
    return EnvironmentProfileV2(**payload)


def make_initial_state_snapshot(
    *,
    episode_pk: str,
    agent_pks: list[str],
    initial_resources_per_agent: Optional[dict[str, dict[str, float]]] = None,
    initial_reputation_per_agent: Optional[dict[str, float]] = None,
    market_state: Optional[dict[str, float]] = None,
    resource_pool: Optional[dict[str, float]] = None,
) -> "SystemStateSnapshot":
    """生成 day=0 的初始 ``SystemStateSnapshot``。"""

    return SystemStateSnapshot(
        episode_pk=episode_pk,
        day=0,
        trust_matrix={a: {} for a in agent_pks},
        public_opinion={a: 0.0 for a in agent_pks},
        market_state=market_state or {},
        resource_pool=resource_pool or {},
        agent_resources=initial_resources_per_agent or {a: {} for a in agent_pks},
        agent_reputation=initial_reputation_per_agent or {a: 50.0 for a in agent_pks},
    )


def make_event_script_from_dict(spec: dict[str, Any]) -> "EventScript":
    """从 dict（通常来自 JSON 文件）构造 ``EventScript``。

    - ``spec['effects']`` 应为 ``EffectOp`` 兼容的 dict 列表
    - 其他字段名与 ``BaseEventScript`` 一致
    """

    effects_raw = spec.get("effects", [])
    effects = [EffectOp(**e) if not isinstance(e, EffectOp) else e for e in effects_raw]
    payload = {**spec, "effects": effects}
    return EventScript(**payload)


__all__ = [
    "BaseAgentProfileV2",
    "AgentProfileV2",
    "BaseEnvironmentProfileV2",
    "EnvironmentProfileV2",
    "EventVisibility",
    "EventCategory",
    "EffectOp",
    "BaseEventScript",
    "EventScript",
    "ContractStatus",
    "BaseContract",
    "Contract",
    "BaseSystemStateSnapshot",
    "SystemStateSnapshot",
    "BaseEpisodeLogV2",
    "EpisodeLogV2",
    "upgrade_agent_profile",
    "upgrade_environment_profile",
    "make_initial_state_snapshot",
    "make_event_script_from_dict",
]
