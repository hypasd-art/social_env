"""日内与跨日累积的可变系统状态（信任、市场、公共资源、个体资源与声誉）。

与 ``benchmark_v2_data_models.SystemStateSnapshot`` 对齐：``to_snapshot`` 可把当前
内存态冻结为持久化快照行。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sotopia.benchmark_v2_data_models import SystemStateSnapshot


class SystemState:
    """维护长周期评测所需的一组可变 dict，均由 agent 名称（环境与 inbox 中使用的主键）索引。"""

    def __init__(
        self,
        agent_keys: list[str],
        *,
        trust_matrix: dict[str, dict[str, float]] | None = None,
        public_opinion: dict[str, float] | None = None,
        market_state: dict[str, float] | None = None,
        resource_pool: dict[str, float] | None = None,
        agent_resources: dict[str, dict[str, float]] | None = None,
        agent_reputation: dict[str, float] | None = None,
    ) -> None:
        self.agent_keys = list(agent_keys)
        self.trust_matrix: dict[str, dict[str, float]] = trust_matrix or {
            a: {} for a in self.agent_keys
        }
        self.public_opinion: dict[str, float] = public_opinion or {
            a: 0.0 for a in self.agent_keys
        }
        self.market_state: dict[str, float] = dict(market_state or {})
        self.resource_pool: dict[str, float] = dict(resource_pool or {})
        self.agent_resources: dict[str, dict[str, float]] = {
            a: dict(agent_resources.get(a, {}) if agent_resources else {})  # type: ignore[union-attr]
            for a in self.agent_keys
        }
        self.agent_reputation: dict[str, float] = {
            a: float(agent_reputation.get(a, 50.0) if agent_reputation else 50.0)  # type: ignore[union-attr]
            for a in self.agent_keys
        }
        self.scratch: dict[str, Any] = {}

    def rebind_agents(self, agent_keys: list[str]) -> None:
        """reset 换新一批参与者时重置各 dict 的结构（值可再由 init dict 填充）。"""

        self.agent_keys = list(agent_keys)
        seen = set(self.agent_keys)
        self.trust_matrix = {a: dict(self.trust_matrix.get(a, {})) for a in self.agent_keys}
        self.public_opinion = {
            a: float(self.public_opinion.get(a, 0.0)) for a in self.agent_keys
        }
        self.agent_resources = {
            a: dict(self.agent_resources.get(a, {})) for a in self.agent_keys
        }
        self.agent_reputation = {
            a: float(self.agent_reputation.get(a, 50.0)) for a in self.agent_keys
        }
        self.scratch = {}
        for a in list(self.trust_matrix.keys()):
            if a not in seen:
                del self.trust_matrix[a]

    def apply_effect(self, effect: Any) -> None:
        """执行单条 effect（``dict`` / ``EffectOp`` / duck-typed；见 ``effect_dsl.apply_effect_op``）。"""
        from sotopia.events.effect_dsl import apply_effect_op

        apply_effect_op(self, effect)

    def digest_line(self, *, viewer: str) -> str:
        """给单个 agent 的观测附加用：一行可读摘要。"""
        res = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(self.agent_resources.get(viewer, {}).items())
        )
        rep = self.agent_reputation.get(viewer, 50.0)
        keys = ",".join(
            f"{o}:{self.trust_matrix.get(viewer, {}).get(o, 0):.2f}"
            for o in self.agent_keys
            if o != viewer
        )
        mk = ",".join(f"{k}:{v:.3f}" for k, v in sorted(self.market_state.items()))
        return (
            f"[system] reputation={rep:.2f} resources{{{res}}} "
            f"trust{{{keys}}} market{{{mk}}}"
        )

    def to_snapshot(
        self,
        *,
        episode_pk: str,
        day: int,
        triggered_event_pks: list[str],
        contracts_changed_pks: list[str],
    ) -> "SystemStateSnapshot":
        from sotopia.benchmark_v2_data_models import SystemStateSnapshot

        return SystemStateSnapshot(
            episode_pk=episode_pk,
            day=day,
            trust_matrix={
                k: dict(v)
                for k, v in self.trust_matrix.items()
                if k in self.agent_keys
            },
            public_opinion={k: float(self.public_opinion.get(k, 0.0)) for k in self.agent_keys},
            market_state=dict(self.market_state),
            resource_pool=dict(self.resource_pool),
            agent_resources={
                k: dict(self.agent_resources.get(k, {})) for k in self.agent_keys
            },
            agent_reputation=dict(self.agent_reputation),
            triggered_event_pks=list(triggered_event_pks),
            contracts_changed_pks=list(contracts_changed_pks),
        )


def state_from_profile_init(agent_keys: list[str], payload: dict[str, Any]) -> SystemState:
    """解析 ``EnvironmentProfileV2.system_state_init``。"""
    raw_ar = payload.get("agent_resources") or {}
    agent_resources_map: dict[str, dict[str, float]] | None = None
    if isinstance(raw_ar, dict):
        tmp: dict[str, dict[str, float]] = {}
        for a, d in raw_ar.items():
            if not isinstance(d, dict):
                continue
            if agent_keys and str(a) not in agent_keys:
                continue
            tmp[str(a)] = {str(k): float(v) for k, v in d.items()}
        agent_resources_map = tmp or None
    raw_rep = payload.get("agent_reputation")
    reputation_map = (
        {str(a): float(v) for a, v in raw_rep.items()} if isinstance(raw_rep, dict) else None
    )

    return SystemState(
        agent_keys,
        market_state={
            str(k): float(v)
            for k, v in (payload.get("market_state") or {}).items()
        },
        resource_pool={
            str(k): float(v)
            for k, v in (payload.get("resource_pool") or {}).items()
        },
        agent_resources=agent_resources_map if agent_resources_map is not None else None,
        agent_reputation=reputation_map,
        trust_matrix=None,
        public_opinion=None,
    )


__all__ = ["SystemState", "state_from_profile_init"]
