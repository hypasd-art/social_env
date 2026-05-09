"""结构化 effect → ``SystemState`` 的原地更新 DSL。

兼容 ``benchmark_v2_data_models.EffectOp``、普通 ``dict`` 或任意带有
``op`` / ``target`` / ``value`` 属性的对象，便于在无 JsonModel/redis 的配置下仍能跑闭环。
"""

from __future__ import annotations

from typing import Any

from sotopia.state.system_state import SystemState


def _eff_prop(effect: Any, key: str, default: Any = None) -> Any:
    if isinstance(effect, dict):
        return effect.get(key, default)
    return getattr(effect, key, default)


def apply_effect_op(state: SystemState, effect: Any) -> None:
    """就地修改 ``state``。"""
    op = str(_eff_prop(effect, "op", "") or "")
    tgt_raw = _eff_prop(effect, "target", "") or ""
    tgt = str(tgt_raw).strip()
    eff_val = _eff_prop(effect, "value", None)
    if op == "broadcast":
        return

    if tgt.startswith("trust_matrix:"):
        rest = tgt.split(":", 1)[1]
        if "->" in rest:
            src, dst = rest.split("->", 1)
            src, dst = src.strip(), dst.strip()
            state.trust_matrix.setdefault(src, {})
            cur = float(state.trust_matrix[src].get(dst, 0.0))
            if op == "set":
                state.trust_matrix[src][dst] = float(eff_val or 0.0)
            elif op == "delta":
                state.trust_matrix[src][dst] = cur + float(eff_val or 0.0)
        return

    if tgt.startswith("public_opinion."):
        agent = tgt.split(".", 1)[1]
        cur = float(state.public_opinion.get(agent, 0.0))
        if op == "set":
            state.public_opinion[agent] = float(eff_val or 0.0)
        elif op == "delta":
            state.public_opinion[agent] = cur + float(eff_val or 0.0)
        return

    if tgt.startswith("agent_reputation."):
        agent = tgt.split(".", 1)[1]
        cur = float(state.agent_reputation.get(agent, 50.0))
        if op == "set":
            state.agent_reputation[agent] = float(eff_val or 0.0)
        elif op == "delta":
            state.agent_reputation[agent] = cur + float(eff_val or 0.0)
        return

    if tgt.startswith("market_state."):
        field = tgt.split(".", 1)[1]
        cur = float(state.market_state.get(field, 0.0))
        if op == "set":
            state.market_state[field] = float(eff_val or 0.0)
        elif op == "delta":
            state.market_state[field] = cur + float(eff_val or 0.0)
        return

    if tgt.startswith("resource_pool."):
        field = tgt.split(".", 1)[1]
        cur = float(state.resource_pool.get(field, 0.0))
        if op == "set":
            state.resource_pool[field] = float(eff_val or 0.0)
        elif op == "delta":
            state.resource_pool[field] = cur + float(eff_val or 0.0)
        return

    if tgt.startswith("agent_resources."):
        _, agent, resource = tgt.split(".", 2)
        state.agent_resources.setdefault(agent, {})
        cur = float(state.agent_resources[agent].get(resource, 0.0))
        if op == "set":
            state.agent_resources[agent][resource] = float(eff_val or 0.0)
        elif op == "delta":
            state.agent_resources[agent][resource] = cur + float(eff_val or 0.0)
        return

    if op == "disable_action":
        return


__all__ = ["apply_effect_op"]
