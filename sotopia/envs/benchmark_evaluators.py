"""长周期 benchmark 规则指标（§F）：individual / social / behavioral / long_horizon 的**可计算**子集。

这些指标叠加在 ``SocialSystemEnv`` 上：``Evaluator.__call__`` / ``__acall__`` 通过
``kwargs[\"env\"]`` 读取 ``env.system_state``（及可选 ``contracts``）。

长期商业谈判离线脚本可用的 **轻量谈判指标** 见 ``sotopia.settings.long_term_negotiation.negotiation_metrics``，
其键前缀为 ``negotiation_*`` ，可与本节返回的条目合并到一个 dict 再做分析。

大模型主观维度请在实验脚本里另行组合 ``EpisodeLLMEvaluator``。
"""

from __future__ import annotations

import statistics
from typing import Any

from sotopia.envs.evaluators import Evaluator
from sotopia.messages import Message

from sotopia.state.contracts import ContractLedger
from sotopia.state.system_state import SystemState


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(max(0.0, float(x)) for x in values)
    n = len(xs)
    if n == 0 or sum(xs) < 1e-12:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2 * cum / (n * sum(xs))) - (n + 1) / n


def compute_individual_metrics(state: SystemState) -> dict[str, float]:
    """效用代理：现金持有；一致性代理：声誉与信任均值的差距。"""
    cash_list = [float(state.agent_resources.get(a, {}).get("cash", 0.0)) for a in state.agent_keys]
    mean_cash = sum(cash_list) / len(cash_list) if cash_list else 0.0
    regret_proxy = max(cash_list) - min(cash_list) if len(cash_list) > 1 else 0.0
    rep_vals = [float(state.agent_reputation.get(a, 50.0)) for a in state.agent_keys]
    mean_rep = sum(rep_vals) / len(rep_vals) if rep_vals else 0.0
    trust_rows = []
    for a in state.agent_keys:
        row = state.trust_matrix.get(a, {})
        if row:
            trust_rows.append(sum(row.values()) / max(len(row), 1))
    mean_trust = sum(trust_rows) / len(trust_rows) if trust_rows else 0.0
    consistency = 1.0 / (1.0 + abs(mean_rep - 50.0) / 50.0)
    return {
        "individual_mean_cash": mean_cash,
        "individual_regret_range": regret_proxy,
        "individual_mean_reputation": mean_rep,
        "individual_consistency": consistency,
        "individual_mean_trust_out": mean_trust,
    }


def compute_social_metrics(state: SystemState) -> dict[str, float]:
    cash_list = [float(state.agent_resources.get(a, {}).get("cash", 0.0)) for a in state.agent_keys]
    total_welfare = sum(cash_list)
    gini_cash = _gini(cash_list)
    reps = [float(state.agent_reputation.get(a, 50.0)) for a in state.agent_keys]
    stability = 1.0 / (1.0 + float(statistics.pstdev(reps)) if len(reps) > 1 else 0.0)
    return {
        "social_total_welfare": total_welfare,
        "social_gini_cash": gini_cash,
        "social_stability_rep": stability,
    }


def compute_behavioral_metrics(state: SystemState) -> dict[str, float]:
    coop = []
    for a in state.agent_keys:
        for b, v in state.trust_matrix.get(a, {}).items():
            if b != a and v > 0:
                coop.append(float(v))
    cooperation = sum(coop) / len(coop) if coop else 0.0
    punishments = sum(
        1.0 for a in state.agent_keys if float(state.agent_reputation.get(a, 50.0)) < 40.0
    )
    return {
        "behavioral_cooperation": cooperation,
        "behavioral_low_reputation_count": punishments,
    }


def compute_long_horizon_proxy(state: SystemState, turn_number: int) -> dict[str, float]:
    """无轨迹模型时的粗代理：轮次 + scratch 体积。"""
    return {
        "long_horizon_turn_index": float(turn_number),
        "long_horizon_scratch_size": float(len(str(state.scratch))),
    }


def compute_default_rate(ledger: ContractLedger | None) -> float:
    if ledger is None:
        return 0.0
    contracts = list(ledger.iter_contracts())
    if not contracts:
        return 0.0
    n_breach = sum(1 for c in contracts if c.status == "breached")
    return float(n_breach) / float(len(contracts))


class IndividualMetricsEvaluator(Evaluator):
    def __call__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]],
        **kwargs: Any,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if env is None or not hasattr(env, "system_state"):
            return []
        m = compute_individual_metrics(env.system_state)
        return [("environment", ((k, float(v)), "")) for k, v in m.items()]

    async def __acall__(self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any) -> list[
        tuple[str, tuple[tuple[str, int | float | bool], str]]
    ]:
        return self(turn_number, messages, **kwargs)


class SocialMetricsEvaluator(Evaluator):
    def __call__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]],
        **kwargs: Any,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if env is None or not hasattr(env, "system_state"):
            return []
        m = compute_social_metrics(env.system_state)
        dr = compute_default_rate(getattr(env, "contracts", None))
        m2 = {**m, "social_default_rate": dr}
        return [("environment", ((k, float(v)), "")) for k, v in m2.items()]

    async def __acall__(self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any) -> list[
        tuple[str, tuple[tuple[str, int | float | bool], str]]
    ]:
        return self(turn_number, messages, **kwargs)


class BehavioralMetricsEvaluator(Evaluator):
    def __call__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]],
        **kwargs: Any,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if env is None or not hasattr(env, "system_state"):
            return []
        m = compute_behavioral_metrics(env.system_state)
        return [("environment", ((k, float(v)), "")) for k, v in m.items()]

    async def __acall__(self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any) -> list[
        tuple[str, tuple[tuple[str, int | float | bool], str]]
    ]:
        return self(turn_number, messages, **kwargs)


class LongHorizonMetricsEvaluator(Evaluator):
    def __call__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]],
        **kwargs: Any,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if env is None or not hasattr(env, "system_state"):
            return []
        m = compute_long_horizon_proxy(env.system_state, turn_number)
        return [("environment", ((k, float(v)), "")) for k, v in m.items()]

    async def __acall__(self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any) -> list[
        tuple[str, tuple[tuple[str, int | float | bool], str]]
    ]:
        return self(turn_number, messages, **kwargs)


class BenchmarkMetricsBundleEvaluator(Evaluator):
    """一次性展开全部规则指标（便于调试）；生产环境可拆成四类 Evaluator 以降低耦合。"""

    def __call__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]],
        **kwargs: Any,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        env = kwargs.get("env")
        if env is None or not hasattr(env, "system_state"):
            return []
        st = env.system_state
        ledger = getattr(env, "contracts", None)
        merged: dict[str, float] = {}
        merged.update(compute_individual_metrics(st))
        merged.update(compute_social_metrics(st))
        merged.update({"social_default_rate": compute_default_rate(ledger)})
        merged.update(compute_behavioral_metrics(st))
        merged.update(compute_long_horizon_proxy(st, turn_number))
        return [("environment", ((k, float(v)), "")) for k, v in merged.items()]

    async def __acall__(self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any) -> list[
        tuple[str, tuple[tuple[str, int | float | bool], str]]
    ]:
        return self(turn_number, messages, **kwargs)


__all__ = [
    "IndividualMetricsEvaluator",
    "SocialMetricsEvaluator",
    "BehavioralMetricsEvaluator",
    "LongHorizonMetricsEvaluator",
    "BenchmarkMetricsBundleEvaluator",
    "compute_individual_metrics",
    "compute_social_metrics",
    "compute_behavioral_metrics",
    "compute_long_horizon_proxy",
    "compute_default_rate",
]
