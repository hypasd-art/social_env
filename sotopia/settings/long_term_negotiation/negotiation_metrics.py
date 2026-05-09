"""长期谈判 **规则型** 可计算指标（与 ``envs/benchmark_evaluators`` 的键风格对齐）。

不依赖 ``ContractLedger`` / JsonModel；输入为 ``LongTermNegotiationEnv`` 或任意带
``ctrl`` + ``system_state`` 的对象，便于在 episode 结束后与 ``compute_individual_metrics``
等函数并列打印 / 落盘。
"""

from __future__ import annotations

from typing import Any


def compute_negotiation_rule_metrics(env: Any) -> dict[str, float]:
    """从 ``LongTermNegotiationEnv``（或鸭子类型）抽取浮点指标。

    Keys 前缀 ``negotiation_*`` ，与 ``benchmark_evaluators`` 的 ``individual_*`` 等区分开。
    """
    ctrl = env.ctrl
    st = env.system_state
    term = getattr(ctrl, "terminal", None) or ""
    out: dict[str, float] = {}
    out["negotiation_terminal_is_success"] = 1.0 if term == "success" else 0.0
    out["negotiation_terminal_is_timeout"] = 1.0 if term == "timeout" else 0.0
    out["negotiation_terminal_is_failure"] = 1.0 if term == "failure" else 0.0
    out["negotiation_terminal_is_max_steps_cap"] = 1.0 if term == "max_steps" or term == "" else 0.0
    macro = float(getattr(env, "last_episode_macro_steps", 0) or 0)
    out["negotiation_macro_steps_used"] = macro
    out["negotiation_n_session_log"] = float(len(getattr(ctrl, "session_log", []) or []))
    out["negotiation_n_action_log"] = float(len(getattr(ctrl, "action_log", []) or []))
    out["negotiation_n_message_log"] = float(len(getattr(ctrl, "message_log", []) or []))
    vh = getattr(ctrl, "visible_history", {}) or {}
    out["negotiation_visible_history_total_lines"] = float(sum(len(v) for v in vh.values()))

    pcs = getattr(ctrl, "primary_contract_id", None)
    if pcs:
        c = getattr(ctrl, "contracts", {}).get(pcs)
        if c is not None:
            stmap = {"proposed": 1.0, "amended": 2.0, "accepted": 3.0, "signed": 4.0, "rejected": -1.0}
            raw = getattr(c, "status", "") or ""
            out["negotiation_primary_contract_phase"] = float(stmap.get(str(raw), 0.0))

    cash_list = [float(st.agent_resources.get(a, {}).get("cash", 0.0)) for a in st.agent_keys]
    if cash_list:
        out["negotiation_participant_mean_cash"] = float(sum(cash_list) / len(cash_list))
        out["negotiation_participant_min_cash"] = float(min(cash_list))

    return out


__all__ = ["compute_negotiation_rule_metrics"]
