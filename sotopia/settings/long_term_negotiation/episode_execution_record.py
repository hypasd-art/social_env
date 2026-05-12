"""单条谈判 episode 的 **全局执行档案**：时间线 + 合同历史 + 审计日志，便于复盘「何时签约」等。

由 ``llm_evaluation.run_llm_negotiation_episode_evaluation`` 在 ``execution_trace_dir`` 非空时
在 episode 结束后写入 ``{dir}/{tag}.execution.json``。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .env import LongTermNegotiationEnv

__all__ = ["build_episode_execution_record", "safe_execution_trace_filename", "write_episode_execution_record"]


def safe_execution_trace_filename(tag: str) -> str:
    """与 ``model_trace.safe_trace_filename`` 同规则，扩展名为 ``.execution.json``。"""
    base = (tag or "negotiation_episode").strip()
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r"[^-._a-zA-Z0-9]", "_", base)
    base = base.strip("._") or "negotiation_episode"
    if len(base) > 160:
        base = base[:160]
    return f"{base}.execution.json"


def _tupleize_scheduling(row: tuple[Any, ...] | list[Any]) -> list[Any]:
    return list(row) if isinstance(row, tuple) else row


def build_episode_execution_record(env: LongTermNegotiationEnv) -> dict[str, Any]:
    """汇总 ``NegotiationWorldController`` 内已有结构与 ``execution_timeline``。"""
    ctrl = env.ctrl
    contracts_out: dict[str, Any] = {}
    for cid, c in ctrl.contracts.items():
        contracts_out[cid] = {
            "status": getattr(c, "status", None),
            "parties": sorted(getattr(c, "parties", ()) or ()),
            "created_day": getattr(c, "created_day", None),
            "created_slot": getattr(c, "created_slot", None),
            "financing": dict(getattr(c, "financing", {}) or {}),
            "regulatory": dict(getattr(c, "regulatory", {}) or {}),
            "history": list(getattr(c, "history", []) or []),
        }
    return {
        "schema": "sotopia.long_term_negotiation.execution_record.v1",
        "terminal": ctrl.terminal,
        "macro_steps_used": int(getattr(env, "last_episode_macro_steps", 0) or 0),
        "execution_timeline": list(getattr(ctrl, "execution_timeline", []) or []),
        "event_log": list(getattr(ctrl, "event_log", []) or []),
        "scheduling_log": [_tupleize_scheduling(t) for t in getattr(ctrl, "scheduling_log", []) or []],
        "session_log": list(getattr(ctrl, "session_log", []) or []),
        "action_log": list(getattr(ctrl, "action_log", []) or []),
        "message_log": list(getattr(ctrl, "message_log", []) or []),
        "state_snapshots": list(getattr(ctrl, "state_snapshots", []) or []),
        "contracts": contracts_out,
        "primary_contract_id": getattr(ctrl, "primary_contract_id", None),
    }


def write_episode_execution_record(env: LongTermNegotiationEnv, path: Path | str) -> Path:
    """写入 UTF-8 JSON（缩进 2，便于人工阅读）。"""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = build_episode_execution_record(env)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p
