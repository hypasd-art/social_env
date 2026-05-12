"""单条谈判 episode 的 **全局执行档案**：时间线 + 合同历史 + 审计日志，便于复盘「何时签约」等。

由 ``run_llm_negotiation_episode_evaluation`` 在 ``execution_trace_dir`` 非空时
在 episode 结束后写入 ``{dir}/{tag}.execution.json``，并在默认开启时 **额外** 写入同 stem 的
``{dir}/{tag}.execution.transcript.txt``：按时间顺序整理 **完整** ``LongTermNegotiationEnv.inbox``
（环境侧累积的全部 Environment / 各 agent 行动）及调度、会话、可见历史等；若同时开启
``model_trace_dir`` 且传入与 trace 一致的 stem，则合并 **逐次 LLM 调用的完整输入与原始输出**
（``llm_model_traces`` / transcript §8）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .env import LongTermNegotiationEnv

__all__ = [
    "build_episode_execution_record",
    "format_episode_interaction_transcript",
    "safe_execution_trace_filename",
    "transcript_path_for_execution_json",
    "write_episode_execution_record",
]


def safe_execution_trace_filename(tag: str) -> str:
    """与 ``model_trace.safe_trace_filename`` 同规则，扩展名为 ``.execution.json``。"""
    base = (tag or "negotiation_episode").strip()
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r"[^-._a-zA-Z0-9]", "_", base)
    base = base.strip("._") or "negotiation_episode"
    if len(base) > 160:
        base = base[:160]
    return f"{base}.execution.json"


def transcript_path_for_execution_json(execution_json_path: Path | str) -> Path:
    """由 ``*.execution.json`` 路径得到配对的 ``*.execution.transcript.txt`` 路径。"""
    p = Path(execution_json_path).resolve()
    name = p.name
    if name.endswith(".execution.json"):
        stem = name[: -len(".execution.json")]
        return p.with_name(f"{stem}.execution.transcript.txt")
    return p.with_suffix(".execution.transcript.txt")


def _tupleize_scheduling(row: tuple[Any, ...] | list[Any]) -> list[Any]:
    return list(row) if isinstance(row, tuple) else row


def _serialize_messenger_inbox(env: LongTermNegotiationEnv) -> list[dict[str, Any]]:
    """``MessengerMixin.inbox`` 全量序列化（与 JSON 档案一致，无截断）。"""
    rows: list[dict[str, Any]] = []
    for i, (src, msg) in enumerate(getattr(env, "inbox", []) or [], start=1):
        try:
            text = msg.to_natural_language()
        except Exception:  # pragma: no cover — 极端损坏消息
            text = repr(msg)
        rows.append(
            {
                "seq": i,
                "source": str(src),
                "message_class": type(msg).__name__,
                "text": text,
            }
        )
    return rows


def _visible_history_full(ctrl: Any) -> dict[str, list[str]]:
    vh = getattr(ctrl, "visible_history", None) or {}
    return {str(k): list(v) for k, v in sorted(vh.items(), key=lambda kv: kv[0])}


def build_episode_execution_record(
    env: LongTermNegotiationEnv,
    *,
    llm_model_traces: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """汇总 ``NegotiationWorldController`` 内已有结构与 ``execution_timeline``。

    在 v1 基础上增加 ``messenger_inbox``（环境侧完整交互）与 ``visible_history_by_agent``（各视角
    可见会话行全量，与 digest 中 tail 相对；此处不截断）。

    ``llm_model_traces``：可选，为与本 episode 对齐的 **逐次 LLM 调用** 记录（通常由
    ``model_trace.load_model_trace_rows`` 从 ``model_trace_dir`` 合并而来），含各时刻完整
    ``input_values``、渲染后 prompt、首次原始输出及（若有）修复链原文。
    """
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
    payload: dict[str, Any] = {
        "schema": "sotopia.long_term_negotiation.execution_record.v1_2",
        "terminal": ctrl.terminal,
        "macro_steps_used": int(getattr(env, "last_episode_macro_steps", 0) or 0),
        "messenger_inbox": _serialize_messenger_inbox(env),
        "visible_history_by_agent": _visible_history_full(ctrl),
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
    if llm_model_traces is not None:
        payload["llm_model_traces"] = list(llm_model_traces)
    return payload


def format_episode_interaction_transcript(
    env: LongTermNegotiationEnv,
    *,
    llm_model_traces: Sequence[dict[str, Any]] | None = None,
) -> str:
    """生成 UTF-8 纯文本复盘稿：完整 inbox + 调度 / 会话 / 消息 / 动作 / 各 agent 可见历史。"""
    ctrl = env.ctrl
    lines: list[str] = []
    lines.append("LONG-TERM NEGOTIATION — EPISODE INTERACTION TRANSCRIPT")
    lines.append(f"terminal={ctrl.terminal!r}")
    lines.append(f"macro_steps_used={int(getattr(env, 'last_episode_macro_steps', 0) or 0)}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("§1 Environment messenger inbox (full chronological log, no truncation)")
    lines.append("=" * 80)
    for row in _serialize_messenger_inbox(env):
        lines.append(f"[{row['seq']:04d}] source={row['source']} class={row['message_class']}")
        for part in str(row["text"]).splitlines():
            lines.append(f"    {part}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("§2 Scheduling log (day, slot, agent, natural_language)")
    lines.append("=" * 80)
    for t in getattr(ctrl, "scheduling_log", []) or []:
        if isinstance(t, (tuple, list)) and len(t) >= 4:
            lines.append(f"day={t[0]} slot={t[1]} | {t[2]}: {t[3]}")
        else:
            lines.append(repr(t))
    lines.append("")

    lines.append("=" * 80)
    lines.append("§3 Session / slot structured log (session_log entries)")
    lines.append("=" * 80)
    for entry in getattr(ctrl, "session_log", []) or []:
        lines.append(json.dumps(entry, ensure_ascii=False, default=str))
    lines.append("")

    lines.append("=" * 80)
    lines.append("§4 In-session natural-language lines (message_log)")
    lines.append("=" * 80)
    for row in getattr(ctrl, "message_log", []) or []:
        lines.append(json.dumps(row, ensure_ascii=False, default=str))
    lines.append("")

    lines.append("=" * 80)
    lines.append("§5 Formal / control action audit (action_log, full)")
    lines.append("=" * 80)
    for row in getattr(ctrl, "action_log", []) or []:
        lines.append(json.dumps(row, ensure_ascii=False, default=str))
    lines.append("")

    lines.append("=" * 80)
    lines.append("§6 Per-agent visible session transcript (full visible_history)")
    lines.append("=" * 80)
    for agent, hist in sorted(_visible_history_full(ctrl).items(), key=lambda kv: kv[0]):
        lines.append(f"## {agent}")
        for j, hline in enumerate(hist, start=1):
            lines.append(f"  {j:04d}  {hline}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("§7 Execution timeline (human-readable global events)")
    lines.append("=" * 80)
    for ev in getattr(ctrl, "execution_timeline", []) or []:
        lines.append(json.dumps(ev, ensure_ascii=False, default=str))

    if llm_model_traces:
        lines.append("")
        lines.append("=" * 80)
        lines.append(
            "§8 LLM calls (per-step full inputs + raw outputs; also duplicated in .execution.json "
            "under llm_model_traces)"
        )
        lines.append("=" * 80)
        for row in llm_model_traces:
            lines.append(
                json.dumps(
                    {
                        "step_index": row.get("step_index"),
                        "step_kind": row.get("step_kind"),
                        "trace_agent": row.get("trace_agent"),
                        "model_name": row.get("model_name"),
                        "generation_meta": row.get("generation_meta"),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
            fp = row.get("full_rendered_prompt")
            if fp is None and isinstance(row.get("messages"), list) and row["messages"]:
                m0 = row["messages"][0]
                if isinstance(m0, dict):
                    fp = m0.get("content")
            if fp:
                lines.append("--- full_rendered_prompt (exact user message) ---")
                lines.append(str(fp))
            iv = row.get("input_values")
            if iv is not None:
                lines.append("--- input_values (template fields at this call) ---")
                lines.append(json.dumps(iv, ensure_ascii=False, default=str))
            raw = row.get("raw_model_content")
            if raw is not None:
                lines.append("--- raw_model_content (first completion, unparsed) ---")
                lines.append(str(raw))
            r2 = row.get("raw_model_content_repaired")
            if r2 is not None:
                lines.append("--- raw_model_content_repaired (bad-output repair model, unparsed) ---")
                lines.append(str(r2))
            lines.append("--- parsed (post-parser structured snapshot) ---")
            lines.append(json.dumps(row.get("parsed"), ensure_ascii=False, default=str))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_episode_execution_record(
    env: LongTermNegotiationEnv,
    path: Path | str,
    *,
    write_transcript: bool = True,
    model_trace_dir: Path | str | None = None,
    model_trace_stem: str | None = None,
) -> Path:
    """写入 UTF-8 JSON（缩进 2）。

    当 ``write_transcript=True``（默认）时，另写 ``transcript_path_for_execution_json(path)``，
    内容为 ``format_episode_interaction_transcript``，保留 **完整** 环境侧交互历史且便于直接阅读。

    ``model_trace_dir`` + ``model_trace_stem`` 与 ``begin_episode_trace`` 所用 stem 一致时，将对应
    ``*.jsonl`` 合并入 ``llm_model_traces`` 字段并写入 transcript §8。
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    llm_rows: list[dict[str, Any]] | None = None
    if model_trace_dir is not None and model_trace_stem:
        try:
            from .model_trace import load_model_trace_rows

            llm_rows = load_model_trace_rows(model_trace_dir, model_trace_stem)
        except Exception:
            llm_rows = None
    payload = build_episode_execution_record(env, llm_model_traces=llm_rows)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if write_transcript:
        tp = transcript_path_for_execution_json(p)
        tp.write_text(
            format_episode_interaction_transcript(env, llm_model_traces=llm_rows),
            encoding="utf-8",
        )
    return p
