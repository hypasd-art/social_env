"""长期谈判评测期间将 **每次 LLM 调用** 的提示与原始输出追加到 JSONL。

由 ``run_llm_negotiation_episode_evaluation(..., model_trace_dir=...)`` 激活；``agenerate`` /
``agenerate_action`` / ``agenerate_goal`` 等经 ``generation_utils.generate.agenerate`` 的路径
会自动写入 ``step_kind=agenerate`` 行。每行含：

* ``messages`` / ``full_rendered_prompt``：该次调用送入模型的完整 user 文本；
* ``input_values``：模板变量全量（含 ``history``、``goal`` 等），便于对齐「当时输入」；
* ``raw_model_content``：主模型 **首次** API 返回正文（解析前）；若有坏输出修复链则另有
  ``raw_model_content_repaired``；
* ``parsed``：解析器产出的结构化结果。

终局 ``EpisodeLLMEvaluator`` 由 ``record_terminal_eval_step`` 单独写 ``step_kind=terminal_eval``。

``load_model_trace_rows`` 供 ``episode_execution_record`` 合并进 ``*.execution.json``。

**分文件策略**：``record_generation_step`` 按 ``input_values["agent"]`` 写入
``{stem}_{<agent>}.jsonl``；无 ``agent`` 时写入 ``{stem}_no_agent.jsonl``。终局评测写入
``{stem}_terminal_eval.jsonl``。同一场 episode 内所有文件共用单调递增的 ``step_index``，
便于跨文件按序合并。

并发：同一条 episode 内日程阶段可能并行多次 ``agenerate``，对 **本 episode 的全部 trace 文件**
使用同一把 ``threading.Lock``，保证行级追加不交错损坏。
"""

from __future__ import annotations

import json
import re
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "begin_episode_trace",
    "end_episode_trace",
    "load_model_trace_rows",
    "record_generation_step",
    "record_terminal_eval_step",
    "safe_trace_filename",
    "sanitize_trace_segment",
]


@dataclass
class _TraceState:
    trace_dir: Path
    stem: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    step_seq: list[int] = field(default_factory=lambda: [0])


_ctx: ContextVar[_TraceState | None] = ContextVar("negotiation_model_trace", default=None)


def safe_trace_filename(tag: str) -> str:
    """将 experiment_tag 等转为安全、较短的 ``*.jsonl`` 文件名。"""
    base = (tag or "negotiation_episode").strip()
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r"[^-._a-zA-Z0-9]", "_", base)
    base = base.strip("._") or "negotiation_episode"
    if len(base) > 180:
        base = base[:180]
    return f"{base}.jsonl"


def sanitize_trace_segment(s: str) -> str:
    """将 agent 名等转为可作文件名一段的安全字符串。"""
    base = (s or "").strip() or "no_agent"
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r"[^-._a-zA-Z0-9]", "_", base)
    base = base.strip("._") or "no_agent"
    if len(base) > 80:
        base = base[:80]
    return base


def _agent_bucket(input_values: dict[str, Any] | None) -> str:
    if not input_values:
        return "no_agent"
    raw = input_values.get("agent")
    if raw is None:
        return "no_agent"
    t = str(raw).strip()
    return t if t else "no_agent"


def _trace_path_for_bucket(state: _TraceState, bucket: str) -> Path:
    safe = sanitize_trace_segment(bucket)
    # 与 ``{stem}_terminal_eval.jsonl`` 终局文件区分，避免角色名撞车。
    if safe == "terminal_eval":
        safe = "role_terminal_eval"
    return state.trace_dir / f"{state.stem}_{safe}.jsonl"


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="python")
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return str(obj)


def begin_episode_trace(path: Path) -> Token:
    """开始一条 episode 的追踪。

    ``path`` 为 **逻辑上的** 主文件名（如 ``.../my_episode_tag.jsonl``）；实际写入会按 agent
    拆成 ``my_episode_tag_firm_a.jsonl`` 等多文件，终局评测为 ``my_episode_tag_terminal_eval.jsonl``。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not str(path).lower().endswith(".jsonl"):
        path = path.with_suffix(".jsonl")
    path = path.resolve()
    state = _TraceState(trace_dir=path.parent, stem=path.stem)
    return _ctx.set(state)


def end_episode_trace(token: Token) -> None:
    """结束追踪，恢复上下文。"""
    _ctx.reset(token)


def load_model_trace_rows(trace_dir: Path | str, stem: str) -> list[dict[str, Any]]:
    """读取某次 episode 写入目录下的 ``{stem}_*.jsonl``，按 ``step_index`` 合并排序。"""
    root = Path(trace_dir).resolve()
    stem = str(stem or "").strip() or "negotiation_episode"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"{stem}_*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: int(r.get("step_index", 0) or 0))
    return rows


def record_generation_step(
    *,
    step_kind: str,
    model_name: str,
    messages: list[dict[str, Any]],
    raw_content: str,
    parsed: Any,
    input_values: dict[str, Any] | None = None,
    full_rendered_prompt: str | None = None,
    raw_model_content_repaired: str | None = None,
    generation_meta: dict[str, Any] | None = None,
) -> None:
    """由 ``generation_utils.generate.agenerate`` 在每次 completion 解析成功后调用。

    ``raw_content``：主模型 **首次** 返回的 ``message.content``（解析前原文）。
    ``raw_model_content_repaired``：若走 ``format_bad_output`` 修复链，则为修复模型产出的原文
    （仍未经业务解析器以外的二次改写）；否则省略。
    ``full_rendered_prompt``：模板变量替换后的完整 user 侧文本（与 ``messages[0].content`` 一致，便于单字段检索）。
    ``input_values``：写入 **全量** 模板键值（含 ``history``、``goal`` 等），便于复盘「该时刻模型见到了什么」。
    """
    state = _ctx.get()
    if state is None:
        return
    bucket = _agent_bucket(input_values)
    out_path = _trace_path_for_bucket(state, bucket)
    with state.lock:
        state.step_seq[0] += 1
        idx = state.step_seq[0]
        row: dict[str, Any] = {
            "step_index": idx,
            "step_kind": step_kind,
            "trace_agent": bucket,
            "model_name": model_name,
            "messages": _serialize(messages),
            "raw_model_content": raw_content,
            "parsed": _serialize(parsed),
        }
        if full_rendered_prompt is not None:
            row["full_rendered_prompt"] = full_rendered_prompt
        if raw_model_content_repaired is not None:
            row["raw_model_content_repaired"] = raw_model_content_repaired
        if generation_meta:
            row["generation_meta"] = _serialize(generation_meta)
        if input_values:
            row["input_values_summary"] = _serialize(
                {k: v for k, v in input_values.items() if k in ("agent", "turn_number", "goal")}
            )
            row["input_values"] = _serialize(dict(input_values))
        line = json.dumps(row, ensure_ascii=False, default=str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def record_terminal_eval_step(
    *,
    model_name: str,
    history: str,
    aggregate: Any,
) -> None:
    """终局 ``EpisodeLLMEvaluator`` 聚合结果（在 ``llm_evaluation`` 内调用）。"""
    state = _ctx.get()
    if state is None:
        return
    out_path = state.trace_dir / f"{state.stem}_terminal_eval.jsonl"
    with state.lock:
        state.step_seq[0] += 1
        idx = state.step_seq[0]
        row: dict[str, Any] = {
            "step_index": idx,
            "step_kind": "terminal_eval",
            "trace_agent": "terminal_eval",
            "model_name": model_name,
            "eval_history": history,
            "aggregate": _serialize(aggregate),
        }
        line = json.dumps(row, ensure_ascii=False, default=str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
