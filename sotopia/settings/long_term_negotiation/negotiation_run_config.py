"""长期谈判 **运行配置**（JSON）：由 ``negotiation-batch --run-config`` 加载，用于选用 Agent 变体与记忆后端。

设计目标：不把「用哪种 Agent / 是否做记忆总结」写死在业务分支里，而是由配置文件驱动，
便于 A/B 测试与复现实验。

当前支持::

    negotiation_agent: "negotiation_social_llm"  （唯一内置实现）

    memory.backend: "plain" | "summarizing"
        - ``plain``：``EpisodicMemory``（默认）
        - ``summarizing``：``SummarizingEpisodicMemory``；需 ``summary_model``（LiteLLM 路由键）
          或特殊值 ``"$env"`` 表示复用 ``model_dict["env"]``（与终局评测模型相同）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .negotiation_llm_agent import (
    NegotiationSocialLLMAgent,
    build_negotiation_social_llm_agents,
)

DEFAULT_NEGOTIATION_RUN_CONFIG: dict[str, Any] = {
    "negotiation_agent": "negotiation_social_llm",
    "memory": {"backend": "plain"},
}


def load_negotiation_run_config(path: Path | str | None) -> dict[str, Any]:
    """从 JSON 文件加载运行配置；``path is None`` 时返回内置默认。"""
    if path is None:
        return json.loads(json.dumps(DEFAULT_NEGOTIATION_RUN_CONFIG))
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"negotiation run config not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("run config root must be a JSON object")
    merged = json.loads(json.dumps(DEFAULT_NEGOTIATION_RUN_CONFIG))
    merged.update(raw)
    if isinstance(raw.get("memory"), dict):
        mem_def = dict(DEFAULT_NEGOTIATION_RUN_CONFIG.get("memory") or {})
        mem_def.update(raw["memory"])
        merged["memory"] = mem_def
    return merged


def build_negotiation_agents_from_run_config(
    model_dict: dict[str, str],
    roster: tuple[str, ...],
    run_cfg: dict[str, Any] | None,
) -> dict[str, NegotiationSocialLLMAgent]:
    """按 ``run_cfg`` 构造谈判 LLM Agent 映射（与 ``build_llm_negotiation_agents`` 对齐的返回类型）。"""
    cfg = run_cfg if run_cfg is not None else DEFAULT_NEGOTIATION_RUN_CONFIG
    agent = str(cfg.get("negotiation_agent") or "negotiation_social_llm")
    if agent != "negotiation_social_llm":
        raise ValueError(
            f"unsupported negotiation_agent {agent!r}; supported: 'negotiation_social_llm'"
        )

    mem = cfg["memory"] if isinstance(cfg.get("memory"), dict) else {}
    backend = str(mem.get("backend") or "plain").lower()

    if backend == "plain":
        return build_negotiation_social_llm_agents(model_dict, roster)

    if backend == "summarizing":
        sm = mem.get("summary_model")
        if sm == "$env":
            sm = model_dict.get("env")
        if not sm:
            raise ValueError(
                "run_config.memory.summary_model is required when backend is 'summarizing' "
                "(LiteLLM model string, or '$env' to reuse model_dict['env'])"
            )
        social_kw: dict[str, Any] = {}
        if "max_recent_chars" in mem:
            social_kw["memory_max_recent_chars"] = int(mem["max_recent_chars"])
        if "preserve_tail_lines" in mem:
            social_kw["memory_preserve_tail_lines"] = int(mem["preserve_tail_lines"])
        if "memory_max" in mem:
            social_kw["memory_max"] = int(mem["memory_max"])
        if "memory_inject_lines" in mem:
            social_kw["memory_inject_lines"] = int(mem["memory_inject_lines"])
        return build_negotiation_social_llm_agents(
            model_dict,
            roster,
            memory_summary_model=str(sm),
            social_memory_kwargs=social_kw,
        )

    raise ValueError(
        f"unsupported memory.backend {backend!r}; expected 'plain' or 'summarizing'"
    )


__all__ = [
    "DEFAULT_NEGOTIATION_RUN_CONFIG",
    "build_negotiation_agents_from_run_config",
    "load_negotiation_run_config",
]
