"""谈判评测日志：统一 logger 名称与episode 单行摘要格式。

**输出到哪里去**

1. **结构化结果**：``negotiation-batch -o xxx.jsonl`` —— 每行一条 episode 的机器可读记录；不是传统 log。
2. **进度条**：``tqdm`` 写到 stderr（``run_long_term_negotiation_eval_batch_async``）。
3. **控制台 / 文件日志**：logger ``sotopia.negotiation`` 及以下子 logger；CLI ``--print-logs`` 用 Rich，
   ``--log-file`` 追加 **纯文本** UTF-8（无 ANSI）。
4. **模型 trace**（``--model-trace-dir``）：每场 episode 下按 agent 多份 ``*.jsonl`` + 一份终局
   ``*_terminal_eval.jsonl``，见 ``model_trace`` 模块说明。
5. **第三方**：``sotopia.generation_utils.generate`` / LiteLLM / litellm 等在 DEBUG/INFO 时可能额外刷屏；
   需要时可 ``export LITELLM_LOG=WARNING``（视版本而定）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER_NAME_ROOT = "sotopia.negotiation"
LOGGER_NAME_BATCH = f"{LOGGER_NAME_ROOT}.batch"


def configure_negotiation_cli_logging(
    *,
    verbose_console: bool,
    log_file: Path | None,
) -> None:
    """为 ``negotiation-batch`` 配置 root logging：可选 Rich 控制台 + 可选纯文本文件追加。"""
    from rich.logging import RichHandler

    handlers: list[logging.Handler] = []
    if verbose_console:
        handlers.append(RichHandler(rich_tracebacks=True, show_path=False, markup=False))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(fh)
    if not handlers:
        handlers.append(logging.NullHandler())

    wants_detail = verbose_console or log_file is not None
    if wants_detail:
        logging.getLogger(LOGGER_NAME_ROOT).setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO if wants_detail else logging.WARNING,
        handlers=handlers,
        force=True,
    )


def get_negotiation_batch_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME_BATCH)


def compact_rule_metrics(metrics: dict[str, Any], *, max_pairs: int = 12) -> str:
    """单行规则指标缩写，过长时截断。"""
    pairs: list[tuple[str, Any]] = []
    for i, key in enumerate(sorted(metrics)):
        if i >= max_pairs:
            pairs.append(("...", "..."))
            break
        v = metrics[key]
        try:
            fv = float(v)
            pairs.append((key, f"{fv:.6g}"))
        except (TypeError, ValueError):
            pairs.append((key, str(v)[:32]))
    return "; ".join(f"{k}={val}" for k, val in pairs)


def episode_start_line(
    *,
    seq: int,
    agent_model: str,
    env_pk: str | None,
    quartet: bool,
    num_participants: int,
    tag: str,
) -> str:
    return (
        f"episode_start seq={seq} agent_model={agent_model!r} "
        f"env_pk={env_pk or '-'} num_participants={num_participants} quartet={quartet} "
        f"experiment_tag_suffix={tag!r}"
    )


def episode_done_line(
    *,
    seq: int,
    terminal: str,
    quartet: bool,
    num_participants: int,
    agent_model: str,
    env_pk: str | None,
    scenario_codename: str | None,
    rule_metrics: dict[str, Any],
    scored_llm: bool,
    tag: str,
) -> str:
    scm = scenario_codename or "-"
    fs_score = rule_metrics.get("negotiation_final_state_score")
    try:
        fs_score_str = f"{float(fs_score):.4f}" if fs_score is not None else "-"
    except (TypeError, ValueError):
        fs_score_str = "-"
    return (
        f"episode_done seq={seq} terminal={terminal!r} num_participants={num_participants} quartet={quartet} "
        f"agent_model={agent_model!r} env_pk={env_pk or '-'} scenario_codename={scm!r} "
        f"llm_terminal_scoring={'yes' if scored_llm else 'no'} experiment_tag_suffix={tag!r} "
        f"final_state_score={fs_score_str} "
        f"rule_metrics{{{compact_rule_metrics(rule_metrics)}}}"
    )


__all__ = [
    "LOGGER_NAME_BATCH",
    "LOGGER_NAME_ROOT",
    "compact_rule_metrics",
    "configure_negotiation_cli_logging",
    "episode_done_line",
    "episode_start_line",
    "get_negotiation_batch_logger",
]
