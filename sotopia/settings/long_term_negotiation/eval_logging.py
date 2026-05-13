"""谈判评测日志：统一 logger 名称与episode 单行摘要格式。

**输出到哪里去**

1. **结构化结果**：``negotiation-batch -o xxx.jsonl`` —— 每行一条 episode 的机器可读记录；不是传统 log。
2. **进度条**：``tqdm`` 写到 stderr（``run_long_term_negotiation_eval_batch_async``）。
3. **控制台 / 文件日志**：logger ``sotopia.negotiation`` 及以下子 logger；CLI ``--print-logs`` 用 Rich，
   ``--log-file`` 追加 **纯文本** UTF-8（无 ANSI）。
4. **模型 trace**（``--model-trace-dir`` 或 ``--artifact-root``）：每场 episode 下按 agent 多份 ``*.jsonl`` + 一份终局
   ``*_terminal_eval.jsonl``，见 ``model_trace`` 模块说明。批量 CLI 默认写入
   ``{根目录}/{测试模型名}/{时间戳}/``（``--trace-flat`` 可关闭嵌套）。
5. **执行档案**（``--execution-trace-dir`` 或 ``--artifact-root``）：每场 ``*.execution.json`` + 同 stem 的 ``*.execution.transcript.txt``
   （全量交互纯文本），以及默认 **每角色一份** ``{tag}_{agent}.agent_episode.json``（该角色执行轨迹子集 + 其全部 LLM 输入输出合一），见 ``episode_execution_record``。批量时默认与 trace 相同的
   ``{根目录}/{测试模型名}/{时间戳}/`` 嵌套。
6. **统一根目录**（``negotiation-batch --artifact-root``）：将 execution、model trace、默认
   ``negotiation_batch.log`` 的根目录设为同一路径；**单模型**时三者落在同一叶子文件夹。
7. **第三方**：``sotopia.generation_utils.generate`` / LiteLLM / litellm 等在 DEBUG/INFO 时可能额外刷屏；
   需要时可 ``export LITELLM_LOG=WARNING``（视版本而定）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

LOGGER_NAME_ROOT = "sotopia.negotiation"
LOGGER_NAME_BATCH = f"{LOGGER_NAME_ROOT}.batch"


def sanitized_llm_model_name_for_path(model: str) -> str:
    """将 LiteLLM 路由键等转为可作目录名的短段（避免非法路径字符）。"""
    base = (model or "unknown").strip() or "unknown"
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r"[^-._a-zA-Z0-9]+", "_", base)
    base = base.strip("._") or "unknown"
    if len(base) > 120:
        base = base[:120]
    return base


def negotiation_artifact_leaf_dir(agent_model: str, run_ts: str) -> Path:
    """批量评测产物子路径：``{模型目录名}/{时间戳}/``（``run_ts`` 形如 ``YYYYMMDD_HHMMSS``）。"""
    return Path(sanitized_llm_model_name_for_path(agent_model)) / str(run_ts).strip()


def negotiation_default_batch_log_file(agent_models: Sequence[str], run_ts: str) -> Path:
    """未指定 ``--log-file`` 时：``logs/{模型目录}/{时间戳}/negotiation_batch.log``。

    多模型时根目录为 ``首模型_sanitized__Nmodels``，避免不同模型混在同一日志目录。
    """
    models = [str(m) for m in agent_models if str(m).strip()]
    if not models:
        folder = "unknown"
    elif len(models) == 1:
        folder = sanitized_llm_model_name_for_path(models[0])
    else:
        folder = f"{sanitized_llm_model_name_for_path(models[0])}__{len(models)}models"
    return Path("logs") / folder / str(run_ts).strip() / "negotiation_batch.log"


def negotiation_batch_log_under_artifact_root(
    artifact_root: Path | str,
    agent_models: Sequence[str],
    run_ts: str,
) -> Path:
    """与 ``negotiation_default_batch_log_file`` 相同的子路径规则，但根目录为 ``artifact_root``。

    与 ``negotiation-batch --artifact-root`` 配合：单模型时与 execution / model trace 写入同一叶子目录。
    """
    models = [str(m) for m in agent_models if str(m).strip()]
    if not models:
        folder = "unknown"
    elif len(models) == 1:
        folder = sanitized_llm_model_name_for_path(models[0])
    else:
        folder = f"{sanitized_llm_model_name_for_path(models[0])}__{len(models)}models"
    return Path(artifact_root).resolve() / folder / str(run_ts).strip() / "negotiation_batch.log"


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
    "negotiation_artifact_leaf_dir",
    "negotiation_batch_log_under_artifact_root",
    "negotiation_default_batch_log_file",
    "sanitized_llm_model_name_for_path",
]
