"""长期谈判评测的 **异步批量调度**（参考 ``benchmark.run_async_benchmark_in_batch`` 的 batch/concurrency 思路）。

不依赖 EpisodeLog / Redis；每条任务调用 ``run_llm_negotiation_episode_evaluation``，结果可为 JSONL。

**本模块在整条评测链中的位置**

夹在 CLI（``cli/benchmark/negotiation_batch.negotiation_batch``）与单次评测核心
（``llm_evaluation.run_llm_negotiation_episode_evaluation``）之间。

对单个 (seq, agent_model) 作业的 **顺序与作用**：

1. ``uniform_negotiation_model_dict`` — 构造 ``model_dict``（参与者模型 + ``env`` 键上的评测模型）。
2. ``run_llm_negotiation_episode_evaluation``（``llm_evaluation.py``）— 跑环境与可选终局 LLM 评分，
   返回 ``LongTermNegotiationEvalResult``。
3. ``build_eval_record`` — 把结果收成一条实验记录（experiment_tag / terminal / metrics 等）。
4. … 外层 ``negotiation_eval_record_to_jsonable`` — 整条 batch 末尾对每条记录再走一遍，便于 ``json.dumps``。

**日志与可读输出**（参见 ``eval_logging.py``）：JSONL ``-o`` 为结构化结果；``tqdm`` 为 stderr 进度；
``sotopia.negotiation.batch`` logger 在每集输出 ``episode_start`` / ``episode_done`` 单行摘要（需 CLI
``--print-logs`` 或 ``--log-file`` 将 root level 调至 INFO）。摘要中含 ``num_participants``（2/3/4）。

**异步并发**

- ``run_long_term_negotiation_eval_batch_async`` 为每条作业建协程，经 ``asyncio_gather_bounded``
  用信号量施加 ``batch_size`` 上限。
- ``run_long_term_negotiation_eval_batch`` 只是 ``asyncio.run`` 的同步封装，便于脚本与非 async  CLI。
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Awaitable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .llm_evaluation import LongTermNegotiationEvalResult, run_llm_negotiation_episode_evaluation
from .eval_logging import (
    episode_done_line,
    episode_start_line,
    get_negotiation_batch_logger,
    negotiation_artifact_leaf_dir,
)
from .scenario_loader import load_negotiation_scenario_from_environment_profile_pk
from .types import NegotiationTimelineParams


def uniform_negotiation_model_dict(
    agent_model: str,
    evaluator_model: str,
    *,
    quartet: bool | None = None,
    num_participants: int | None = None,
) -> dict[str, str]:
    """与 ``minimalist_demo`` / ``llm_evaluation`` 一致：单方模型复用到 ``agent1``…``agentN``。"""
    if num_participants is None:
        n = 4 if quartet else 2
    else:
        n = num_participants
    if n < 2 or n > 4:
        raise ValueError(f"num_participants must be 2..4, got {n}")
    md: dict[str, str] = {"env": evaluator_model}
    for i in range(1, n + 1):
        md[f"agent{i}"] = agent_model
    return md


def negotiation_eval_record_to_jsonable(record: dict[str, Any]) -> dict[str, Any]:
    """确保可 ``json.dumps``（嵌套 tuple 等转为 JSON 原生类型）。"""
    import json as _json

    def default(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="python")
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(repr(o))

    return _json.loads(_json.dumps(record, default=default))


def build_eval_record(
    *,
    experiment_tag: str,
    seq: int,
    agent_model: str,
    evaluator_model: str,
    quartet: bool,
    num_participants: int,
    result: LongTermNegotiationEvalResult,
    environment_profile_pk: str | None = None,
    scenario_codename: str | None = None,
    negotiation_run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    llm_dump: Any = None
    if result.llm_aggregate is not None:
        llm_dump = result.llm_aggregate.model_dump(mode="python")
    row = {
        "experiment_tag": experiment_tag,
        "seq": seq,
        "agent_model": agent_model,
        "evaluator_model": evaluator_model,
        "quartet": quartet,
        "num_participants": num_participants,
        "terminal": result.terminal,
        "rule_metrics": result.rule_metrics,
        "llm_aggregate": llm_dump,
    }
    if environment_profile_pk:
        row["environment_profile_pk"] = environment_profile_pk
    if scenario_codename:
        row["scenario_codename"] = scenario_codename
    if negotiation_run_config is not None:
        row["negotiation_run_config"] = negotiation_run_config
    return row


async def asyncio_gather_bounded(
    coroutines: Sequence[Awaitable[Any]],
    limit: int,
    *,
    progress_desc: str = "negotiation eval batch",
) -> list[Any]:
    """对标 ``benchmark`` 里的 ``batch_size``：同一时刻至多 ``limit`` 个 episode 在执行。"""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    sem = asyncio.Semaphore(limit)

    async def wrap(coro: Awaitable[Any]) -> Any:
        async with sem:
            return await coro

    barred = [wrap(c) for c in coroutines]
    n_tot = len(barred)
    pbar = tqdm(
        total=n_tot,
        desc=progress_desc[:100],
        smoothing=0.05,
        dynamic_ncols=True,
        ascii=False,
        bar_format="{desc}: |{bar:18}| {n}/{total} [{elapsed}<{remaining}] {rate_fmt}",
        file=sys.stderr,
    )

    async def track(coro: Awaitable[Any]) -> Any:
        try:
            return await coro
        finally:
            pbar.update(1)

    try:
        return await asyncio.gather(*(track(c) for c in barred))
    finally:
        pbar.close()


async def run_long_term_negotiation_eval_batch_async(
    *,
    agent_models: Sequence[str],
    evaluator_model: str,
    quartet: bool = False,
    repeats_per_model: int = 1,
    batch_size: int = 3,
    params: NegotiationTimelineParams | None = None,
    scenario_environment_pks: Sequence[str] | None = None,
    max_macro_steps: int = 4000,
    run_terminal_llm_eval: bool = True,
    experiment_tag_base: str = "negotiation_eval_batch",
    run_id: str | None = None,
    history_max_action_log: int | None = 500,
    num_participants: int | None = None,
    model_trace_dir: Path | str | None = None,
    execution_trace_dir: Path | str | None = None,
    nest_trace_dirs_by_model_time: bool = False,
    run_timestamp: str | None = None,
    negotiation_run_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """对多个 ``agent_models`` 各重复 ``repeats_per_model`` 次，并发上限 ``batch_size``。

    ``scenario_environment_pks`` 非空时：按 (场景 pk × agent_model × repeats) 展开；每场从
    ``EnvironmentProfile.game_metadata`` 加载 ``NegotiationTimelineParams``、``quartet`` 与
    ``num_participants``（缺省由 quartet 推断）。显式 ``num_participants`` 覆盖单场人数（含场景模式）。
    此模式下 ``params`` / ``quartet`` 主参数不再用于单场时间轴与人数（仍可为无场景路径保留兼容）。

    返回已序列化友好的 ``dict`` 列表（可直接写 JSONL）。

    ``model_trace_dir``：若指定，每场 ``run_llm_negotiation_episode_evaluation`` 将原始 LLM
    completion 与解析结果写入该目录下 **按 agent 分文件** 的 JSONL，并另写终局评测文件（文件名由
    episode 的 ``tag`` 派生，见 ``model_trace``）。

    ``nest_trace_dirs_by_model_time``（默认 ``False``）：为 ``True`` 时，在
    ``model_trace_dir`` / ``execution_trace_dir`` 下追加
    ``{sanitized_agent_model}/{run_timestamp}/``，便于按测试模型名与时间戳分子目录归档；
    ``run_timestamp`` 缺省为批量开始时生成的 ``YYYYMMDD_HHMMSS``（整条 batch 共用同一时间戳目录）。

    ``execution_trace_dir``：若指定，每场结束后写入 **全局执行档案** ``*.execution.json``
    （合同里程碑、时间线、完整 ``action_log``、**完整** ``messenger_inbox`` 等），文件名由 ``tag`` 派生；
    同时默认写入配对的 ``*.execution.transcript.txt``（纯文本复盘稿，无截断）。

    ``negotiation_run_config``：与 ``negotiation-batch --run-config`` 相同；写入每条 JSONL
    记录的 ``negotiation_run_config`` 字段以便复现。
    """
    if repeats_per_model < 1:
        raise ValueError("repeats_per_model must be >= 1")
    rid = run_id or uuid.uuid4().hex[:12]
    artifact_stamp = (run_timestamp or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    scenarios = list(scenario_environment_pks or ())
    jobs: list[tuple[int, str, str | None]]
    seq = 0
    jobs = []
    if scenarios:
        for env_pk in scenarios:
            for am in agent_models:
                for _ in range(repeats_per_model):
                    jobs.append((seq, am, env_pk))
                    seq += 1
    else:
        for am in agent_models:
            for _ in range(repeats_per_model):
                jobs.append((seq, am, None))
                seq += 1

    async def one(seq_i: int, agent_model: str, env_pk: str | None) -> dict[str, Any]:
        log = get_negotiation_batch_logger()
        quartet_j = quartet
        n_eff: int
        params_j = params
        env_meta_pk: str | None = None
        codename_display: str | None = None
        if env_pk is not None:
            sc = load_negotiation_scenario_from_environment_profile_pk(env_pk)
            quartet_j = sc.quartet
            n_eff = num_participants if num_participants is not None else sc.num_participants
            params_j = None
            env_meta_pk = sc.environment_profile_pk
            codename_display = sc.codename or None
        else:
            n_eff = num_participants if num_participants is not None else (4 if quartet else 2)

        md = uniform_negotiation_model_dict(
            agent_model, evaluator_model, num_participants=n_eff
        )
        tag = f"{experiment_tag_base}_{rid}_{seq_i}"
        sl = episode_start_line(
            seq=seq_i,
            agent_model=agent_model,
            env_pk=env_pk,
            quartet=quartet_j,
            num_participants=n_eff,
            tag=tag,
        )
        log.info("── %s", sl)

        try:
            trace_kw: dict[str, Any] = {}

            def _resolved_trace_base(base: Path | str | None) -> Path | None:
                if base is None:
                    return None
                pb = Path(base)
                if nest_trace_dirs_by_model_time:
                    resolved = pb.resolve() / negotiation_artifact_leaf_dir(agent_model, artifact_stamp)
                else:
                    resolved = pb.resolve()
                resolved.mkdir(parents=True, exist_ok=True)
                return resolved

            resolved_mt = _resolved_trace_base(model_trace_dir)
            resolved_et = _resolved_trace_base(execution_trace_dir)
            if resolved_mt is not None:
                trace_kw["model_trace_dir"] = resolved_mt
                trace_kw["model_trace_tag"] = tag
            if resolved_et is not None:
                trace_kw["execution_trace_dir"] = resolved_et
                trace_kw["execution_trace_tag"] = tag
            res = await run_llm_negotiation_episode_evaluation(
                md,
                quartet=quartet_j,
                num_participants=num_participants,
                params=params_j,
                environment_profile_pk=env_pk,
                max_macro_steps=max_macro_steps,
                run_terminal_llm_eval=run_terminal_llm_eval,
                history_max_action_log=history_max_action_log,
                negotiation_run_config=negotiation_run_config,
                **trace_kw,
            )
        except Exception:
            log.exception(
                "── episode_fail seq=%s agent_model=%r env_pk=%r tag=%r",
                seq_i,
                agent_model,
                env_pk,
                tag,
            )
            raise

        dl = episode_done_line(
            seq=seq_i,
            terminal=str(res.terminal),
            quartet=quartet_j,
            num_participants=n_eff,
            agent_model=agent_model,
            env_pk=env_meta_pk or env_pk,
            scenario_codename=codename_display,
            rule_metrics=dict(res.rule_metrics),
            scored_llm=res.llm_aggregate is not None,
            tag=tag,
        )
        log.info("── %s", dl)

        return build_eval_record(
            experiment_tag=tag,
            seq=seq_i,
            agent_model=agent_model,
            evaluator_model=evaluator_model,
            quartet=quartet_j,
            num_participants=n_eff,
            result=res,
            environment_profile_pk=env_meta_pk,
            scenario_codename=codename_display,
            negotiation_run_config=negotiation_run_config,
        )

    coros = [one(s, am, pk) for s, am, pk in jobs]
    prog = f"LTR batch jobs={len(coros)} concurrency<={batch_size}"
    raw = await asyncio_gather_bounded(coros, batch_size, progress_desc=prog)
    return [negotiation_eval_record_to_jsonable(r) for r in raw]


def run_long_term_negotiation_eval_batch(
    *,
    agent_models: Sequence[str],
    evaluator_model: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """同步入口：内部 ``asyncio.run``。"""
    return asyncio.run(
        run_long_term_negotiation_eval_batch_async(
            agent_models=agent_models,
            evaluator_model=evaluator_model,
            **kwargs,
        )
    )


__all__ = [
    "asyncio_gather_bounded",
    "build_eval_record",
    "negotiation_eval_record_to_jsonable",
    "run_long_term_negotiation_eval_batch",
    "run_long_term_negotiation_eval_batch_async",
    "uniform_negotiation_model_dict",
]
