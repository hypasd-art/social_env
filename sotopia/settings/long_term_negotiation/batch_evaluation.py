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

**异步并发**

- ``run_long_term_negotiation_eval_batch_async`` 为每条作业建协程，经 ``asyncio_gather_bounded``
  用信号量施加 ``batch_size`` 上限。
- ``run_long_term_negotiation_eval_batch`` 只是 ``asyncio.run`` 的同步封装，便于脚本与非 async  CLI。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Sequence
from typing import Any

from tqdm import tqdm

from .llm_evaluation import LongTermNegotiationEvalResult, run_llm_negotiation_episode_evaluation
from .scenario_loader import load_negotiation_scenario_from_environment_profile_pk
from .types import NegotiationTimelineParams


def uniform_negotiation_model_dict(agent_model: str, evaluator_model: str, *, quartet: bool) -> dict[str, str]:
    """与 ``minimalist_demo`` / ``llm_evaluation`` 一致：单方模型复用到 ``agent1``…``agentN``。"""
    md: dict[str, str] = {
        "env": evaluator_model,
        "agent1": agent_model,
        "agent2": agent_model,
    }
    if quartet:
        md["agent3"] = agent_model
        md["agent4"] = agent_model
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
    result: LongTermNegotiationEvalResult,
    environment_profile_pk: str | None = None,
    scenario_codename: str | None = None,
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
        "terminal": result.terminal,
        "rule_metrics": result.rule_metrics,
        "llm_aggregate": llm_dump,
    }
    if environment_profile_pk:
        row["environment_profile_pk"] = environment_profile_pk
    if scenario_codename:
        row["scenario_codename"] = scenario_codename
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
    pbar = tqdm(total=len(barred), desc=progress_desc)

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
) -> list[dict[str, Any]]:
    """对多个 ``agent_models`` 各重复 ``repeats_per_model`` 次，并发上限 ``batch_size``。

    ``scenario_environment_pks`` 非空时：按 (场景 pk × agent_model × repeats) 展开；每场从
    ``EnvironmentProfile.game_metadata`` 加载 ``NegotiationTimelineParams`` 与 ``quartet``。
    此模式下 ``params`` / ``quartet`` 主参数不再用于单场（仍可为默认路径保留兼容）。

    返回已序列化友好的 ``dict`` 列表（可直接写 JSONL）。
    """
    if repeats_per_model < 1:
        raise ValueError("repeats_per_model must be >= 1")
    rid = run_id or uuid.uuid4().hex[:12]

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
        quartet_j = quartet
        params_j = params
        env_meta_pk: str | None = None
        codename_display: str | None = None
        if env_pk is not None:
            sc = load_negotiation_scenario_from_environment_profile_pk(env_pk)
            quartet_j = sc.quartet
            params_j = None
            env_meta_pk = sc.environment_profile_pk
            codename_display = sc.codename or None

        md = uniform_negotiation_model_dict(agent_model, evaluator_model, quartet=quartet_j)
        tag = f"{experiment_tag_base}_{rid}_{seq_i}"
        res = await run_llm_negotiation_episode_evaluation(
            md,
            quartet=False if env_pk else quartet_j,
            params=params_j,
            environment_profile_pk=env_pk,
            max_macro_steps=max_macro_steps,
            run_terminal_llm_eval=run_terminal_llm_eval,
            history_max_action_log=history_max_action_log,
        )
        return build_eval_record(
            experiment_tag=tag,
            seq=seq_i,
            agent_model=agent_model,
            evaluator_model=evaluator_model,
            quartet=quartet_j,
            result=res,
            environment_profile_pk=env_meta_pk,
            scenario_codename=codename_display,
        )

    coros = [one(s, am, pk) for s, am, pk in jobs]
    raw = await asyncio_gather_bounded(coros, batch_size, progress_desc="long_term_negotiation batch")
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
