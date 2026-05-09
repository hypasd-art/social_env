# 长期谈判：大模型参与者 + 终局 LLM 评测（``model_dict`` 约定对齐 ``minimalist_demo.py``）。
#
# 建议在仓库 ``social_env/.env`` 中配置 ``OPENAI_API_KEY``（及可选 ``OPENAI_API_BASE``），
# 并可选 ``NEGOTIATION_AGENT_MODEL`` / ``NEGOTIATION_EVAL_MODEL``；脚本启动时会 ``load_dotenv``。
#
#   conda activate social_env && cd social_env && PYTHONPATH=. python examples/long_term_negotiation_llm_eval_demo.py
#
# 四方谈判::
#
#   PYTHONPATH=. python examples/long_term_negotiation_llm_eval_demo.py --quartet

from __future__ import annotations

import os
from pathlib import Path


def _load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo = Path(__file__).resolve().parent.parent
    load_dotenv(repo / ".env", override=False)


_load_repo_dotenv()

import argparse
import asyncio
import json
import logging
import sys

from rich.logging import RichHandler

from sotopia.settings import NegotiationTimelineParams
from sotopia.settings.long_term_negotiation.llm_evaluation import (
    run_llm_negotiation_episode_evaluation,
)


async def _main(quartet: bool, skip_llm_scoring: bool, *, max_macro_steps: int) -> None:
    agent_model = os.getenv("NEGOTIATION_AGENT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    eval_model = os.getenv("NEGOTIATION_EVAL_MODEL", agent_model).strip() or agent_model

    model_dict = {
        "env": eval_model,
        "agent1": agent_model,
        "agent2": agent_model,
    }
    if quartet:
        model_dict["agent3"] = agent_model
        model_dict["agent4"] = agent_model

    params = NegotiationTimelineParams(
        D=8,
        s_max_per_day=2,
        max_session_rounds=32,
        max_total_turns_per_session=64,
    )

    print(
        "[llm_demo] OPENAI_API_BASE=",
        os.getenv("OPENAI_API_BASE", ""),
        "| agent_model=",
        agent_model,
        "| eval_model=",
        eval_model,
        "| quartet=",
        quartet,
        "| skip_llm_scoring=",
        skip_llm_scoring,
        flush=True,
    )

    result = await run_llm_negotiation_episode_evaluation(
        model_dict,
        quartet=quartet,
        params=params,
        max_macro_steps=max_macro_steps,
        run_terminal_llm_eval=not skip_llm_scoring,
    )
    print("terminal:", result.terminal)
    print("rule_metrics:", json.dumps(result.rule_metrics, indent=2, sort_keys=True))
    if result.llm_aggregate is not None:
        agg = result.llm_aggregate
        print("llm p1_rate:", agg.p1_rate)
        print("llm p2_rate:", agg.p2_rate)
        print("llm comments (excerpt):", (agg.comments or "")[:800])


if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    ap = argparse.ArgumentParser(description="Long-term negotiation LLM eval (agents + optional terminal LLM scoring)")
    ap.add_argument("--quartet", action="store_true", help="strict design roster (4 agents); requires agent3/agent4 in code path")
    ap.add_argument(
        "--skip-llm-scoring",
        action="store_true",
        help="only run LLM agents for the episode; skip EpisodeLLMEvaluator at the end",
    )
    ap.add_argument(
        "--max-macro-steps",
        type=int,
        default=3500,
        help="macro step ceiling for run_episode_async (lower for quicker smoke)",
    )
    args = ap.parse_args()
    try:
        asyncio.run(
            _main(
                quartet=args.quartet,
                skip_llm_scoring=args.skip_llm_scoring,
                max_macro_steps=args.max_macro_steps,
            )
        )
    except Exception as exc:  # pragma: no cover
        print(exc, file=sys.stderr)
        sys.exit(1)
