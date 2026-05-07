#!/usr/bin/env python
"""用 LLM 批量生成 V2 EnvironmentProfile。

流程：
1. 读 .env 里的 OPENAI_API_KEY / OPENAI_BASE_URL
2. 按 scenario_type 用预设 inspiration_prompt 调 ``agenerate_env_profile``
   生成 N 条老 ``EnvironmentProfile``（LLM 输出，结构化 parse）
3. 用 ``upgrade_environment_profile`` 升级成 V2 并落本地后端
4. 同步把老 ``EnvironmentProfile`` 也存一份（V2 与 V1 共存，老脚本仍能读）

用法
----
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_v2_with_llm.py \\
        --scenario-type negotiation --n 5 --model gpt-4o-mini

    # 试运行（只打印，不调 LLM、不落库）
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_v2_with_llm.py --scenario-type investment --dry-run

scenario_type 与对应 inspiration_prompt 的映射：
    negotiation : 二手交易、债务谈判、薪资协商
    investment  : 联合投资、股权分配、风险共担
    commons     : 公共资源（水、电、带宽）共享
    generic     : 任意社交场景（向后兼容）
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")

# 加载 .env（里面有 OPENAI_API_KEY / OPENAI_BASE_URL）
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from sotopia.benchmark_v2_data_models import (  # noqa: E402
    upgrade_environment_profile,
)
from sotopia.database import EnvironmentProfile  # noqa: E402
from sotopia.generation_utils.generate import agenerate_env_profile  # noqa: E402


INSPIRATION_PROMPTS: dict[str, list[str]] = {
    "negotiation": [
        "Two roommates negotiating who should pay extra for the broken washing machine",
        "A freelancer renegotiating an overdue payment with a long-time client",
        "Two startup co-founders splitting equity after one decides to leave early",
        "A buyer and seller bargaining a used electric scooter on Craigslist",
        "Two siblings deciding how to share the cost of caring for an aging parent",
    ],
    "investment": [
        "Two friends deciding whether to jointly invest in a small coffee shop",
        "An angel investor and founder negotiating valuation and board seats",
        "Two analysts disagreeing on whether to short a hyped AI stock",
        "Two cousins debating whether to pool money into rural farmland",
    ],
    "commons": [
        "Three farmers sharing a single irrigation well during a drought",
        "Neighbors coordinating WiFi bandwidth caps in a co-living apartment",
        "Two roommates rationing electricity during a heatwave power shortage",
    ],
    "generic": [
        "Two old friends meeting after one of them came out as queer",
        "A parent and teenager arguing over screen time after school",
        "Two strangers trapped in an elevator during a power cut",
    ],
}

DEFAULTS = {
    "negotiation": dict(scenario_type="negotiation", max_days=3, intra_day_steps=4),
    "investment": dict(scenario_type="investment", max_days=5, intra_day_steps=3),
    "commons": dict(scenario_type="commons", max_days=7, intra_day_steps=2),
    "generic": dict(scenario_type="generic", max_days=1, intra_day_steps=6),
}


async def gen_one(prompt: str, model: str) -> EnvironmentProfile:
    print(f"  [LLM] inspiration: {prompt[:60]}...")
    env = await agenerate_env_profile(
        model_name=model,
        inspiration_prompt=prompt,
        examples="",
    )
    return env


async def main_async(args: argparse.Namespace) -> int:
    prompts = INSPIRATION_PROMPTS[args.scenario_type][: args.n]
    if len(prompts) < args.n:
        print(
            f"[warn] {args.scenario_type} 只有 {len(prompts)} 条预设 prompt，"
            f"实际生成数 = {len(prompts)}"
        )

    if args.dry_run:
        print(f"[dry-run] 将对 {args.scenario_type} 用 {args.model} 生成 {len(prompts)} 条")
        for p in prompts:
            print(f"  - {p}")
        return 0

    print(f"[gen] scenario_type={args.scenario_type}, model={args.model}, n={len(prompts)}")
    tasks = [gen_one(p, args.model) for p in prompts]
    raw_envs = await asyncio.gather(*tasks, return_exceptions=True)

    saved_v1 = 0
    saved_v2 = 0
    for i, env_or_err in enumerate(raw_envs):
        if isinstance(env_or_err, Exception):
            print(f"  [skip] #{i} 生成失败: {env_or_err}")
            continue
        env: EnvironmentProfile = env_or_err  # type: ignore[assignment]
        env.source = f"llm_{args.model}"
        env.codename = env.codename or f"{args.scenario_type}_llm_{i}"
        env.save()
        saved_v1 += 1

        v2 = upgrade_environment_profile(
            env,
            scenario_type=DEFAULTS[args.scenario_type]["scenario_type"],  # type: ignore[arg-type]
            n_agents=2,
            max_days=DEFAULTS[args.scenario_type]["max_days"],  # type: ignore[arg-type]
            intra_day_steps=DEFAULTS[args.scenario_type]["intra_day_steps"],  # type: ignore[arg-type]
            system_state_init={
                "market_state": {"interest_rate": 0.05, "price_index": 100.0},
                "resource_pool": {"water": 100.0},
            },
        )
        v2.save()
        saved_v2 += 1
        print(f"  [ok] #{i} V1={env.pk[:8]}... V2={v2.pk[:8]}...")

    print(f"\n[done] 落库 EnvironmentProfile={saved_v1}, EnvironmentProfileV2={saved_v2}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-type",
        choices=list(INSPIRATION_PROMPTS.keys()),
        required=True,
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("[err] OPENAI_API_KEY 未设置；先在 .env 里配好或 export")
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
