#!/usr/bin/env python
"""完全从零造数据（LLM 版）：用 LLM 批量生成 AgentProfile + EnvironmentProfile，
然后把 RelationshipProfile / EnvAgentComboStorage / EnvironmentList / V2 全套
按 ``scripts/generate_from_scratch.py`` 的同款规则一并落库。

设计目标
--------
- **不修改** ``scripts/generate_from_scratch.py``，落库相关函数直接 import 复用。
- **LLM 生成层** 参考既有套路：
  - ``scripts/generate_v2_with_llm.py``：async gather + 失败容忍
  - ``sotopia/generation_utils/generate.py:agenerate_env_profile``：环境 prompt 模板
  - ``sotopia/generation_utils/generate.py:agenerate``：通用 ``PydanticOutputParser`` 调用入口
- **theme 驱动**：给一个主题关键词（如 "rural farming community" /
  "post-AI white-collar layoffs"），脚本自动衍生多条 inspiration prompt，
  让同一主题下的 agent 与 env 互相呼应。

用法
----
    # 1) 默认主题，造 6 角色 + 4 场景
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_from_scratch_with_llm.py --clean --with-v2

    # 2) 指定主题 + 模型 + 规模
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_from_scratch_with_llm.py \\
        --theme "AI startup founders facing a market crash" \\
        --n-agents 8 --n-envs 5 --combos-per-env 3 \\
        --model gpt-4o-mini --tag llm_run_v1 --with-v2

    # 3) dry-run（只打印 prompts，不调 LLM、不落库）
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python \\
        scripts/generate_from_scratch_with_llm.py --dry-run

落库目录与 ``generate_from_scratch.py`` 一致：
    ~/.sotopia/data/{AgentProfile,EnvironmentProfile,RelationshipProfile,
                     EnvAgentComboStorage,EnvironmentList,
                     AgentProfileV2,EnvironmentProfileV2,
                     EventScript,Contract,SystemStateSnapshot}/<pk>.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 必须在 import sotopia 之前固定后端，否则 V2 模型会按 Redis 路径 import
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")

# 中文注释：加载 .env 文件，让 OPENAI_API_KEY / OPENAI_BASE_URL 可用
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from sotopia.database import (  # noqa: E402
    AgentProfile,
    EnvironmentProfile,
)
from sotopia.generation_utils.generate import (  # noqa: E402
    agenerate,
    agenerate_env_profile,
)
from sotopia.generation_utils.output_parsers import PydanticOutputParser  # noqa: E402

# 中文注释：复用 generate_from_scratch.py 的落库工具，避免重复造轮子
from generate_from_scratch import (  # type: ignore[import-not-found]  # noqa: E402
    LOCAL_DATA_DIR,
    maybe_build_v2,
    save_combos,
    save_environment_list,
    save_relationships,
    wipe_local_data,
)


# ---------------------------------------------------------------------------
# 1) Theme → inspiration prompts 的派生规则
#    主题关键词决定 agent 职业池 + env 场景池，让同一批数据风格自洽
# ---------------------------------------------------------------------------


THEME_PRESETS: dict[str, dict[str, list[str]]] = {
    "default": {
        "agent_briefs": [
            "a 32-year-old veterinarian in a small town who recently inherited debt",
            "a 47-year-old laid-off factory foreman trying to retrain as a software tester",
            "a 24-year-old graduate student whose advisor just left academia",
            "a 58-year-old retired schoolteacher running a community garden",
            "a 29-year-old non-binary musician who side-hustles as a delivery driver",
            "a 41-year-old single-parent civil engineer juggling custody disputes",
            "a 19-year-old college dropout starting a vintage clothing reselling business",
            "a 53-year-old recently widowed accountant returning to dating apps",
        ],
        "scenario_briefs": [
            "two roommates negotiating who pays for a broken washing machine",
            "two coworkers splitting equity in a side project that started making money",
            "two neighbors arguing over a shared driveway repair cost",
            "two siblings deciding how to share elder care for an aging parent",
            "a freelancer renegotiating an overdue payment with a long-time client",
            "a small landlord and tenant negotiating a late rent payment plan",
        ],
    },
    "ai_startup_crash": {
        "agent_briefs": [
            "a 38-year-old founder of an AI agent startup whose Series A just collapsed",
            "a 28-year-old senior engineer holding 1% equity, deciding whether to stay",
            "a 45-year-old VC partner who led the last round and faces LP heat",
            "a 33-year-old chief of staff who suspects the founder is hiding numbers",
            "a 26-year-old new hire who joined two weeks ago without knowing the situation",
            "a 51-year-old enterprise sales lead trying to close one final deal",
        ],
        "scenario_briefs": [
            "founder and senior engineer negotiating equity acceleration after the failed round",
            "founder and VC negotiating a bridge note vs a fire-sale acquihire",
            "co-founders deciding whether to lay off half the team this Friday",
            "the new hire confronting the founder about misrepresented financial runway",
            "two co-founders splitting IP rights as one wants to fork the codebase",
        ],
    },
    "drought_village": {
        "agent_briefs": [
            "a 55-year-old farmer relying on the village well for crop irrigation",
            "a 48-year-old livestock owner whose cattle need daily water",
            "a 62-year-old village council elder responsible for water allocation",
            "a 31-year-old local NGO worker pushing for fair distribution",
            "a 40-year-old well operator paid by the council, with side incentives",
        ],
        "scenario_briefs": [
            "the farmer and livestock owner deciding how to share reduced well output",
            "the council elder and NGO worker negotiating an emergency rationing rule",
            "two farmers debating whether to pool water rights to save half their crops",
        ],
    },
    "post_layoff_white_collar": {
        "agent_briefs": [
            "a 34-year-old software engineer recently laid off, with two children",
            "a 41-year-old mid-level manager whose role was replaced by AI tools",
            "a 27-year-old recruiter pivoting to running a cohort-based course",
            "a 50-year-old former financial analyst struggling with age bias in interviews",
            "a 30-year-old contract worker forming a new freelancer collective",
        ],
        "scenario_briefs": [
            "two laid-off colleagues deciding whether to start a consultancy together",
            "former boss and former report negotiating a referral letter terms",
            "two friends arguing whether to take a buyout package or fight termination",
        ],
    },
}


def derive_briefs(theme: str) -> tuple[list[str], list[str]]:
    """根据 theme 选/造 agent_briefs + scenario_briefs。

    - 命中 ``THEME_PRESETS`` 的预设 → 直接用
    - 未命中（自由文本） → 用通用模板派生
    """

    if theme in THEME_PRESETS:
        spec = THEME_PRESETS[theme]
        return spec["agent_briefs"], spec["scenario_briefs"]

    # 中文注释：自由 theme 时，让 LLM 在 agenerate_env_profile / agent prompt
    # 里自行解读，这里只产出几个轻引导字符串
    free_agent_briefs = [
        f"a person in their 30s deeply involved in '{theme}', under financial pressure",
        f"a person in their 50s with long experience related to '{theme}'",
        f"a young adult navigating '{theme}' while supporting family",
        f"a mid-career professional pivoting because of '{theme}'",
        f"a peripheral observer / journalist documenting '{theme}'",
        f"a regulator/authority figure trying to govern '{theme}'",
    ]
    free_scenario_briefs = [
        f"two characters embedded in '{theme}' negotiating an immediate trade-off",
        f"a regulator and a regulated party debating consequences of '{theme}'",
        f"two peers debating whether to publicly disclose information about '{theme}'",
        f"two collaborators in '{theme}' splitting credit and risk",
    ]
    return free_agent_briefs, free_scenario_briefs


# ---------------------------------------------------------------------------
# 2) LLM 生成：单个 agent / 单个 environment
# ---------------------------------------------------------------------------


AGENT_TEMPLATE = """Please generate a complete, REALISTIC agent profile filling
ALL fields below. Make the character distinctive, with concrete personality details
and a meaningful private 'secret' that could create dramatic tension in social
interactions.

Brief seed (use as inspiration; you may diversify gender / age within reason):
{brief}

CONSTRAINTS:
- first_name and last_name MUST be plausible real-world names (not placeholders).
- big_five MUST follow the format:
  "Openness: <high|medium|low>; Conscientiousness: ...; Extraversion: ...;
   Agreeableness: ...; Neuroticism: ..."
- moral_values is a list of 1-3 strings from:
  ["care", "fairness", "loyalty", "authority", "purity", "liberty"]
- schwartz_personal_values is a list of 1-3 strings from:
  ["self-direction", "stimulation", "hedonism", "achievement", "power",
   "security", "conformity", "tradition", "benevolence", "universalism"]
- mbti is a 4-letter type (e.g., "INTJ", "ENFP")
- secret is ONE concrete private fact that the agent would NOT casually share

Please use the following format strictly. DO NOT echo the schema; fill it in.
{format_instructions}
"""


async def generate_agent(
    *, brief: str, model: str, temperature: float = 0.8
) -> AgentProfile:
    """用 LLM 生成一个 AgentProfile。结构化 JSON 由 PydanticOutputParser 解析。"""

    print(f"  [agent] brief: {brief[:70]}...")
    return await agenerate(
        model_name=model,
        template=AGENT_TEMPLATE,
        input_values=dict(brief=brief),
        output_parser=PydanticOutputParser(pydantic_object=AgentProfile),
        temperature=temperature,
    )


async def generate_env(
    *, brief: str, model: str, temperature: float = 0.8
) -> EnvironmentProfile:
    """用 LLM 生成一个 EnvironmentProfile。

    直接复用 ``agenerate_env_profile``——它的 prompt 已经被官方调过，能稳定
    产出 ``scenario`` + ``agent_goals`` 两个核心字段。
    """

    print(f"  [env]   brief: {brief[:70]}...")
    return await agenerate_env_profile(
        model_name=model,
        inspiration_prompt=brief,
        examples="",
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 3) 落 AgentProfile / EnvironmentProfile（保留 LLM 输出，加上 tag/source）
# ---------------------------------------------------------------------------


def save_llm_agents(
    agents: list[AgentProfile], *, tag: str, theme: str
) -> list[AgentProfile]:
    saved = []
    for i, a in enumerate(agents):
        a.tag = tag
        # 中文注释：first_name/last_name 缺失时打个兜底，避免 redis-om 索引报错
        if not a.first_name:
            a.first_name = f"AgentX{i}"
        if not a.last_name:
            a.last_name = f"Synth{i}"
        a.save()
        saved.append(a)
    print(f"[save] AgentProfile (LLM, theme={theme}) x {len(saved)}")
    return saved


def save_llm_envs(
    envs: list[EnvironmentProfile], *, theme: str, model: str
) -> list[EnvironmentProfile]:
    saved = []
    for i, e in enumerate(envs):
        e.source = f"llm_{model}_{theme}"
        e.codename = e.codename or f"{theme}_llm_{i}"
        e.save()
        saved.append(e)
    print(f"[save] EnvironmentProfile (LLM, theme={theme}) x {len(saved)}")
    return saved


# ---------------------------------------------------------------------------
# 4) main
# ---------------------------------------------------------------------------


async def _generate_all(
    *,
    n_agents: int,
    n_envs: int,
    theme: str,
    model: str,
    temperature: float,
    concurrency: int,
) -> tuple[list[AgentProfile], list[EnvironmentProfile]]:
    """并发生成 n_agents 角色 + n_envs 场景；用信号量限并发，避免触发 RPM。"""

    agent_briefs, scenario_briefs = derive_briefs(theme)
    # 不够就轮回循环填满
    agent_briefs = (agent_briefs * ((n_agents // len(agent_briefs)) + 1))[:n_agents]
    scenario_briefs = (
        scenario_briefs * ((n_envs // len(scenario_briefs)) + 1)
    )[:n_envs]

    sem = asyncio.Semaphore(concurrency)

    async def _agent(b: str) -> AgentProfile | Exception:
        async with sem:
            try:
                return await generate_agent(
                    brief=b, model=model, temperature=temperature
                )
            except Exception as e:
                return e

    async def _env(b: str) -> EnvironmentProfile | Exception:
        async with sem:
            try:
                return await generate_env(
                    brief=b, model=model, temperature=temperature
                )
            except Exception as e:
                return e

    print(f"[gen] {n_agents} agents + {n_envs} envs, model={model}, "
          f"concurrency={concurrency}")
    agent_tasks = [_agent(b) for b in agent_briefs]
    env_tasks = [_env(b) for b in scenario_briefs]
    raw_agents, raw_envs = await asyncio.gather(
        asyncio.gather(*agent_tasks),
        asyncio.gather(*env_tasks),
    )

    ok_agents: list[AgentProfile] = []
    for i, r in enumerate(raw_agents):
        if isinstance(r, Exception):
            print(f"  [skip] agent #{i} failed: {r}")
        else:
            ok_agents.append(r)
    ok_envs: list[EnvironmentProfile] = []
    for i, r in enumerate(raw_envs):
        if isinstance(r, Exception):
            print(f"  [skip] env #{i} failed: {r}")
        else:
            ok_envs.append(r)

    print(f"[gen] success: agents={len(ok_agents)}/{n_agents}, "
          f"envs={len(ok_envs)}/{n_envs}")
    return ok_agents, ok_envs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="先清空 ~/.sotopia/data/")
    parser.add_argument(
        "--theme",
        default="default",
        help="主题关键词；命中 THEME_PRESETS 用预设，否则按自由文本派生",
    )
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--combos-per-env", type=int, default=2)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--list-name", default="scratch_llm_env_set")
    parser.add_argument(
        "--override-hard-list",
        action="store_true",
        help="把 EnvironmentList.pk 设为官方 hard 列表 ULID，让 task=hard 命中",
    )
    parser.add_argument("--with-v2", action="store_true", help="同时造 V2 数据")
    parser.add_argument("--tag", default="scratch_llm_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[backend] SOTOPIA_STORAGE_BACKEND={os.environ['SOTOPIA_STORAGE_BACKEND']}")
    print(f"[paths]   {LOCAL_DATA_DIR}")
    print(f"[plan] theme='{args.theme}' agents={args.n_agents} "
          f"envs={args.n_envs} combos/env={args.combos_per_env} model={args.model}")

    if args.dry_run:
        agent_briefs, scenario_briefs = derive_briefs(args.theme)
        print(f"[dry-run] agent_briefs (前 {args.n_agents} 条):")
        for b in agent_briefs[: args.n_agents]:
            print(f"  - {b}")
        print(f"[dry-run] scenario_briefs (前 {args.n_envs} 条):")
        for b in scenario_briefs[: args.n_envs]:
            print(f"  - {b}")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("[err] OPENAI_API_KEY 未设置；先在 .env 里配好或 export")
        return 1

    if args.clean:
        wipe_local_data(yes=True)

    # 中文注释：先 LLM 生成，再批量落库；分开有助于失败后只重跑生成步
    agents, envs = asyncio.run(
        _generate_all(
            n_agents=args.n_agents,
            n_envs=args.n_envs,
            theme=args.theme,
            model=args.model,
            temperature=args.temperature,
            concurrency=args.concurrency,
        )
    )

    if not agents or not envs:
        print("[err] 至少需要 1 个 agent + 1 个 env 才能继续，跑不下去")
        return 1
    if len(agents) < 2:
        print("[err] 至少需要 2 个 agent 才能凑成 combo（每个 env 配 2 个 agent）")
        return 1

    print("\n[save] 开始落库")
    agents = save_llm_agents(agents, tag=args.tag, theme=args.theme)
    envs = save_llm_envs(envs, theme=args.theme, model=args.model)
    save_relationships(agents, tag=args.tag)
    combos = save_combos(
        envs, agents, combos_per_env=args.combos_per_env, seed=args.seed
    )
    save_environment_list(
        envs, combos, name=args.list_name, override_hard_pk=args.override_hard_list
    )

    if args.with_v2:
        print("\n[v2] 升级 V2 数据 ...")
        maybe_build_v2(agents, envs, tag=args.tag)

    print("\n========== DONE ==========")
    for sub in [
        "AgentProfile",
        "EnvironmentProfile",
        "RelationshipProfile",
        "EnvAgentComboStorage",
        "EnvironmentList",
        "AgentProfileV2",
        "EnvironmentProfileV2",
        "EventScript",
        "Contract",
        "SystemStateSnapshot",
    ]:
        d = LOCAL_DATA_DIR / sub
        n = len(list(d.glob("*.json"))) if d.exists() else 0
        print(f"  {sub:<25} {n}")

    print("\n下一步：")
    print(
        "  SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \\"
        "\n    --models gpt-4o-mini --partner-model gpt-4o-mini \\"
        f"\n    --evaluator-model gpt-4o-mini --task scratch \\"
        f"\n    --tag {args.tag}_run0 --push-to-db --batch-size 4"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
