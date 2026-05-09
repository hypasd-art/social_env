"""长期谈判 **大模型仿真 + LLM 终局评测** 入口（对齐 ``examples/minimalist_demo.py`` 的 ``model_dict`` 用法）。

未挂载到 ``sotopia.settings`` 顶层导入，便于仅跑规则 agent 的路径不拉起 LLM/评测依赖。推荐::

    from sotopia.settings.long_term_negotiation.llm_evaluation import (
        run_llm_negotiation_episode_evaluation,
    )

``model_dict`` 约定::

    ``env``: 评测模型（传给 ``EpisodeLLMEvaluator``）
    ``agent1`` … ``agentN``: 与 ``roster`` 稳定排序后的第 i 名参与者对应的行动模型；
    bilateral 时需 ``agent1``/``agent2``；四方谈判需 ``agent1``…``agent4``.

================================================================================
程序化「跑一局评测」：函数调用顺序与作用（批量 CLI 最终会走到这里）
================================================================================

主函数：**``run_llm_negotiation_episode_evaluation``**（本模块）

调用顺序简述：

1. ``default_negotiation_roster`` —  bilateral / quartet 下列出参与者名字顺序（ roster ）。
2. ``build_llm_negotiation_agents`` — 实际是 ``negotiation_llm_agent.build_negotiation_social_llm_agents``，
   为 roster 每一名建 ``NegotiationSocialLLMAgent``（文件 ``negotiation_llm_agent.py``）。
3. ``LongTermNegotiationEnv``（``env.py``）— 挂载 ``NegotiationWorldController``、``SystemState``、
   messenger、外部事件 runner 等；用于一条 episode 的宏观调度与会话闭环。
4. ``await LongTermNegotiationEnv.run_episode_async`` — 驱动 ``ctrl`` 的各 ``Phase``
   （约见 → 应答 → SESSION 内多轮 Agent 行动），直到终止或 ``max_macro_steps``；
   内部通过各 agent 的 ``aact`` 生成 ``AgentAction``（见 ``negotiation_llm_agent`` 与 ``controller.parse_agent_action_payload``）。
5. ``compute_negotiation_rule_metrics``（``negotiation_metrics.py``）— 从环境与 controller 日志抽取**规则向**标量指标。
6. （可选）若 ``run_terminal_llm_eval``：

   - ``format_negotiation_episode_for_llm_eval`` — 把调度 / 会话 / 动作日志压成单段文本；
   - ``EpisodeLLMEvaluator.__acall__``（``sotopia.envs.evaluators``）— 用 ``model_dict['env']`` 做终局主观评分；
   - ``unweighted_aggregate_evaluate`` — 聚合成 ``ScriptEnvironmentResponse``。

返回 **``LongTermNegotiationEvalResult``**（terminal 字符串 + rule_metrics + 可选 llm_aggregate）。

同步封装：**``evaluate_long_term_negotiation_llm_sync``** — 单测或脚本里 ``asyncio.run`` 一行调用。

批量场景不要在本层手写循环，请用 ``batch_evaluation.run_long_term_negotiation_eval_batch``。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from sotopia.database import SotopiaDimensions
from sotopia.database.base_models import LLMEvalBaseModel

from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    unweighted_aggregate_evaluate,
)
from sotopia.messages import ScriptEnvironmentResponse

from .env import LongTermNegotiationEnv
from .negotiation_llm_agent import (
    NegotiationSocialLLMAgent,
    build_negotiation_social_llm_agents,
)
from .negotiation_metrics import compute_negotiation_rule_metrics
from .roles import CANONICAL_NEGOTIATION_ROSTER
from .scenario_loader import load_negotiation_scenario_from_environment_profile_pk
from .types import NegotiationTimelineParams


@dataclass(frozen=True)
class LongTermNegotiationEvalResult:
    """一次 episode 的规则指标 + 可选的终局 LLM 主观评分聚合。"""

    terminal: str
    rule_metrics: dict[str, float]
    llm_aggregate: ScriptEnvironmentResponse | None


def default_negotiation_roster(*, quartet: bool) -> tuple[str, ...]:
    """与 ``examples/long_term_negotiation_dummy_run`` 一致的 roster 顺序（按名字排序）。"""
    if quartet:
        return tuple(sorted(CANONICAL_NEGOTIATION_ROSTER))
    return ("firm_a", "firm_b")


def format_negotiation_episode_for_llm_eval(env: LongTermNegotiationEnv, *, max_action_log: int | None = 500) -> str:
    """将调度与会话轨迹压成便于 ``EpisodeLLMEvaluator`` 使用的单段文本。"""
    ctrl = env.ctrl
    lines: list[str] = []
    lines.append("# Scheduling")
    for day, slot, agent, nl in ctrl.scheduling_log:
        lines.append(f"day={day} slot={slot} | {agent}: {nl}")
    lines.append("# Session log")
    for entry in ctrl.session_log:
        lines.append(json.dumps(entry, ensure_ascii=False, default=str))
    lines.append("# Action log")
    alog = ctrl.action_log
    if max_action_log is not None and len(alog) > max_action_log:
        alog = alog[-max_action_log:]
        lines.append(f"(truncated to last {max_action_log} entries)")
    for row in alog:
        lines.append(json.dumps(row, ensure_ascii=False, default=str))
    return "\n".join(lines)


def build_llm_negotiation_agents(
    model_dict: dict[str, str],
    roster: tuple[str, ...],
) -> dict[str, NegotiationSocialLLMAgent]:
    """构造谈判专用 ``NegotiationSocialLLMAgent`` 映射（等价于 ``build_negotiation_social_llm_agents``）。"""
    return build_negotiation_social_llm_agents(model_dict, roster)


async def run_llm_negotiation_episode_evaluation(
    model_dict: dict[str, str],
    *,
    quartet: bool = False,
    params: NegotiationTimelineParams | None = None,
    environment_profile_pk: str | None = None,
    max_macro_steps: int = 4000,
    run_terminal_llm_eval: bool = True,
    evaluation_dimension_model: type[LLMEvalBaseModel] = SotopiaDimensions,
    history_max_action_log: int | None = 500,
) -> LongTermNegotiationEvalResult:
    """跑通一期 **全流程 LLM 参与者** negotiation，并可选用 ``EpisodeLLMEvaluator`` 做终局主观评分。

    ``model_dict`` 须至少包含::

        ``env``: 评测用模型；
        ``agent1``, ``agent2``（ bilateral ）；若 ``quartet=True`` 则还需 ``agent3``, ``agent4``。

    ``environment_profile_pk`` 若非空则从本地/Redis ``EnvironmentProfile`` 读取
    ``game_metadata.timeline``（及 ``quartet`` / ``strict_design_v1``），见
    ``scenario_loader``. 传入时以场景为准决定 ``quartet`` / 时间轴；若仍需覆盖时间轴可显式传
    ``params``（不推荐与场景混用）。

    需要可用的 LiteLLM / 对应后端 API（与仓库其它 LLM demo 一致），否则仿真或评测调用会失败。
    """
    if "env" not in model_dict:
        raise KeyError("model_dict must contain key 'env' for the evaluator / scoring model.")

    quartet_run = quartet
    strict_run = quartet_run
    if environment_profile_pk:
        scen = load_negotiation_scenario_from_environment_profile_pk(environment_profile_pk)
        quartet_run = scen.quartet
        strict_run = scen.strict_design_v1
        params_run = scen.params if params is None else params
    else:
        quartet_run = quartet
        strict_run = quartet_run
        params_run = params or NegotiationTimelineParams(
            D=8,
            s_max_per_day=2,
            max_session_rounds=40,
            max_total_turns_per_session=80,
        )

    roster = default_negotiation_roster(quartet=quartet_run)
    agents_map = build_llm_negotiation_agents(model_dict, roster)

    env = LongTermNegotiationEnv(
        agents_map,
        params=params_run,
        strict_design_v1=strict_run,
    )

    terminal = await env.run_episode_async(max_macro_steps=max_macro_steps)
    rule_metrics = compute_negotiation_rule_metrics(env)

    llm_agg: ScriptEnvironmentResponse | None = None
    if run_terminal_llm_eval:
        history = format_negotiation_episode_for_llm_eval(env, max_action_log=history_max_action_log)
        evaluator = EpisodeLLMEvaluator(
            model_name=model_dict["env"],
            response_format_class=EvaluationForAgents[evaluation_dimension_model],  # type: ignore[valid-type]
        )
        raw = await evaluator.__acall__(turn_number=-1, history=history, messages=None)
        llm_agg = unweighted_aggregate_evaluate(list(raw))

    return LongTermNegotiationEvalResult(
        terminal=terminal,
        rule_metrics=rule_metrics,
        llm_aggregate=llm_agg,
    )


def evaluate_long_term_negotiation_llm_sync(
    model_dict: dict[str, str],
    **kwargs: Any,
) -> LongTermNegotiationEvalResult:
    """同步薄封装，便于与 ``asyncio.run(run_async_server(...))`` 同样的脚本风格一行调用。"""
    return asyncio.run(run_llm_negotiation_episode_evaluation(model_dict, **kwargs))


__all__ = [
    "LongTermNegotiationEvalResult",
    "NegotiationSocialLLMAgent",
    "build_llm_negotiation_agents",
    "build_negotiation_social_llm_agents",
    "default_negotiation_roster",
    "evaluate_long_term_negotiation_llm_sync",
    "format_negotiation_episode_for_llm_eval",
    "run_llm_negotiation_episode_evaluation",
]
