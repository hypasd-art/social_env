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

1. ``default_negotiation_roster`` — 按 ``num_participants``（2/3/4）或 ``quartet`` 推断 N，列出 roster。
2. ``build_llm_negotiation_agents`` — 按 ``negotiation_run_config``（可选）选用
   ``NegotiationSocialLLMAgent`` 与记忆后端（见 ``negotiation_run_config.py``；CLI 为 ``--run-config``）。
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
from contextvars import Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sotopia.database import EnvironmentProfile, SotopiaDimensions
from sotopia.database.base_models import LLMEvalBaseModel

from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    unweighted_aggregate_evaluate,
)
from sotopia.messages import ScriptEnvironmentResponse

from .env import LongTermNegotiationEnv
from .negotiation_llm_agent import NegotiationSocialLLMAgent, build_negotiation_social_llm_agents
from .negotiation_run_config import (
    DEFAULT_NEGOTIATION_RUN_CONFIG,
    build_negotiation_agents_from_run_config,
    load_negotiation_run_config,
)
from .negotiation_metrics import compute_negotiation_rule_metrics
from .scenario_loader import load_negotiation_scenario_from_environment_profile_pk
from .types import (
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
    NegotiationTimelineParams,
    SESSION_SPEAKER_ROLE_ORDER,
    SUPPORTED_NEGOTIATION_LINEUPS,
    negotiation_role_order,
)


@dataclass(frozen=True)
class LongTermNegotiationEvalResult:
    """一次 episode 的规则指标 + 可选的终局 LLM 主观评分聚合。"""

    terminal: str
    rule_metrics: dict[str, float]
    llm_aggregate: ScriptEnvironmentResponse | None


def default_negotiation_roster(
    *,
    quartet: bool | None = None,
    num_participants: int | None = None,
    lineup: str = NEGOTIATION_LINEUP_WITH_INSTITUTIONAL,
) -> tuple[str, ...]:
    """按 ``lineup`` 取前 N 名 canonical 角色（与设计 §4.3 发言顺序一致）。

    ``lineup="with_institutional"`` 沿用 ``SESSION_SPEAKER_ROLE_ORDER``：
    ``firm_a, firm_b, investor, regulator``。

    ``lineup="firms_only"`` 取 ``firm_a, firm_b, firm_c, firm_d``（3+ 家公司互谈）。
    """
    if lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"unknown negotiation lineup {lineup!r}; expected one of "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}"
        )
    order = tuple(negotiation_role_order(lineup))
    if num_participants is not None:
        if num_participants < 2 or num_participants > len(order):
            raise ValueError(f"num_participants must be in [2, {len(order)}], got {num_participants}")
        return order[:num_participants]
    if quartet:
        return order
    return order[:2]


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
    *,
    negotiation_run_config: dict[str, Any] | None = None,
) -> dict[str, NegotiationSocialLLMAgent]:
    """构造谈判专用 ``NegotiationSocialLLMAgent`` 映射。

    ``negotiation_run_config`` 非空时由 ``negotiation_run_config.build_negotiation_agents_from_run_config``
    解析（JSON 文件经 ``load_negotiation_run_config``）；否则使用默认 plain 记忆。
    """
    from .negotiation_run_config import build_negotiation_agents_from_run_config

    return build_negotiation_agents_from_run_config(model_dict, roster, negotiation_run_config)


async def run_llm_negotiation_episode_evaluation(
    model_dict: dict[str, str],
    *,
    quartet: bool = False,
    num_participants: int | None = None,
    lineup: str | None = None,
    params: NegotiationTimelineParams | None = None,
    environment_profile_pk: str | None = None,
    max_macro_steps: int = 4000,
    run_terminal_llm_eval: bool = True,
    evaluation_dimension_model: type[LLMEvalBaseModel] = SotopiaDimensions,
    history_max_action_log: int | None = 500,
    model_trace_dir: Path | str | None = None,
    model_trace_tag: str | None = None,
    execution_trace_dir: Path | str | None = None,
    execution_trace_tag: str | None = None,
    negotiation_run_config: dict[str, Any] | None = None,
) -> LongTermNegotiationEvalResult:
    """跑通一期 **全流程 LLM 参与者** negotiation，并可选用 ``EpisodeLLMEvaluator`` 做终局主观评分。

    ``model_dict`` 须至少包含::

        ``env``: 评测用模型；
        ``agent1``…``agentN``：N 为 ``num_participants``（或 ``quartet``/场景元数据推断的 2/3/4）。

    ``environment_profile_pk`` 若非空则从本地/Redis ``EnvironmentProfile`` 读取
    ``game_metadata.timeline``（及 ``quartet`` / ``num_participants`` / ``strict_design_v1``），见
    ``scenario_loader``. 传入时以场景为准决定人数与时间轴；显式 ``num_participants`` 可覆盖场景中的 N。
    若仍需覆盖时间轴可显式传 ``params``（不推荐与场景混用）。

    需要可用的 LiteLLM / 对应后端 API（与仓库其它 LLM demo 一致），否则仿真或评测调用会失败。

    ``negotiation_run_config``：可选，与 ``negotiation-batch --run-config`` 相同语义的 dict，
    用于选择记忆后端（plain / summarizing）等；默认 plain。

    ``model_trace_dir`` 非空时：在本 episode 期间激活 ``model_trace`` 上下文，将每次 ``agenerate``
    的原始 completion 与解析结果按 **agent** 分文件追加写入
    ``{model_trace_dir}/{stem}_{<agent>}.jsonl``（``stem`` 来自 ``model_trace.safe_trace_filename`` 去掉
    后缀）；无 ``agent`` 元数据时写入 ``{stem}_no_agent.jsonl``；终局评测写入
    ``{stem}_terminal_eval.jsonl``。各行含全局单调 ``step_index``。

    ``execution_trace_dir`` 非空时：在 episode 跑完后将 **全局执行档案**（时间线、合同 history、
    ``action_log`` / ``session_log`` 等）写入 ``{execution_trace_dir}/{execution_trace_tag}.execution.json``
    （见 ``episode_execution_record.safe_execution_trace_filename``）。
    """
    if "env" not in model_dict:
        raise KeyError("model_dict must contain key 'env' for the evaluator / scoring model.")

    trace_token: Token | None = None
    if model_trace_dir is not None:
        from .model_trace import begin_episode_trace, safe_trace_filename

        trace_path = Path(model_trace_dir).resolve() / safe_trace_filename(
            model_trace_tag or "negotiation_episode"
        )
        trace_token = begin_episode_trace(trace_path)

    n_from_scen: int | None = None
    lineup_from_scen: str | None = None
    predefined_rule: dict[str, Any] | None = None
    if environment_profile_pk:
        scen = load_negotiation_scenario_from_environment_profile_pk(environment_profile_pk)
        env_profile = EnvironmentProfile.get(environment_profile_pk)
        gm = env_profile.game_metadata if isinstance(env_profile.game_metadata, dict) else {}
        raw_rule = gm.get("predefined_outcome_rule")
        if isinstance(raw_rule, dict):
            predefined_rule = dict(raw_rule)
        strict_run = scen.strict_design_v1
        n_from_scen = scen.num_participants
        lineup_from_scen = scen.lineup
        params_run = scen.params if params is None else params
    else:
        strict_run = quartet
        params_run = params or NegotiationTimelineParams(
            D=8,
            s_max_per_day=2,
            max_session_rounds=40,
            max_total_turns_per_session=80,
        )

    if num_participants is not None:
        n = num_participants
    elif n_from_scen is not None:
        n = n_from_scen
    else:
        n = 4 if quartet else 2

    effective_lineup = lineup or lineup_from_scen or NEGOTIATION_LINEUP_WITH_INSTITUTIONAL
    if effective_lineup not in SUPPORTED_NEGOTIATION_LINEUPS:
        raise ValueError(
            f"unknown negotiation lineup {effective_lineup!r}; expected one of "
            f"{sorted(SUPPORTED_NEGOTIATION_LINEUPS)}"
        )
    if n < 2 or n > 4:
        raise ValueError(f"effective num_participants must be 2..4, got {n}")

    try:
        roster = default_negotiation_roster(num_participants=n, lineup=effective_lineup)
        if len(roster) < 4 or effective_lineup == NEGOTIATION_LINEUP_FIRMS_ONLY:
            strict_run = False
        agents_map = build_llm_negotiation_agents(
            model_dict, roster, negotiation_run_config=negotiation_run_config
        )

        env = LongTermNegotiationEnv(
            agents_map,
            params=params_run,
            strict_design_v1=strict_run,
        )

        terminal = await env.run_episode_async(max_macro_steps=max_macro_steps)
        rule_metrics = compute_negotiation_rule_metrics(env, predefined_outcome_rule=predefined_rule)

        if execution_trace_dir is not None:
            from .episode_execution_record import (
                safe_execution_trace_filename,
                write_episode_execution_record,
            )

            ex_path = Path(execution_trace_dir).resolve() / safe_execution_trace_filename(
                execution_trace_tag or "negotiation_episode"
            )
            write_episode_execution_record(env, ex_path)

        llm_agg: ScriptEnvironmentResponse | None = None
        if run_terminal_llm_eval:
            history = format_negotiation_episode_for_llm_eval(
                env, max_action_log=history_max_action_log
            )
            evaluator = EpisodeLLMEvaluator(
                model_name=model_dict["env"],
                response_format_class=EvaluationForAgents[evaluation_dimension_model],  # type: ignore[valid-type]
            )
            raw = await evaluator.__acall__(turn_number=-1, history=history, messages=None)
            llm_agg = unweighted_aggregate_evaluate(list(raw))
            if trace_token is not None:
                from .model_trace import record_terminal_eval_step

                record_terminal_eval_step(
                    model_name=model_dict["env"],
                    history=history,
                    aggregate=llm_agg,
                )

        return LongTermNegotiationEvalResult(
            terminal=terminal,
            rule_metrics=rule_metrics,
            llm_aggregate=llm_agg,
        )
    finally:
        if trace_token is not None:
            from .model_trace import end_episode_trace

            end_episode_trace(trace_token)


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
    "load_negotiation_run_config",
    "build_negotiation_agents_from_run_config",
    "DEFAULT_NEGOTIATION_RUN_CONFIG",
]
