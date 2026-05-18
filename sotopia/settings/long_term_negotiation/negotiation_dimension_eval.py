"""长期谈判双维度 LLM 评测模块。

两个评测维度（分数 1-10）：

1. **persona_style_consistency**（个性与语言风格一致性）
   评估 agent 的交互行为是否符合其设定的人物个性、语言风格和角色定位。

2. **goal_behavioral_competence**（目标完成与行为执行能力）
   评估 agent 在谈判中是否有效推进自身目标，以及是否展现出合理的合作/竞争行为。

输入：交互历史 + agent 角色设定/目标描述
输出：每个 agent 在每个维度上的分数 + reasoning + 统计摘要
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field

from sotopia.database.base_models import LLMBaseModel
from sotopia.generation_utils.generate import agenerate
from sotopia.generation_utils.output_parsers import PydanticOutputParser

log = logging.getLogger("sotopia.negotiation.dim_eval")


# ============================================================================
# 维度分数模型（含 1-10 各级别描述）
# ============================================================================


class PersonaStyleScore(LLMBaseModel):
    """个性与语言风格一致性评分（1-10）。

    各级别含义：
    1-2 完全不符：行为与设定个性/风格完全矛盾，语言模式在所有 agent 间无法区分。
    3-4 明显偏离：大部分交互偏离设定风格，偶尔有一些符合但总体上不可辨认。
    5-6 部分符合：基本风格有时可见，但存在明显的偏离、不连贯或风格混用。
    7-8 较好符合：大部分交互贴合设定个性和语言风格，仅有少量不自然之处。
    9-10 高度符合：完全体现设定个性与语言风格，与其他 agent 有明显的风格区分，
           表达自然、连贯、有辨识度。
    """

    reasoning: str = Field(
        ...,
        description=(
            "详细推理解释：1) 指出 agent 的设定个性/语言风格要点；"
            "2) 引用交互历史中的具体片段说明符合或不符合之处；"
            "3) 给出分数的逻辑依据。"
        ),
    )
    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="个性与语言风格一致性分数（1-10）",
    )


class GoalBehavioralScore(LLMBaseModel):
    """目标完成与行为执行能力评分（1-10）。

    各级别含义：
    1-2 几乎无进展：未采取任何有意义的推进目标的行为，完全被动或无关互动。
    3-4 少量进展：有基本的交互尝试但未形成有效推进，目标无明显推进。
    5-6 部分完成：采取了若干有效行动，部分子目标有所推进，但整体执行力一般。
    7-8 大部分完成：目标大部分已推进或完成，展现了较好的合作/竞争策略和谈判技巧。
    9-10 完全/超额完成：所有核心目标均已达成，展现出优秀的战略规划能力，
           能根据局势灵活调整合作/竞争姿态，行为高效且富有策略性。
    """

    reasoning: str = Field(
        ...,
        description=(
            "详细推理解释：1) 指出 agent 的核心目标和子目标；"
            "2) 引用交互历史中的具体行为说明目标推进情况；"
            "3) 评估 agent 的合作/竞争策略是否恰当有效。"
        ),
    )
    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="目标完成与行为执行能力分数（1-10）",
    )


class AgentTwoDimensionEvaluation(LLMBaseModel):
    """单个 agent 的双维度评测结果。"""

    persona_style_consistency: PersonaStyleScore
    goal_behavioral_competence: GoalBehavioralScore


class TwoDimensionEvaluationResult(LLMBaseModel):
    """一次 episode 中所有 agent 的双维度评测结果。"""

    evaluations: dict[str, AgentTwoDimensionEvaluation] = Field(
        ...,
        description="按 agent key（如 agent_1, agent_2, ...）组织的评测结果",
    )


# ============================================================================
# 评测 prompt 模板
# ============================================================================

EVALUATION_SYSTEM_PROMPT_EN = """You are an expert evaluator of multi-agent negotiation simulations. Your task is to evaluate each agent's performance on TWO dimensions. You must be strict, evidence-based, and provide detailed reasoning.

## Dimension 1: Persona & Language Style Consistency (1-10)
Evaluate how well each agent's behavior and language matches their assigned personality, role profile, and speaking style.

### Score Levels:
**1-2 (Completely Inconsistent):** Behavior contradicts assigned persona. Language is indistinguishable from other agents. No identifiable personal style.
**3-4 (Largely Inconsistent):** Most interactions deviate from the assigned style. Occasional moments of alignment but overall unrecognizable. Generic responses dominate.
**5-6 (Partially Consistent):** Basic style sometimes visible, but notable inconsistencies, style drift, or mixing of incompatible patterns. The agent's unique voice is occasionally identifiable.
**7-8 (Largely Consistent):** Most interactions align with assigned persona and language style. Only minor deviations. The agent has a recognizable and distinct voice.
**9-10 (Highly Consistent):** Fully embodies assigned personality and language style. Clearly distinguishable from other agents. Expression is natural, coherent, and distinctive throughout.

### What to look for:
- Speech register (formal/blunt/warm/calculating)
- Pacing (terse vs chatty)
- Distinctive phrases, hedges, negotiation tactics matching the profile
- Consistency of tone across all interactions
- Differentiation from other agents' voices

## Dimension 2: Goal Achievement & Behavioral Competence (1-10)
Evaluate how effectively each agent advances its stated goals and demonstrates appropriate cooperation/competition behaviors.

### Score Levels:
**1-2 (Almost No Progress):** No meaningful actions toward any goal. Completely passive or irrelevant interactions.
**3-4 (Minimal Progress):** Basic interaction attempts exist but produce no effective advancement. Goals remain essentially untouched.
**5-6 (Partial Achievement):** Some effective actions taken. Partial sub-goals advanced. Overall execution is mediocre. Basic cooperation/competition attempts but lacking sophistication.
**7-8 (Substantial Achievement):** Most goals substantially advanced or achieved. Demonstrates good cooperation/competition strategy and negotiation skills. Adapts to situations reasonably well.
**9-10 (Full/Exceptional Achievement):** All core goals achieved. Demonstrates excellent strategic planning. Flexibly adjusts cooperation/competition posture. Behavior is efficient, strategic, and highly effective.

### What to look for:
- Whether the agent actively pursues stated objectives
- Quality of proposals, counter-offers, and information gathering
- Appropriate use of cooperation vs competition based on context
- Strategic adaptation to events and other agents' moves
- Whether beneficial outcomes (contracts, deals) are achieved

## Output Format
For each agent, provide:
1. **reasoning**: Detailed analysis citing specific examples from the interaction history. Explain WHY the score is given, not just what happened.
2. **score**: An integer from 1 to 10.

Be strict. Reserve extreme scores (1-2, 9-10) for clear cases. Most agents should score in the 4-8 range unless there is strong evidence otherwise."""


AGENT_PROFILE_SECTION_TEMPLATE = """
## Agent Profile: {agent_name}
{profile_text}
"""

EVALUATION_USER_PROMPT_TEMPLATE = """{agent_profiles}

## Interaction History
{history}

Based on the above interaction history and each agent's profile, evaluate EVERY agent on the TWO dimensions (persona_style_consistency and goal_behavioral_competence).
{agent_key_instruction}

For each agent, provide detailed reasoning that references specific examples from the history, and a score from 1 to 10 for each dimension.
{extra}"""


# ============================================================================
# 评测执行
# ============================================================================


def _build_agent_profiles_text(
    agent_roles: list[str],
    agent_profiles: dict[str, str],
    agent_display_names: dict[str, str] | None = None,
) -> str:
    """将各 agent 的角色设定和 profile 信息拼接为评测 prompt 的前缀。"""
    names = agent_display_names or {}
    sections: list[str] = []
    for role in agent_roles:
        name = names.get(role, role)
        profile = agent_profiles.get(role, "")
        if not profile:
            profile = f"Role: {role}. No additional profile provided."
        sections.append(
            AGENT_PROFILE_SECTION_TEMPLATE.format(
                agent_name=name, profile_text=profile
            )
        )
    return "\n".join(sections)


def _build_agent_key_instruction(num_agents: int) -> str:
    """生成固定 agent key 的指令。"""
    keys = [f'"agent_{i+1}"' for i in range(num_agents)]
    return (
        f"There are exactly {num_agents} agents. Under the 'evaluations' field, "
        f"use exactly these keys: [{', '.join(keys)}] (no other keys)."
    )


def _extract_agent_profiles(
    env: Any,
    agent_roles: list[str],
) -> dict[str, str]:
    """从环境中提取各 agent 的 profile/goal 文本。"""
    profiles: dict[str, str] = {}
    agents_map = getattr(env, "agents", {}) or {}
    for role in agent_roles:
        agent = agents_map.get(role)
        if agent is None:
            profiles[role] = f"Role: {role}"
            continue
        parts: list[str] = []
        # goal
        goal = str(getattr(agent, "goal", "") or "").strip()
        if goal:
            parts.append(f"Goal: {goal}")
        # personality / background if available
        bg = str(getattr(agent, "background", "") or "").strip()
        if bg:
            parts.append(f"Background: {bg}")
        if not parts:
            parts.append(f"Role: {role}")
        profiles[role] = "\n".join(parts)
    return profiles


@dataclass(frozen=True)
class DimensionEvalResult:
    """一次双维度评测的完整结果。"""

    terminal: str
    agent_evaluations: dict[str, dict[str, Any]]
    aggregate_stats: dict[str, float]

    @property
    def summary_text(self) -> str:
        lines = [f"terminal={self.terminal}"]
        for dim in ("persona_style_consistency", "goal_behavioral_competence"):
            mean_key = f"{dim}_mean"
            if mean_key in self.aggregate_stats:
                lines.append(
                    f"  {dim}: mean={self.aggregate_stats[mean_key]:.2f} "
                    f"min={self.aggregate_stats.get(f'{dim}_min', 0):.0f} "
                    f"max={self.aggregate_stats.get(f'{dim}_max', 0):.0f}"
                )
        return "\n".join(lines)


def compute_dimension_statistics(
    evaluations: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """对双维度评测结果计算统计摘要。

    Args:
        evaluations: ``{agent_key: {"persona_style_consistency": {"score": int, "reasoning": str},
                                     "goal_behavioral_competence": {"score": int, "reasoning": str}}}``

    Returns:
        包含各维度 mean/min/max 的扁平 dict。
    """
    dims = ("persona_style_consistency", "goal_behavioral_competence")
    stats: dict[str, float] = {"n_agents_evaluated": float(len(evaluations))}
    for dim in dims:
        scores = []
        for agent_key, ev in evaluations.items():
            dim_data = ev.get(dim, {})
            s = dim_data.get("score")
            if isinstance(s, (int, float)):
                scores.append(float(s))
        if scores:
            stats[f"{dim}_mean"] = sum(scores) / len(scores)
            stats[f"{dim}_min"] = min(scores)
            stats[f"{dim}_max"] = max(scores)
            stats[f"{dim}_sum"] = sum(scores)
        else:
            stats[f"{dim}_mean"] = 0.0
            stats[f"{dim}_min"] = 0.0
            stats[f"{dim}_max"] = 0.0
            stats[f"{dim}_sum"] = 0.0
    return stats


async def run_two_dimension_evaluation(
    *,
    model_name: str,
    history: str,
    agent_roles: list[str],
    agent_profiles: dict[str, str] | None = None,
    agent_display_names: dict[str, str] | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> TwoDimensionEvaluationResult | None:
    """对一组 agent 的交互历史执行双维度 LLM 评测。

    Args:
        model_name: LiteLLM 模型名称。
        history: 格式化的交互历史文本（如 ``format_negotiation_episode_for_llm_eval`` 的输出）。
        agent_roles: 参与评测的 agent 角色列表（如 ``["firm_a", "firm_b"]``）。
        agent_profiles: 每个 role 的 profile/goal 文本。若为 None 则仅使用 role 名。
        agent_display_names: 每个 role 的显示名称。
        temperature: LLM 采样温度。
        max_retries: 最大重试次数。

    Returns:
        ``TwoDimensionEvaluationResult`` 或 None（全部重试失败时）。
    """
    if not agent_roles:
        raise ValueError("agent_roles must not be empty")

    profiles = agent_profiles or {r: f"Role: {r}" for r in agent_roles}
    profiles_text = _build_agent_profiles_text(
        agent_roles, profiles, agent_display_names
    )
    key_instruction = _build_agent_key_instruction(len(agent_roles))

    schema_echo_warn = (
        "\n\nIMPORTANT: Do NOT echo the JSON Schema. "
        "The response must be CONCRETE evaluation data filling the schema, "
        "NOT the schema definition itself. "
        "It must NOT contain any of these JSON Schema keywords: "
        '"$ref", "additionalProperties", "title": "Evaluations", '
        '"type": "object" at the top of \'evaluations\'. '
        "Each agent key must contain real numeric scores and reasoning strings."
    )

    last_exc: Exception | None = None

    for attempt in range(max_retries):
        extra = "" if attempt == 0 else schema_echo_warn
        try:
            response = await agenerate(
                model_name=model_name,
                template=EVALUATION_USER_PROMPT_TEMPLATE,
                input_values=dict(
                    agent_profiles=profiles_text,
                    history=history,
                    agent_key_instruction=key_instruction,
                    extra=extra,
                    agent="two_dimension_evaluator",
                ),
                output_parser=PydanticOutputParser[TwoDimensionEvaluationResult](
                    pydantic_object=TwoDimensionEvaluationResult
                ),
                temperature=temperature,
                structured_output=True,
            )
            # 二次校验
            keys = set(response.evaluations.keys())
            schema_keywords = {
                "additionalProperties",
                "$ref",
                "title",
                "type",
                "properties",
            }
            if keys & schema_keywords:
                raise ValueError(
                    f"Schema echo detected: evaluations keys {keys}"
                )
            return response
        except Exception as e:
            last_exc = e
            log.debug(
                f"[two_dim_eval] attempt {attempt + 1}/{max_retries} failed: {e}"
            )

    log.warning(
        f"[two_dim_eval] all {max_retries} attempts failed: {last_exc}"
    )
    return None


def evaluate_two_dimension_sync(
    model_name: str,
    history: str,
    agent_roles: list[str],
    **kwargs: Any,
) -> TwoDimensionEvaluationResult | None:
    """同步封装。"""
    return asyncio.run(
        run_two_dimension_evaluation(
            model_name=model_name,
            history=history,
            agent_roles=agent_roles,
            **kwargs,
        )
    )


# ============================================================================
# 与现有 LongTermNegotiationEnv 集成的便捷入口
# ============================================================================


async def evaluate_negotiation_episode_two_dim(
    model_dict: dict[str, str],
    env: Any,
    *,
    history: str | None = None,
    max_action_log: int | None = 500,
) -> DimensionEvalResult | None:
    """对已完成的 negotiation episode 执行双维度评测。

    与 ``run_llm_negotiation_episode_evaluation`` 独立，可在 episode 结束后单独调用。

    Args:
        model_dict: 包含 ``"env"`` key 的模型字典。
        env: ``LongTermNegotiationEnv`` 实例（episode 已完成）。
        history: 预格式化的历史文本；若为 None 则从 env 自动构建。
        max_action_log: 传入 ``format_negotiation_episode_for_llm_eval`` 的最大 action 条目数。

    Returns:
        ``DimensionEvalResult`` 或 None。
    """
    from .llm_evaluation import format_negotiation_episode_for_llm_eval

    if "env" not in model_dict:
        raise KeyError("model_dict must contain key 'env' for the evaluator model.")

    if history is None:
        history = format_negotiation_episode_for_llm_eval(
            env, max_action_log=max_action_log
        )

    ctrl = env.ctrl
    roster = list(ctrl.agent_names)
    profiles = _extract_agent_profiles(env, roster)
    dnames: dict[str, str] = getattr(env, "agent_display_names", {}) or {}

    result = await run_two_dimension_evaluation(
        model_name=model_dict["env"],
        history=history,
        agent_roles=roster,
        agent_profiles=profiles,
        agent_display_names=dnames,
    )

    if result is None:
        return None

    evaluations_dict: dict[str, dict[str, Any]] = {}
    for agent_key, ev in result.evaluations.items():
        evaluations_dict[agent_key] = {
            "persona_style_consistency": {
                "score": ev.persona_style_consistency.score,
                "reasoning": ev.persona_style_consistency.reasoning,
            },
            "goal_behavioral_competence": {
                "score": ev.goal_behavioral_competence.score,
                "reasoning": ev.goal_behavioral_competence.reasoning,
            },
        }

    stats = compute_dimension_statistics(evaluations_dict)
    terminal = str(getattr(ctrl, "terminal", "") or "")

    return DimensionEvalResult(
        terminal=terminal,
        agent_evaluations=evaluations_dict,
        aggregate_stats=stats,
    )


# ============================================================================
# 格式化输出
# ============================================================================


def format_dimension_eval_report(result: DimensionEvalResult) -> str:
    """将 ``DimensionEvalResult`` 格式化为可读报告。"""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("Two-Dimension Evaluation Report")
    lines.append("=" * 64)
    lines.append(f"Terminal: {result.terminal}")
    lines.append(f"Agents evaluated: {result.aggregate_stats.get('n_agents_evaluated', 0):.0f}")
    lines.append("")

    dims = [
        ("persona_style_consistency", "Persona & Language Style Consistency"),
        ("goal_behavioral_competence", "Goal Achievement & Behavioral Competence"),
    ]
    for dim_key, dim_label in dims:
        lines.append(f"--- {dim_label} ---")
        mean = result.aggregate_stats.get(f"{dim_key}_mean", 0)
        mn = result.aggregate_stats.get(f"{dim_key}_min", 0)
        mx = result.aggregate_stats.get(f"{dim_key}_max", 0)
        lines.append(f"  Mean: {mean:.2f}  |  Min: {mn:.0f}  |  Max: {mx:.0f}")
        lines.append("")

    lines.append("--- Per-Agent Scores ---")
    for agent_key, ev in result.agent_evaluations.items():
        lines.append(f"  {agent_key}:")
        for dim_key, dim_label in dims:
            dim_data = ev.get(dim_key, {})
            score = dim_data.get("score", "-")
            lines.append(f"    {dim_label}: {score}")
        lines.append("")

    lines.append("--- Detailed Reasoning ---")
    for agent_key, ev in result.agent_evaluations.items():
        lines.append(f"\n[{agent_key}]")
        for dim_key, dim_label in dims:
            dim_data = ev.get(dim_key, {})
            reasoning = dim_data.get("reasoning", "(no reasoning)")
            lines.append(f"  --- {dim_label} ---")
            lines.append(f"  Score: {dim_data.get('score', '-')}")
            lines.append(f"  Reasoning: {reasoning}")
    lines.append("\n" + "=" * 64)
    return "\n".join(lines)


__all__ = [
    "AgentTwoDimensionEvaluation",
    "DimensionEvalResult",
    "EVALUATION_SYSTEM_PROMPT_EN",
    "EVALUATION_USER_PROMPT_TEMPLATE",
    "GoalBehavioralScore",
    "PersonaStyleScore",
    "TwoDimensionEvaluationResult",
    "compute_dimension_statistics",
    "evaluate_negotiation_episode_two_dim",
    "evaluate_two_dimension_sync",
    "format_dimension_eval_report",
    "run_two_dimension_evaluation",
]
