import abc
import logging
from collections import defaultdict
from typing import Any, Generic, TypeVar

import gin
from pydantic import BaseModel, validate_call

from litellm.utils import supports_response_schema

from sotopia.generation_utils import (
    PydanticOutputParser,
    agenerate,
)
from sotopia.database import LLMEvalBaseModel
from sotopia.messages import (
    AgentAction,
    Message,
    ScriptEnvironmentResponse,
)

log = logging.getLogger("evaluators")

T_eval_dim = TypeVar("T_eval_dim", bound=BaseModel)


class EvaluationForAgents(LLMEvalBaseModel, Generic[T_eval_dim]):
    evaluations: dict[str, T_eval_dim]


class Evaluator(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError


class SocialGameEndEvaluator(Evaluator):
    """Base evaluator for social game win conditions.

    Subclasses should implement _check_win_conditions() to check
    game-specific win conditions using the environment state.
    """

    def __init__(self, max_turn_number: int = 100) -> None:
        self.max_turn_number = max_turn_number

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Check turn limit
        if turn_number >= self.max_turn_number:
            return [("environment", (("terminated", True), "Max turns reached"))]

        # Extract environment from kwargs
        env = kwargs.get("env")
        if not env:
            return [("environment", (("terminated", False), ""))]

        # Check game-specific win conditions
        terminated, reason = self._check_win_conditions(env, turn_number, messages)
        return [("environment", (("terminated", terminated), reason))]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return self.__call__(turn_number, messages, **kwargs)

    def _check_win_conditions(
        self, env: Any, turn_number: int, messages: list[tuple[str, Message]]
    ) -> tuple[bool, str]:
        """Check game-specific win conditions. Override in subclasses."""
        return False, ""


class RuleBasedTerminatedEvaluator(Evaluator):
    def __init__(self, max_turn_number: int = 20, max_stale_turn: int = 2) -> None:
        self.max_turn_number = max_turn_number
        self.max_stale_turn = max_stale_turn

    @validate_call
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number >= self.max_turn_number
        # Rule 2: If fewer than two agents remain active (not left), terminate
        # Determine latest action per agent, and count those whose latest is not "leave"
        latest_action_by_agent: dict[str, str] = {}
        observed_agents: set[str] = set()
        for speaker, msg in messages:
            if speaker != "Environment":
                observed_agents.add(speaker)

        for speaker, msg in messages[::-1]:
            if speaker == "Environment":
                continue
            if not isinstance(msg, AgentAction):
                continue
            if speaker not in latest_action_by_agent:
                latest_action_by_agent[speaker] = msg.action_type

        # If we haven't observed any agent messages yet, do not terminate early
        env = kwargs.get("env")
        if env:
            all_agents = set(env.agents)
        else:
            all_agents = observed_agents

        if all_agents:
            num_active_agents = sum(
                1
                for agent in all_agents
                if latest_action_by_agent.get(agent, "speak") != "leave"
            )
        else:
            num_active_agents = 2

        too_few_agents = num_active_agents < 2
        # Rule 3: If the conversation is stale for too long, terminate the conversation
        stale_count = 0
        for message in messages[::-1]:
            if message[0] == "Environment":
                continue
            assert isinstance(message[1], AgentAction)
            if message[1].action_type == "none":
                stale_count += 1
            else:
                break
            if stale_count > self.max_stale_turn:
                break
        stale_too_long = stale_count > self.max_stale_turn
        terminated = conversation_too_long or too_few_agents or stale_too_long
        reasons_for_termination = (
            f"{'The conversation is too long; ' if conversation_too_long else ''}"
            f"{'Too few active agents; ' if too_few_agents else ''}"
            f"{'The conversation stales for too long; ' if stale_too_long else ''}"
        )
        return [
            (
                "environment",
                (("terminated", terminated), reasons_for_termination),
            )
        ]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return self(turn_number, messages, **kwargs)


class EpisodeLLMEvaluator(Evaluator, Generic[T_eval_dim]):
    def __init__(
        self,
        model_name: str,
        response_format_class: type[EvaluationForAgents[T_eval_dim]],
    ) -> None:
        self.model_name = model_name
        self.prompt = ""
        self.response_format_class = response_format_class

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]], **kwargs: Any
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError(
            "ReachGoalLLMEvaluator is not implemented for synchronous evaluation"
        )

    @gin.configurable
    @validate_call
    async def __acall__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]] | None,
        history: str = "",
        temperature: float | None = 0.0,
        **kwargs: Any,
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # 中文注释：该评估器在“整段历史”上做一次 LLM 评分，
        # 输出按 agent × 维度展开后的扁平结果，供环境统一聚合。
        # filter did nothing
        if not history and messages:
            messages_filtered = [
                (x, y)
                for x, y in messages
                if "did nothing" not in y.to_natural_language()
            ]
            history = "\n".join(
                [
                    (
                        f"{x} {y.to_natural_language()}"
                        if x != "Environment"
                        else y.to_natural_language()
                    )
                    for x, y in messages_filtered
                ]
            )

        try:
            # 中文注释：动态统计参与者数量，避免多智能体场景下 key 数不匹配。
            # Count actual participating agents (exclude Environment)
            participating_agents = set()
            if messages:
                for speaker, _ in messages:
                    if speaker != "Environment":
                        participating_agents.add(speaker)
            num_agents = len(participating_agents)

            # 中文注释：明确要求模型使用固定 key（agent_1 ... agent_n），
            # 降低 structured output 中动态键名带来的解析歧义。
            # Build explicit agent label instruction to avoid ambiguous dynamic keys in structured output
            agent_instruction = ""
            if num_agents > 0:
                agent_instruction = (
                    "There are exactly "
                    + str(num_agents)
                    + " agents. Under the 'evaluations' field, use exactly these keys: "
                    + "["
                    + ", ".join([f'"agent_{i+1}"' for i in range(num_agents)])
                    + "] (no other keys).\n"
                )

            # 中文注释：优先走结构化输出；若模型不支持则降级为普通文本+解析。
            # Use structured output if model supports it (not just custom/structured endpoints)
            use_structured_output = self.model_name.startswith(
                "custom/structured"
            ) or supports_response_schema(model=self.model_name)

            # 中文注释：尝试最多 3 次：第 1 次正常请求；第 2/3 次额外注入
            # "禁止返回 schema、必须填具体数据" 的提示，缓解 GPT-5 系列把
            # response_format schema 当成数据原样回复的退化模式。
            schema_echo_warn = (
                "\n\nIMPORTANT: Do NOT echo the JSON Schema. "
                "The response must be CONCRETE evaluation data filling "
                "the schema, NOT the schema definition itself. "
                "It must NOT contain any of these JSON Schema keywords: "
                '"$ref", "additionalProperties", "title": "Evaluations", '
                '"type": "object" at the top of \'evaluations\'. '
                "Each agent key must contain real numeric scores and reasoning strings."
            )

            last_exc: Exception | None = None
            response: EvaluationForAgents[T_eval_dim] | None = None
            for attempt in range(3):
                extra = "" if attempt == 0 else schema_echo_warn
                try:
                    response = await agenerate(
                        model_name=self.model_name,
                        template="""{history}
                            Based on previous interactions, evaluate how well participants achieve their goals.
                            {agent_instruction}
                            Please follow the format:
                            {format_instructions}
                            {extra}
                        """,
                        input_values=dict(
                            history=history,
                            agent_instruction=agent_instruction,
                            extra=extra,
                        ),
                        output_parser=PydanticOutputParser[self.response_format_class](  # type: ignore[name-defined]
                            pydantic_object=self.response_format_class
                        ),
                        temperature=temperature,
                        structured_output=use_structured_output,
                    )
                    # 二次校验：检测 schema-echo 退化（attempt0 也走这一关）
                    keys = set(response.evaluations.keys())  # type: ignore[union-attr]
                    schema_keywords = {
                        "additionalProperties",
                        "$ref",
                        "title",
                        "type",
                        "properties",
                    }
                    if keys & schema_keywords:
                        raise ValueError(
                            f"Schema echo detected: evaluations keys {keys}; "
                            f"will retry with stronger instruction"
                        )
                    break  # 成功
                except Exception as e:
                    last_exc = e
                    log.debug(
                        f"[evaluator] attempt {attempt + 1}/3 failed: {e}"
                    )
                    response = None

            if response is None:
                raise last_exc or RuntimeError("Evaluator failed after 3 attempts")
            response_list = []
            # 中文注释：只消费真实参与 agent 的评估，避免越界或脏键。
            # Only process evaluations for the actual number of agents.
            # 中文注释：每个维度（如 believability）现在是
            #     {"reasoning": str, "score": int}
            # 老代码用 [1]/[0] 当 score/reasoning 索引，会抛 KeyError(1)
            # 被外层 except 静默吞掉，导致 evaluator 输出全部丢失，
            # 下游 p1_rate/p2_rate=None → rewards=[0,0] → 整局被 quarantine。
            for i, evaluation in enumerate(
                list(response.evaluations.values())[:num_agents]
            ):
                agent_key = f"agent_{i+1}"
                dump = evaluation.model_dump()
                for dimension, value in dump.items():
                    if isinstance(value, dict):
                        score = value.get("score", 0)
                        reasoning = value.get("reasoning", "")
                    elif isinstance(value, (list, tuple)) and len(value) >= 2:
                        # 兼容老 schema：(reasoning, score)
                        reasoning, score = value[0], value[1]
                    else:
                        score, reasoning = 0, str(value)
                    response_list.append(
                        (
                            agent_key,
                            (
                                (dimension, score),
                                reasoning,
                            ),
                        )
                    )
            # print(f"response_list: {response_list}")
            return response_list
        except Exception as e:
            log.warning(
                f"[evaluator] Failed to convert LLM evaluation into response_list: "
                f"{type(e).__name__}: {e}"
            )
            return []


@validate_call
def _reduce(
    responses_per_reducer: list[tuple[tuple[str, float | int | bool], str]],
) -> tuple[dict[str, float | int | bool], str]:
    responses_dict = defaultdict(list)
    comments_dict: dict[str, str] = defaultdict(str)
    reduced_dict: dict[str, float | int | bool] = {}
    for response, reasoning in responses_per_reducer:
        responses_dict[response[0]].append(response[1])
        comments_dict[response[0]] += reasoning
    scores: list[float | int] = []
    for k, v in responses_dict.items():
        if k == "terminated":
            assert all([isinstance(x, bool) for x in v])
            reduced_dict[k] = any(v)
        else:
            assert all([isinstance(x, (float, int)) for x in v])
            reduced_dict[k] = sum(v) / len(v)
            scores.append(reduced_dict[k])
    if len(scores) and "overall_score" not in responses_dict:
        scores = [x for x in scores if x is not None]
        reduced_dict["overall_score"] = sum(scores) / len(scores)
    comments = "\n".join([f"{k}: {v}" for k, v in comments_dict.items()])
    return reduced_dict, comments


@validate_call
def unweighted_aggregate_evaluate(
    responses: list[tuple[str, tuple[tuple[str, int | float | bool], str]]],
) -> ScriptEnvironmentResponse:
    """
    Aggregate the responses from the environment

    Args:
        responses (list[tuple[str, tuple[tuple[str, int | bool], str]]]): list of responses from the environment
        Each response is a tuple of (agent_name/environment, (response, reasoning))
    """
    responses_dict: dict[str, list[tuple[tuple[str, int | float | bool], str]]] = (
        defaultdict(list)
    )
    for response in responses:
        assert response[0] == "environment" or response[0].startswith("agent")
        responses_dict[response[0]].append(response[1])

    environment_responses: tuple[dict[str, float | int | bool], str] = ({}, "")
    agent_responses: dict[str, tuple[dict[str, float | int | bool], str]] = {}

    for k, v in responses_dict.items():
        if k == "environment":
            environment_responses = _reduce(v)
        else:
            # Support any number of agents (agent_1, agent_2, agent_3, etc.)
            agent_responses[k] = _reduce(v)

    # Build comments from all agents dynamically
    agent_comments = ""
    for agent_key, (_, comment) in agent_responses.items():
        if comment:
            agent_name = agent_key.replace("_", " ").title()
            agent_comments += f"{agent_name} comments:\n{comment}\n"

    comments = (
        f"Environment comments: {environment_responses[1]}\n"
        if environment_responses[1]
        else ""
    ) + agent_comments
    if (
        "terminated" in environment_responses[0]
        and environment_responses[0]["terminated"]
    ):
        log.debug(f"[green] The conversation is terminated. {response}")
    # Get first two agents for backward compatibility with ScriptEnvironmentResponse
    agent_1_responses = agent_responses.get("agent_1", ({}, ""))
    agent_2_responses = agent_responses.get("agent_2", ({}, ""))

    return ScriptEnvironmentResponse(
        terminated=environment_responses[0]["terminated"]
        if "terminated" in environment_responses[0]
        else False,
        p1_rate=(
            agent_1_responses[0]["overall_score"]
            if "overall_score" in agent_1_responses[0]
            else 0,
            agent_1_responses[0],
        )
        if agent_1_responses != ({}, "")
        else None,
        p2_rate=(
            agent_2_responses[0]["overall_score"]
            if "overall_score" in agent_2_responses[0]
            else 0,
            agent_2_responses[0],
        )
        if agent_2_responses != ({}, "")
        else None,
        comments=comments,
    )
