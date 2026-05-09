"""长期谈判场景的大模型参与者：继承 ``SocialLLMAgent``，并注入 **可用 JSON 动作** 的提示模板。

``NegotiationWorldController`` 已在 ``Observation.last_turn`` 中写明各阶段 JSON 形态；本类通过
``custom_template`` 与 ``{action_instructions}`` 强调：在 ``action_type="action"`` 时
``argument`` 必须是可解析的结构化对象（``negotiation_op`` / ``verb`` / ``terms`` 等），
与 ``agenerate_action(..., structured_output=True)`` 对齐。

推荐导入（避免在仅加载规则 agent 的路径上引入本模块）::

    from sotopia.settings.long_term_negotiation.negotiation_llm_agent import (
        NegotiationSocialLLMAgent,
        NEGOTIATION_LLM_CUSTOM_TEMPLATE,
    )

**评测链：** ``build_negotiation_social_llm_agents``（本模块末尾）由
``llm_evaluation.build_llm_negotiation_agents`` / ``run_llm_negotiation_episode_evaluation`` 在
构造 ``LongTermNegotiationEnv`` 之前调用。
"""

from __future__ import annotations

from typing import Any

from sotopia.agents.social_agent import SocialLLMAgent
from sotopia.generation_utils.generate import agenerate_action, agenerate_goal, fill_template
from sotopia.messages import AgentAction, Observation

from .roles import ROLE_SUMMARY_EN

# 与 ``agenerate_action`` 默认模板变量一致：由 ``agenerate`` 注入 goal / format_instructions 等。
NEGOTIATION_LLM_CUSTOM_TEMPLATE = """
You are playing one turn in a **long-term business negotiation** simulator: calendar scheduling
(§3) and formal session moves (§5–§6) with strict JSON tool payloads.

How to use ``action_type`` and ``argument``:
- ``speak``: ``argument`` is a short natural-language string (in-session dialogue).
- ``non-verbal communication``: ``argument`` is a short string.
- ``action``: ``argument`` MUST be a **JSON object** (mapping), not a quoted JSON string. Include fields
  such as ``negotiation_op``, ``verb``, ``terms``, ``requester``, ``accept``, ``proposed_participants``,
  ``purpose``, ``contract_id``, etc. **Copy field names and shapes from the latest Environment message**
  in the interaction history (the block that starts with "Scheduling —" or "Active session").
- ``leave``: use when the rules in the Environment text allow leaving the session as a top-level move;
  if the Environment instead asks for ``session_control`` + ``leave`` inside JSON, prefer ``action`` with that payload.
- ``none``: pass / no-op only when truly appropriate.

Never invent negotiation_op names; follow the protocol text in the history for **this** turn.

Role & design hints (not a substitute for the Environment JSON examples):
{action_instructions}

Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
Note that {agent}'s goal is only visible to you.
You should try your best to achieve {agent}'s goal in a way that aligns with their character traits.
Additionally, maintaining naturalness is essential (e.g., avoid repeating others verbatim without reason).
{history}
You are at Turn #{turn_number}. Your available action types are {action_list}.

Please only generate a JSON output that matches the following format instructions (action type + argument + to):
{format_instructions}
"""


class NegotiationSocialLLMAgent(SocialLLMAgent):
    """在长周期短时记忆通路（``SocialLLMAgent``）上叠加谈判动作协议提示。"""

    def __init__(
        self,
        *args: Any,
        all_participant_names: list[str] | None = None,
        role_goal_addon: str = "",
        negotiation_prompt_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        tpl = negotiation_prompt_template or NEGOTIATION_LLM_CUSTOM_TEMPLATE
        kwargs["custom_template"] = tpl
        super().__init__(*args, **kwargs)
        self._all_participant_names = list(all_participant_names) if all_participant_names else None
        self._role_goal_addon = role_goal_addon
        self._negotiation_prompt_template = tpl

    def _action_instruction_block(self, obs: Observation) -> str:
        role_line = ROLE_SUMMARY_EN.get(self.agent_name, self.agent_name)
        extra = self._role_goal_addon.strip()
        parts = [f"- Your id: {self.agent_name!r}. Role summary: {role_line}"]
        if extra:
            parts.append(f"- Additional goal / constraints: {extra}")
        if obs.action_instruction.strip():
            parts.append(f"- Environment action_instruction field: {obs.action_instruction.strip()}")
        return "\n".join(parts)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][1].to_natural_language(),
            )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="", to=[])

        mem_block = self.memory.recent(self.memory_inject_lines)
        goal_effective = self._goal
        if mem_block:
            goal_effective = (
                (self._goal or "")
                + "\n\n[Recent episode memory — use for long-horizon consistency]\n"
                + mem_block
            )

        custom_template = fill_template(
            self._negotiation_prompt_template,
            action_instructions=self._action_instruction_block(obs),
        )

        agent_names = self._all_participant_names
        if agent_names is None and self.script_background is not None:
            agent_names = self.script_background.agent_names

        action = await agenerate_action(
            self.model_name,
            history="\n".join(f"{y.to_natural_language()}" for _, y in self.inbox),
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self.agent_name,
            goal=goal_effective,
            script_like=self.script_like,
            custom_template=custom_template,
            structured_output=True,
            agent_names=agent_names,
            sender=self.agent_name,
        )
        breakpoint()
        self.memory.add(
            f"T{obs.turn_number} [{self.agent_name}] {action.to_natural_language()}"
        )
        return action


def build_negotiation_social_llm_agents(
    model_dict: dict[str, str],
    roster: tuple[str, ...],
) -> dict[str, NegotiationSocialLLMAgent]:
    """与 ``minimalist_demo`` / ``llm_evaluation`` 一致：`agent1`…`agentN` 对齐 ``roster`` 顺序。"""
    n = len(roster)
    for i in range(n):
        key = f"agent{i + 1}"
        if key not in model_dict:
            raise KeyError(
                f"model_dict must contain key {key!r} for roster of size {n}; "
                f"expected keys agent1..agent{n}. Got roster={roster}."
            )
    participants = list(roster)
    agents: dict[str, NegotiationSocialLLMAgent] = {}
    for idx, role in enumerate(roster):
        mname = model_dict[f"agent{idx + 1}"]
        ag = NegotiationSocialLLMAgent(
            agent_name=role,
            model_name=mname,
            all_participant_names=participants,
        )
        summary = ROLE_SUMMARY_EN.get(role, role)
        ag.goal = (
            f"[{role}] {summary}\n"
            "You negotiate under the environment protocol: use only allowed action types and "
            "negotiation payloads exactly as described in each Environment observation."
        )
        agents[role] = ag
    return agents


__all__ = [
    "NEGOTIATION_LLM_CUSTOM_TEMPLATE",
    "NegotiationSocialLLMAgent",
    "build_negotiation_social_llm_agents",
]
