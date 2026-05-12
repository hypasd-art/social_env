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
You are **{agent}** in a **long-horizon business negotiation** simulator (multi-day calendar, formal JSON moves).

## Priority (read in order)
1. **Latest Environment turn** in the history below: it states the current phase (scheduling vs active session),
   allowed moves, and often **literal JSON examples** for this step. Treat that text as authoritative.
2. **Your private goal** appears under "Here is the context of the interaction" / goal lines — only you see your side's goal text.
3. **Role & roster hints** (supplement only; do not contradict the Environment):
{action_instructions}

## Action types (must pick one of ``{action_list}`` for this turn)
- **speak** — In-session natural language; ``argument`` = short dialogue string (no JSON).
- **non-verbal communication** — ``argument`` = short string.
- **action** — Structured move; ``argument`` MUST be a **plain JSON object** (Python dict / mapping), **not** a string that contains JSON, **not** wrapped in markdown fences. Keys such as ``negotiation_op``, ``verb``, ``terms``, ``contract_id``, ``proposed_participants``, ``purpose``, ``accept``, etc. must **match names and nesting** shown in the Environment message for **this** turn (scheduling blocks vs "Active session" blocks differ).
- **leave** — Only if the Environment explicitly allows a top-level leave; otherwise use **action** with the ``session_control`` / ``leave`` payload the Environment describes.
- **none** — Only when the Environment makes clear that skipping is correct; avoid lazy ``none`` if a substantive move is expected.

## JSON discipline (for ``action``)
- Re-use **exact** ``negotiation_op`` / ``verb`` tokens from the Environment; do not invent new op names.
- Omit keys you do not need; do not paste schema ``description`` / ``type`` / ``$defs`` text into ``argument``.
- If unsure between two valid moves, prefer the smallest valid payload that still advances your goal and respects calendars/session rules.

## Behaviour
Stay in character, pursue your goal, and keep dialogue concise and non-repetitive relative to other participants' lines.

--- Interaction history (newest relevant context is near the end) ---
{history}

--- Turn index ---
You are at **Turn #{turn_number}** (environment counter). Available action types this turn: **{action_list}**.

Output: a single JSON object matching the schema below (action type + argument + ``to`` list):
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
        parts = [
            f"- Your canonical id: {self.agent_name!r} (use this exact token when the protocol names actors).",
            f"- Role summary: {role_line}",
        ]
        if self._all_participant_names:
            roster = ", ".join(repr(n) for n in self._all_participant_names)
            parts.append(f"- Episode roster (canonical participants): {roster}")
        if extra:
            parts.append(f"- Scenario-specific goal / constraints: {extra}")
        if obs.action_instruction.strip():
            parts.append(
                "- Environment ``action_instruction`` (high-priority hint for this observation): "
                f"{obs.action_instruction.strip()}"
            )
        return "\n".join(parts)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][1].to_natural_language(),
                agent=self.agent_name,
            )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="", to=[])

        mem_block = await self.memory.arecent(self.memory_inject_lines)
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
        # breakpoint()
        self.memory.add(
            f"T{obs.turn_number} [{self.agent_name}] {action.to_natural_language()}"
        )
        return action


def build_negotiation_social_llm_agents(
    model_dict: dict[str, str],
    roster: tuple[str, ...],
    *,
    memory_summary_model: str | None = None,
    social_memory_kwargs: dict[str, Any] | None = None,
) -> dict[str, NegotiationSocialLLMAgent]:
    """与 ``minimalist_demo`` / ``llm_evaluation`` 一致：`agent1`…`agentN` 对齐 ``roster`` 顺序。

    记忆相关参数由 ``negotiation_run_config``（``--run-config`` JSON）经
    ``build_negotiation_agents_from_run_config`` 注入；也可在代码里显式传入
    ``memory_summary_model`` / ``social_memory_kwargs`` 覆盖 ``SocialLLMAgent`` 记忆行为。
    """
    n = len(roster)
    mem_kw = dict(social_memory_kwargs or {})
    if memory_summary_model is not None:
        mem_kw["memory_summary_model"] = memory_summary_model
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
            **mem_kw,
        )
        summary = ROLE_SUMMARY_EN.get(role, role)
        ag.goal = (
            f"[{role}] {summary}\n"
            "Operate strictly inside the simulator protocol: each turn, read the latest Environment "
            "message for allowed action types and JSON shapes; when you use ``action``, the ``argument`` "
            "must be a JSON object matching that message (never a quoted JSON string or markdown). "
            "Advance your interests without breaking calendar/session rules or inventing negotiation_op names."
        )
        agents[role] = ag
    return agents


__all__ = [
    "NEGOTIATION_LLM_CUSTOM_TEMPLATE",
    "NegotiationSocialLLMAgent",
    "build_negotiation_social_llm_agents",
]
