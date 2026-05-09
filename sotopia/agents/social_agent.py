"""面向 ``SocialSystemEnv`` 的 LLM Agent：附加短时情景记忆与可选目标补丁。"""

from __future__ import annotations

from typing import Any

from sotopia.agents.llm_agent import LLMAgent
from sotopia.agents.memory import EpisodicMemory
from sotopia.generation_utils.generate import agenerate_action, agenerate_goal, fill_template
from sotopia.messages import AgentAction, Observation


class SocialLLMAgent(LLMAgent):
    """在 ``LLMAgent`` 上叠加 ``EpisodicMemory``；每轮把最近若干条写入 prompt 上下文。"""

    def __init__(
        self,
        *args: Any,
        memory_max: int = 40,
        memory_inject_lines: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory = EpisodicMemory(max_entries=memory_max)
        self.memory_inject_lines = memory_inject_lines

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

        custom_template = None
        if self.custom_template:
            custom_template = fill_template(
                self.custom_template, action_instructions=obs.action_instruction
            )

        agent_names: list[str] | None = (
            self.script_background.agent_names
            if self.script_background is not None
            else None
        )

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
        self.memory.add(
            f"T{obs.turn_number} [{self.agent_name}] {action.to_natural_language()}"
        )
        return action


__all__ = ["SocialLLMAgent"]
