"""长期谈判场景的大模型参与者：继承 ``SocialLLMAgent``，并注入 **可用 JSON 动作** 的提示模板。

``NegotiationWorldController`` 已在 ``Observation.last_turn`` 中写明各阶段 JSON 形态；本类通过
``custom_template`` 与 ``{action_instructions}`` 强调：在 ``action_type="action"`` 时
``argument`` 必须是可解析的结构化对象（``negotiation_op`` / ``verb`` / ``terms`` 等），
与 ``agenerate_action(..., structured_output=True)`` 对齐。``{goal}`` 置于模板 **末尾**（JSON schema 之前），
含私密目标、情景记忆、**[agent_design_digest]** 设计摘要（与 ``llm_evaluation`` 注入块互补）。

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

import re
from collections.abc import Mapping
from typing import Any

from sotopia.agents.social_agent import SocialLLMAgent
from sotopia.generation_utils.generate import agenerate_action, agenerate_goal, fill_template
from sotopia.messages import AgentAction, Observation
from sotopia.utils import truncate_chars

from .roles import ROLE_PERSONA_EN, ROLE_SUMMARY_EN, default_display_name_for_role

# 与 ``agenerate_action`` 默认模板变量一致：由 ``agenerate`` 注入 goal / format_instructions 等。
NEGOTIATION_LLM_CUSTOM_TEMPLATE = """
You are **{agent}** in a long-horizon negotiation simulator (calendar slots + structured JSON moves).

## Authority stack (higher wins)
1. **Latest Environment message** in the history block: phase, allowed moves, literal JSON examples for **this** turn — follow it mechanically.
2. **Live state in that message** (`[system]`, contracts, session bookkeeping): current numbers and legality; overrides stale narrative.
3. **Private block at the end** (section titled ``Here is the context of the interaction`` below): `[persona]`, DialogueVoice, `[agent_design_digest]`, `[design_economics]`, loaded profiles / relationships — use for **voice, motives, stance** only; never contradict (1–2).
4. **Role & roster** bullets below — hints only.

## Visibility
- `[system]` / `[contracts_visible_to_you]` = your view only.
- `[other_participants_public_role]` = public one-liners; **not** their private cash, thresholds, or hidden goals.

## Persona & voice (from private block)
- Match **DialogueVoice** / `[persona]` for register, pacing, hedges, and taboos; keep each **speak** short and distinct from other roster members.
- In **speak**, address people only by **personal names** (as in Environment / roster); never say internal roster codes aloud.
- In **action** JSON, use the **same personal name strings** as in this turn's Environment examples for any participant fields (`proposed_participants`, `requester`, `receiver`, etc.).

## Action types (choose exactly one; must appear in **{action_list}**)
- **speak** — `argument`: short in-character text (no JSON).
- **non-verbal communication** — `argument`: short string.
- **action** — `argument`: **one** JSON object matching this turn’s Environment (exact `negotiation_op` / `verb` names). No markdown fences; no JSON-as-string.
- **leave** — only if Environment allows that escape; else use **action** with the described `session_control` / `leave` shape.
- **none** — only when Environment clearly expects silence; avoid lazy **none** if a move is due.

## JSON discipline (**action** only)
- Copy token names from the Environment; do not invent ops.
- Omit unused keys; never paste schema `description` / `type` / `$defs` into `argument`.
- If two moves are valid, prefer the **smaller** payload that still respects calendars/session caps.

## Silent pre-flight (do not print)
- [ ] `action_type` ∈ **{action_list}**
- [ ] If **action**: `argument` is a flat JSON object, not a string, not fenced
- [ ] No schema echo; no extra keys the Environment did not use in its example

## Role & roster (supplement)
{action_instructions}

--- Interaction history (newest lines matter most) ---
{history}

--- Turn ---
**Turn #{turn_number}** · allowed types: **{action_list}**

--- Private context (read immediately before the schema) ---
{goal}

Return **only** one JSON object as specified (no preamble, no markdown outside the value):
{format_instructions}
"""


class NegotiationSocialLLMAgent(SocialLLMAgent):
    """在长周期短时记忆通路（``SocialLLMAgent``）上叠加谈判动作协议提示。"""

    def __init__(
        self,
        *args: Any,
        all_participant_names: list[str] | None = None,
        canonical_display_names: dict[str, str] | None = None,
        role_goal_addon: str = "",
        negotiation_prompt_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        tpl = negotiation_prompt_template or NEGOTIATION_LLM_CUSTOM_TEMPLATE
        kwargs["custom_template"] = tpl
        super().__init__(*args, **kwargs)
        self._all_participant_names = list(all_participant_names) if all_participant_names else None
        roster = list(self._all_participant_names or [self.agent_name])
        base = dict(canonical_display_names or {})
        self._canonical_display_names: dict[str, str] = {
            r: str(base.get(r) or default_display_name_for_role(r)) for r in roster
        }
        if self.agent_name not in self._canonical_display_names:
            self._canonical_display_names[self.agent_name] = str(
                base.get(self.agent_name) or default_display_name_for_role(self.agent_name)
            )
        self._role_goal_addon = role_goal_addon
        self._negotiation_prompt_template = tpl

    def bind_episode_display_names(self, mapping: Mapping[str, str]) -> None:
        """与 ``LongTermNegotiationEnv.agent_display_names`` 对齐（如 profile 落库后的人名）。"""
        roster = set(self._all_participant_names or []) | {self.agent_name}
        self._canonical_display_names = {
            r: str(mapping.get(r) or default_display_name_for_role(r)) for r in roster
        }

    def _prompt_self_label(self) -> str:
        return self._canonical_display_names.get(self.agent_name, self.agent_name)

    def _rewrite_nl_for_prompt(self, text: str) -> str:
        """历史正文中的 canonical id 换成人名；保留含 ``[action]``+`{` 行（多为 JSON）不替换。"""
        if not self._canonical_display_names:
            return text
        keys = sorted(self._canonical_display_names.keys(), key=len, reverse=True)
        out_lines: list[str] = []
        for line in text.split("\n"):
            if "[action]" in line and "{" in line:
                out_lines.append(line)
                continue
            l2 = line
            for cid in keys:
                disp = self._canonical_display_names.get(cid, cid)
                if not disp or disp == cid:
                    continue
                l2 = re.sub(rf"\b{re.escape(cid)}\b", disp, l2)
            out_lines.append(l2)
        return "\n".join(out_lines)

    def _peer_public_role_block(self) -> str:
        """其他 roster 成员在设计里**对全员公开**的角色一句话（非私密数值/目标）。"""
        names = self._all_participant_names
        if not names:
            return ""
        lines: list[str] = []
        for n in names:
            if n == self.agent_name:
                continue
            desc = ROLE_SUMMARY_EN.get(n, n)
            label = self._canonical_display_names.get(n, n)
            lines.append(f"  - {label}: {desc}")
        if not lines:
            return ""
        return "[other_participants_public_role]\n" + "\n".join(lines)

    def _action_instruction_block(self, obs: Observation) -> str:
        role_line = ROLE_SUMMARY_EN.get(self.agent_name, self.agent_name)
        me = self._prompt_self_label()
        extra = self._role_goal_addon.strip()
        parts = [
            f"- You are **{me}** in this simulator (others address you by this name in **speak**).",
            f"- Role summary (your side): {role_line}",
        ]
        if self._all_participant_names:
            roster_nl = ", ".join(
                self._canonical_display_names.get(n, n) for n in self._all_participant_names
            )
            parts.append(f"- People in this episode: {roster_nl}")
        parts.append(
            "- **Structured JSON:** copy participant **name strings** exactly from this turn's Environment examples."
        )
        parts.append(
            "- **Persona / voice / digest:** see the **Private context** section at the end of this message "
            "(`[persona]`, DialogueVoice, `[agent_design_digest]`, profile/relationship blocks when present). "
            "Use for wording and stance; **live** cash and legality come from the latest Environment `[system]` line."
        )
        peer = self._peer_public_role_block()
        if peer:
            parts.append(peer)
        if extra:
            parts.append(f"- Scenario-specific goal / constraints: {extra}")
        if obs.action_instruction.strip():
            parts.append(
                "- Environment ``action_instruction`` (high-priority hint for this observation): "
                f"{obs.action_instruction.strip()}"
            )
        return "\n".join(parts)

    def _agent_design_digest_footer(self) -> str:
        """设计期稠密提示：接在私密 ``goal`` 末尾，利用单条 user 消息的近因效应。"""
        lines: list[str] = []
        lines.append(
            "[agent_design_digest — internal playbook; do not read aloud unless in-character; "
            "numeric costs below are design-time defaults unless Environment state overrides]"
        )

        profile = getattr(self, "profile", None)
        if profile is not None:
            sub: list[str] = ["[stored AgentProfile snapshot]"]
            nm = f"{getattr(profile, 'first_name', '')} {getattr(profile, 'last_name', '')}".strip()
            if nm:
                sub.append(f"Name: {nm}")
            occ = str(getattr(profile, "occupation", "") or "").strip()
            if occ:
                sub.append(f"Occupation: {occ}")
            age = getattr(profile, "age", 0) or 0
            if age:
                sub.append(f"Age: {age}")
            g = str(getattr(profile, "gender", "") or "").strip()
            if g:
                sub.append(f"Gender: {g}")
            pub = str(getattr(profile, "public_info", "") or "").strip()
            if pub:
                sub.append(f"Public info: {truncate_chars(pub, 520)}")
            big5 = str(getattr(profile, "big_five", "") or "").strip()
            if big5:
                sub.append(f"Big Five (text): {truncate_chars(big5, 420)}")
            pv = str(getattr(profile, "personality_and_values", "") or "").strip()
            if pv:
                sub.append(f"Personality & values: {truncate_chars(pv, 520)}")
            dms = str(getattr(profile, "decision_making_style", "") or "").strip()
            if dms:
                sub.append(f"Decision style: {truncate_chars(dms, 360)}")
            morals = getattr(profile, "moral_values", None) or []
            if morals:
                sub.append("Moral values: " + ", ".join(str(x) for x in morals[:12]))
            sch = getattr(profile, "schwartz_personal_values", None) or []
            if sch:
                sub.append("Schwartz values: " + ", ".join(str(x) for x in sch[:12]))
            sec = str(getattr(profile, "secret", "") or "").strip()
            if sec:
                sub.append(f"Secret (private): {truncate_chars(sec, 320)}")
            lines.append("\n".join(sub))

        persona = ROLE_PERSONA_EN.get(self.agent_name)
        if isinstance(persona, dict):
            econ: list[str] = ["[roster persona — design-time economics & north star]"]
            for key in ("daily_fixed_cost", "short_term_debt_due"):
                if key in persona:
                    econ.append(f"{key}={persona[key]}")
            sp = str(persona.get("survival_pressure", "") or "").strip()
            if sp:
                econ.append(f"Survival pressure: {truncate_chars(sp, 360)}")
            am = str(persona.get("achievement_motivation", "") or "").strip()
            if am:
                econ.append(f"Achievement north star: {truncate_chars(am, 360)}")
            dv = str(persona.get("dialogue_voice", "") or "").strip()
            if dv:
                econ.append(f"DialogueVoice (full): {truncate_chars(dv, 900)}")
            lines.append("\n".join(econ))

        anchor = ROLE_SUMMARY_EN.get(self.agent_name, "").strip()
        if anchor:
            lines.append(f"Role summary (repeat): {anchor}")

        body = "\n\n".join(lines)
        return truncate_chars(body, 4500) if len(body) > 4500 else body

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if self._goal is None:
            obs_nl = self._rewrite_nl_for_prompt(self.inbox[0][1].to_natural_language())
            viewer_ctx = self._action_instruction_block(obs)
            self._goal = await agenerate_goal(
                self.model_name,
                background=(
                    f"{obs_nl}\n\n"
                    "[Task] Write this agent's **private negotiation goal** as a compact brief they alone see each turn.\n"
                    "[Inputs] (a) Environment/scheduling text above — facts they observe. "
                    "(b) Bullet block below — their name, who else is in the episode, others' **public** role lines, "
                    "scenario extras, and this turn's Environment `action_instruction` if any.\n"
                    "[Style] Third person or imperative addressed to the agent; include motivation and risk posture; "
                    "do not contradict the protocol.\n"
                    f"{viewer_ctx}"
                ),
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
        digest = self._agent_design_digest_footer()
        if digest:
            goal_effective = (goal_effective or "").rstrip() + "\n\n" + digest

        custom_template = fill_template(
            self._negotiation_prompt_template,
            action_instructions=self._action_instruction_block(obs),
        )

        # ``agenerate_action`` / ``AgentAction`` 校验 ``to`` 时要求收件人 ∈ context.agent_names。
        # 谈判提示与 Environment 示例用人名，但 ``script_background.agent names`` 等路径可能仍是
        # canonical ``firm_*``；二者混用会导致模型填人名却被校验拒绝 → 只会返回 ``none``，无法对话。
        rk: list[str]
        if self._all_participant_names is not None:
            rk = list(self._all_participant_names)
        elif self.script_background is not None:
            rk = list(self.script_background.agent_names)
        else:
            rk = sorted(self._canonical_display_names.keys())
        _labels = [self._canonical_display_names.get(n, n) for n in rk]
        agent_names_nl = sorted(frozenset(rk) | frozenset(_labels))

        raw_history = "\n".join(f"{y.to_natural_language()}" for _, y in self.inbox)
        history = self._rewrite_nl_for_prompt(raw_history)

        action = await agenerate_action(
            self.model_name,
            history=history,
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self._prompt_self_label(),
            goal=goal_effective,
            script_like=self.script_like,
            custom_template=custom_template,
            structured_output=True,
            agent_names=agent_names_nl,
            sender=self._prompt_self_label(),
        )
        # breakpoint()
        self.memory.add(
            f"T{obs.turn_number} [{self._prompt_self_label()}] {action.to_natural_language()}"
        )
        return action


def build_negotiation_social_llm_agents(
    model_dict: dict[str, str],
    roster: tuple[str, ...],
    *,
    memory_summary_model: str | None = None,
    social_memory_kwargs: dict[str, Any] | None = None,
    agent_display_names: dict[str, str] | None = None,
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
    disp_map = {
        r: str((agent_display_names or {}).get(r) or default_display_name_for_role(r)) for r in roster
    }
    agents: dict[str, NegotiationSocialLLMAgent] = {}
    for idx, role in enumerate(roster):
        mname = model_dict[f"agent{idx + 1}"]
        ag = NegotiationSocialLLMAgent(
            agent_name=role,
            model_name=mname,
            all_participant_names=participants,
            canonical_display_names=disp_map,
            **mem_kw,
        )
        summary = ROLE_SUMMARY_EN.get(role, role)
        persona = dict(ROLE_PERSONA_EN.get(role, {}))
        voice = str(persona.get("dialogue_voice", "") or "").strip()
        chunks = [
            f"Background={persona.get('background_story', '')}",
            f"Personality={persona.get('personality', '')}",
        ]
        if voice:
            chunks.append(f"DialogueVoice={voice}")
        chunks.extend(
            [
                f"CoreSkills={','.join(str(x) for x in (persona.get('core_skills') or []))}",
                f"SurvivalPressure={persona.get('survival_pressure', '')}",
                f"AchievementMotivation={persona.get('achievement_motivation', '')}",
            ]
        )
        persona_line = "; ".join(chunks)
        design_econ: list[str] = []
        if "daily_fixed_cost" in persona:
            design_econ.append(f"daily_fixed_cost={persona['daily_fixed_cost']}")
        if "short_term_debt_due" in persona:
            design_econ.append(f"short_term_debt_due={persona['short_term_debt_due']}")
        econ_line = ("\n[design_economics] " + "; ".join(design_econ)) if design_econ else ""
        label = disp_map.get(role, default_display_name_for_role(role))
        ag.goal = (
            f"[you] {label}\n"
            f"[role_summary] {summary}\n"
            f"[persona] {persona_line}{econ_line}\n"
            "[protocol_discipline]\n"
            "- Each turn: read the **latest** Environment message for allowed `action_type` values and JSON shapes.\n"
            "- For `action`: `argument` must be one JSON object matching that message (not a stringified JSON blob, "
            "not markdown fences).\n"
            "- Reuse exact `negotiation_op` / `verb` tokens from the Environment; do not invent op names.\n"
            "- Respect calendars, session caps, and scheduling rules; advance your interests without hallucinating "
            "others' private numbers unless visible under `[contracts_visible_to_you]` or said in-session."
        )
        agents[role] = ag
    return agents


__all__ = [
    "NEGOTIATION_LLM_CUSTOM_TEMPLATE",
    "NegotiationSocialLLMAgent",
    "build_negotiation_social_llm_agents",
]
