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
You are **{agent}** in a long-horizon multi-party negotiation simulator. Your task is to produce exactly ONE valid move for the current turn. The latest Environment message is the only authoritative source and overrides all prior history, plans, or inferred strategies.

Treat the live state as ground truth, including `[system]`, visible contracts, active offers, session/bookkeeping state, participant availability, and all legality constraints. Do not rely on stale or conflicting historical information.

The private context may include `[persona]`, `DialogueVoice`, relationship information, and hidden incentives. Use these only to shape tone, style, persuasion, and prioritization, but never to override legality, Environment rules, or current state constraints.

Visibility rules: `[system]` and `[contracts_visible_to_you]` are private; `[other_participants_public_role]` is public only. Do not infer hidden utilities, budgets, thresholds, or motivations unless explicitly provided in the Environment.

You must choose exactly one action type from {action_list}. For `speak`, the argument must be short natural language, in-character, negotiation-focused, and must only use participant names from the Environment. For `non-verbal communication`, the argument must be a short phrase describing a gesture or reaction. For `action`, the argument must be a real JSON object strictly following the Environment schema, using exact field and operation names, without adding or inventing any fields, without schema metadata, and preferring the smallest valid payload. For `leave`, only use it if explicitly allowed. For `none`, only use it when silence is required or no legal move exists.

Behaviorally, maintain long-horizon consistency, prefer realistic negotiation behavior over perfect optimization, avoid repetition, avoid over-explaining, preserve leverage when possible, and keep outputs concise.

Before responding, silently verify that the chosen action is legal, the output contains exactly one JSON object, there is no markdown or extra text, `action` uses a JSON object (not a string), all fields match Environment examples, and no schema leakage is present.

Private context: {goal}.

Supplemental role information: {action_instructions}. 

Return ONLY one JSON object matching: {format_instructions}

[Scheduling — Invite]
Submit ONE session_request per slot via action_type='action', e.g.:
  {"negotiation_op":"session_request","proposed_participants":["Alice","Bob"],"purpose":"discuss delivery"}
Or pass: {"negotiation_op":"sched_pass"}. Only one request per slot. No formal budget consumed.

[Scheduling — Response]
Respond via action_type='action':
  {"negotiation_op":"session_response","requester":"Alice","accept":true}
  {"negotiation_op":"session_response_batch","responses":[{"requester":"Alice","accept":true},...]}
Or {"negotiation_op":"sched_pass"}. Accept at most ONE invite targeting you; others auto-declined.

[Session — speak / action]
Use action_type='speak' for natural language, or action_type='action' with JSON for formal ops:
  propose:     {"negotiation_op":"formal","verb":"propose_contract","terms":{price,...},"parties":["Name1","Name2"]} 
  accept:      {"negotiation_op":"formal","verb":"accept","contract_id":"optional"}
  reject:      {"negotiation_op":"formal","verb":"reject_contract","contract_id":"optional"}
  amend:       {"negotiation_op":"formal","verb":"amend_contract","contract_id":"<parent>","terms":{...}}
  sign:        {"negotiation_op":"formal","verb":"sign","contract_id":"optional"}
  share:       {"negotiation_op":"formal","verb":"contract_share","contract_id":"...","receiver":"<name>"}
  leave:       {"negotiation_op":"session_control","verb":"leave"} or {"negotiation_op":"session_control","verb":"terminate_session"}
All JSON uses personal names. Only include fields shown in examples.

Interaction history: 

{history}
"""

# . 

# Current turn: Turn #{turn_number}. 

MULTI_PARTY_NEGOTIATION_SKILL = """
## Multi-party negotiation skill (MANDATORY — every session)

You are in a multi-party market. Success requires engaging multiple counterparties in parallel, not focusing on one.

## Core behavior rules (always apply)

1. Shop around first: never accept an offer before contacting at least 2 counterparties; always compare alternatives before committing.
2. Cross-reference offers: use other parties’ bids to negotiate better terms; you may say “Another vendor offered X — can you match or improve it?” but do not reveal exact private numbers.
3. Competitive pressure: signal that alternatives exist to create bidding tension without fabricating offers.
4. Coalition dynamics: form temporary alliances to pressure others and switch when beneficial.
5. Maximize total value (BATNA-aware): optimize across all counterparties, not single deals; track best alternative agreement.
6. Information arbitrage: use insights from one party to negotiate with another without disclosing confidential details.

## Role-specific strategy

Buyer: contact all sellers, compare bundles (price/delivery/quality), explicitly signal comparison. Incumbent seller: emphasize reliability, loyalty, track record; engage all buyers and defend against challengers. Challenger seller: undercut incumbents, expose weaknesses, offer trials/guarantees, target dissatisfied buyers. Specialist: compete on trust/reliability, identify gaps, engage all parties.

## Negotiation toolkit

Anchoring: set strong initial offer; Concession patterning: smaller concessions over time with reciprocity; Deadline leverage: exploit urgency; Bundle strategy: negotiate full package not just price; Walk-away power: signal credible exit; Nibbling: ask for small extras after agreement; Silence pressure: use pauses to induce concessions.

## Dialogue templates

Opening: “I’m comparing multiple vendors — what’s your best offer?” Cross-reference: “Another offer is X with Y delivery — can you improve?” Urgency: “Need to close soon — otherwise I’ll move on.” Concession: “I can adjust X if you improve Y.” Reliability: “Lower price, but worse reliability.” Coalition: “If we hold this line, buyers can’t split us.” Close: “If you match this, we sign and I stop negotiating.”

## Anti-patterns

Do not focus on a single counterparty; do not accept before evaluating at least 2 alternatives; do not reveal exact private numbers (use ranges/comparisons); do not remain passive—always engage multiple parties.

"""

AVAILABLE_ACTION_RULES = """
[Scheduling — Invite]
Submit ONE session_request per slot via action_type='action', e.g.:
  {"negotiation_op":"session_request","proposed_participants":["Alice","Bob"],"purpose":"discuss delivery"} 
Or pass: {"negotiation_op":"sched_pass"}. Q_i=1: at most one request per slot. No formal budget consumed.

[Scheduling — Response]
Respond via action_type='action':
  {"negotiation_op":"session_response","requester":"Alice","accept":true}
  {"negotiation_op":"session_response_batch","responses":[{"requester":"Alice","accept":true},...]}
Or {"negotiation_op":"sched_pass"}. Accept at most ONE invite targeting you; others auto-declined.

[Session — speak / action]
Use action_type='speak' for natural language, or action_type='action' with JSON for formal ops:
  propose:     {"negotiation_op":"formal","verb":"propose_contract","terms":{price,...},"parties":["Name1","Name2"]}  
  accept:      {"negotiation_op":"formal","verb":"accept","contract_id":"optional"}
  reject:      {"negotiation_op":"formal","verb":"reject_contract","contract_id":"optional"}
  amend:       {"negotiation_op":"formal","verb":"amend_contract","contract_id":"<parent>","terms":{...}}
  sign:        {"negotiation_op":"formal","verb":"sign","contract_id":"optional"}
  share:       {"negotiation_op":"formal","verb":"contract_share","contract_id":"...","receiver":"<name>"}
  leave:       {"negotiation_op":"session_control","verb":"leave"} or {"negotiation_op":"session_control","verb":"terminate_session"}
All JSON uses personal names. Only include fields shown in examples.
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
        self.name = self._prompt_self_label()

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
            label = self._canonical_display_names.get(n, n)
            lines.append(f"  - {label}")
        if not lines:
            return ""
        return "[other_participants_public_role]\n" + "\n".join(lines)

    def _action_instruction_block(self, obs: Observation) -> str:
        me = self._prompt_self_label()
        extra = self._role_goal_addon.strip()
        parts = [
            f"- You are **{me}** and others address you by this name in **speak**.",
        ]
        if self._all_participant_names:
            roster_nl = ", ".join(
                self._canonical_display_names.get(n, n) for n in self._all_participant_names
            )
            parts.append(f"- People in this episode: {roster_nl}")
        
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
                "- Environment ``action_instruction``: "
                f"{obs.action_instruction.strip()}"
            )
        return "\n".join(parts)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)
        if self._goal is None:
            raise Exception("Goal is not set")
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
                agent=self._prompt_self_label(),
            )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="", to=[])
        system_digest = obs.system_digest

        goal_effective = self._goal

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

        raw_history = "".join(f"{y.to_natural_language()}" for _, y in self.inbox) # \n
        history = self._rewrite_nl_for_prompt(raw_history) # 序列化后经 _rewrite_nl_for_prompt 将 canonical 名称（firm_a）替换为显示名（Avery Singh）

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
        persona_line = "\n".join(["- " + x for x in chunks])
        design_econ: list[str] = []
        if "daily_fixed_cost" in persona:
            design_econ.append(f"daily_fixed_cost={persona['daily_fixed_cost']}")
        if "short_term_debt_due" in persona:
            design_econ.append(f"short_term_debt_due={persona['short_term_debt_due']}")
        econ_line = ("\ndesign_economics:\n " + "\n- ".join(design_econ)) if design_econ else ""
        label = disp_map.get(role, default_display_name_for_role(role))
        ag.goal = (
            f"persona:\n{persona_line}\n\n\n{econ_line}\n"
            "[protocol_discipline]\n"
            "- Each turn: read the **latest** Environment message for allowed `action_type` values and JSON shapes.\n"
            "- For `action`: `argument` must be one JSON object matching that message (not a stringified JSON blob, "
            "not markdown fences).\n"
            "- Reuse exact `negotiation_op` / `verb` tokens from the Environment; do not invent op names.\n"
            "- Respect calendars, session caps, and scheduling rules; advance your interests without hallucinating "
            # f"\n\n{MULTI_PARTY_NEGOTIATION_SKILL}"
        )
        agents[role] = ag
    return agents


__all__ = [
    "MULTI_PARTY_NEGOTIATION_SKILL",
    "NEGOTIATION_LLM_CUSTOM_TEMPLATE",
    "NegotiationSocialLLMAgent",
    "build_negotiation_social_llm_agents",
]
