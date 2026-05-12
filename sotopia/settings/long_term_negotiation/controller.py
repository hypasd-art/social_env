"""长期商业谈判 — 确定性控制器（编排 day / slot / phase），不修改底层 env。"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from sotopia.messages.message_classes import Observation

from .scheduling_resolution import SchedulingResolveTrace, resolve_valid_session_sets
from .session_roster import (
    classify_session_roster,
    roster_blocks_trade_contract_drafting,
    section7_session_hints,
)
from .types import (
    PRINCIPAL_PARTY_ROLES,
    SESSION_SPEAKER_ROLE_ORDER,
    NegotiationContract,
    NegotiationTimelineParams,
    Phase,
    ResolvedSession,
    SessionInviteRecord,
    SessionRuntimeMeta,
)


def remaining_days(day: int, D: int) -> int:
    return max(0, D - day + 1)


class NegotiationWorldController:
    """状态机 + 合同簿 + 每 slot 的 invites/responses/sessions。"""

    def __init__(
        self,
        agent_names: tuple[str, ...],
        params: NegotiationTimelineParams | None = None,
    ) -> None:
        if len(agent_names) < 2:
            raise ValueError("Need at least two agents for the negotiation world.")
        self.agent_names = tuple(agent_names)
        self.params = params or NegotiationTimelineParams()
        self.phase: Phase = Phase.INIT
        self.day: int = 1
        self.slot: int = 1
        self.terminal: str | None = None
        self.session_round: int = 0
        self.contracts: dict[str, NegotiationContract] = {}
        self.primary_contract_id: str | None = None

        # Scheduling (current slot)
        self._invites: dict[str, SessionInviteRecord] = {}
        self._responses: dict[tuple[str, frozenset[str], int, int], dict[str, bool]] = {}
        self._resolved_sessions: list[ResolvedSession] = []
        self._active_session_idx: int = 0
        self._session_busy: set[str] = set()
        self._transcript: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.scheduling_log: list[tuple[int, int, str, str]] = []
        #: §1.2 — 结构化审计（会话内 message / formal / control）
        self.message_log: list[dict[str, Any]] = []
        self.action_log: list[dict[str, Any]] = []
        self.session_log: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []
        #: 面向复盘的 **全局时间线**（合同提出/接受/签署、融资与监管里程碑、世界终止等）。
        self.execution_timeline: list[dict[str, Any]] = []
        #: §8.4 — agent -> 待发「外部事件」观察文本（下次 digest 读出后清空）。
        self._external_event_queue: dict[str, list[str]] = {}
        self.state_snapshots: list[dict[str, Any]] = []
        #: 仅 session participants 写入（post-session bookkeeping 与设计 §2.1 对齐的最小版本）
        self.visible_history: dict[str, list[str]] = {n: [] for n in self.agent_names}

        #: §2 — slot 内 session 收口时的 deterministic 审计
        self._pending_session_close_reason: str = "session_closed"
        self._slot_session_close_records: list[dict[str, Any]] = []
        self._viewer_slot_bookkeeping_summary: dict[str, str] = {}

        #: §2.1 — 预算与时间片状态
        self._formal_actions_per_agent_day: dict[str, int] = {}
        #: §6.4 ``H_i(s)`` — session_id -> agent -> formal 计数
        self._formal_actions_per_agent_session: dict[str, dict[str, int]] = {}
        #: §6.5 — session_control 计数（leave / terminate_session）
        self._session_control_actions_per_agent: dict[str, dict[str, int]] = {}
        self._nl_turn_counts: dict[str, dict[str, int]] = {}

        #: §6.4.4 — investor/regulator 退出谈判路径（不必然 world terminal）
        self.investor_financing_path_withdrawn: bool = False
        self.regulator_regulatory_path_withdrawn: bool = False

        #: §3 scheduling — violations、resolution 追溯、上一轮 digest
        self.scheduling_violation_log: list[dict[str, Any]] = []
        self.scheduling_resolution_log: list[dict[str, Any]] = []
        self._last_scheduling_digest: dict[str, str] = {n: "" for n in self.agent_names}
        self._last_resolve_trace: SchedulingResolveTrace | None = None

        #: §4.1 / §4.3 — 本会话原子步（仅 session macro turn 递增，单调）
        self._episode_atomic_turn: int = 0
        self.session_meta: dict[str, SessionRuntimeMeta] = {}
        self._session_total_turns: dict[str, int] = {}
        self._session_agent_turns: dict[str, dict[str, int]] = {}
        self._session_last_actor: dict[str, str | None] = {}

        #: §9.3 ``failure``（stagnation）— 当日是否出现结构化进展 / 已连续 idle 的自然日数
        self._progress_flag_this_calendar_day: bool = False
        self._consecutive_idle_calendar_days: int = 0

    def reset(self) -> None:
        agent_names = self.agent_names
        params = self.params
        self.__init__(agent_names, params)

    def start_episode(self) -> None:
        self.day = 1
        self.slot = 1
        self._formal_actions_per_agent_day.clear()
        self._last_scheduling_digest = {n: "" for n in self.agent_names}
        self._last_resolve_trace = None
        self.investor_financing_path_withdrawn = False
        self.regulator_regulatory_path_withdrawn = False
        self._external_event_queue = {n: [] for n in self.agent_names}
        self._progress_flag_this_calendar_day = False
        self._consecutive_idle_calendar_days = 0
        self._clear_slot_state()
        self.phase = Phase.SCHEDULE_INVITE

    def mark_structural_progress(self) -> None:
        """§9.3 — 当前自然日内出现会话成立、合同或附条件侧的结构性更新（含外生事件）。"""
        if self.terminal or self.phase == Phase.TERMINATED:
            return
        self._progress_flag_this_calendar_day = True

    def _clear_slot_state(self) -> None:
        self._invites.clear()
        self._responses.clear()
        self._resolved_sessions.clear()
        self._active_session_idx = 0
        self._session_busy.clear()
        self.session_round = 0
        self._nl_turn_counts.clear()
        self.session_meta.clear()
        self._session_total_turns.clear()
        self._session_agent_turns.clear()
        self._session_last_actor.clear()
        self._formal_actions_per_agent_session.clear()
        self._session_control_actions_per_agent.clear()

    def append_event_records(self, records: list[dict[str, Any]]) -> None:
        """§1.2 / §1.0 — 外部事件为世界状态更新，记入 event_log。"""
        self.event_log.extend(records)

    def append_state_snapshot(self, payload: dict[str, Any]) -> None:
        """§1.2 — end-of-day 或评测写出的浅快照钩子。"""
        row = dict(payload)
        row.setdefault("day", self.day)
        row.setdefault("slot", self.slot)
        self.state_snapshots.append(row)

    def record_execution_event(self, kind: str, message: str, **extra: Any) -> None:
        """追加一条人类可读的全局事件（与 ``action_log`` 互补；用于导出 ``*.execution.json``）。"""
        row: dict[str, Any] = {
            "seq": len(self.execution_timeline) + 1,
            "day": int(self.day),
            "slot": int(self.slot),
            "phase": self.phase.value,
            "episode_atomic_turn": int(self._episode_atomic_turn),
            "kind": kind,
            "message": message,
        }
        if extra:
            row["detail"] = dict(extra)
        self.execution_timeline.append(row)

    def enqueue_external_event_notification(self, agent: str, block: str) -> None:
        """§8.4 — 将事件观察挂入队列，在后续 ``negotiation_context_addon`` 中读出。"""
        if agent not in self.agent_names:
            return
        self._external_event_queue.setdefault(agent, []).append(block)

    def drain_external_event_observations(self, viewer: str) -> str:
        lst = self._external_event_queue.setdefault(viewer, [])
        out = "\n\n".join(lst) if lst else ""
        self._external_event_queue[viewer] = []
        return out

    def apply_external_negotiation_effects(self, ops: list[dict[str, Any]]) -> None:
        """§8.3 — 合同 / 谈判侧结构化更新（与 ``EffectOp`` 互补）。"""
        touched = False
        for raw in ops:
            op = str(raw.get("op", ""))
            if op == "patch_primary_contract_terms":
                cid = self.primary_contract_id
                if not cid or cid not in self.contracts:
                    continue
                c = self.contracts[cid]
                c.terms.update(dict(raw.get("terms_patch") or {}))
                self._finalize_financing_reg_flags(c)
                touched = True
            elif op == "patch_contract_terms":
                cid = str(raw.get("contract_id", ""))
                c = self.contracts.get(cid)
                if not c:
                    continue
                c.terms.update(dict(raw.get("terms_patch") or {}))
                self._finalize_financing_reg_flags(c)
                touched = True
            elif op == "set_regulatory_required_on_primary":
                cid = self.primary_contract_id
                if not cid or cid not in self.contracts:
                    continue
                c = self.contracts[cid]
                c.terms["regulatory_required"] = int(raw.get("value", 1))
                self._finalize_financing_reg_flags(c)
                touched = True
        if touched:
            self.mark_structural_progress()

    def _audit_action(
        self,
        *,
        agent: str,
        negotiation_op: str | None,
        verb: str | None,
        valid: bool,
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "day": self.day,
            "slot": self.slot,
            "phase": self.phase.value,
            "agent": agent,
            "negotiation_op": negotiation_op,
            "verb": verb,
            "valid": valid,
            "reason": reason,
        }
        if extra:
            row.update(extra)
        self.action_log.append(row)

    def _audit_message_line(self, session_id: str, agent: str, content: str) -> None:
        self.message_log.append(
            {
                "day": self.day,
                "slot": self.slot,
                "session_id": session_id,
                "agent": agent,
                "content": content,
            }
        )

    def _session_formal_used(self, agent: str) -> int:
        sess = self._current_session()
        if sess is None:
            return 0
        return self._formal_actions_per_agent_session.get(sess.session_id, {}).get(agent, 0)

    def _available_formal_actions(self, agent: str) -> int | None:
        """§6.4 ``min(F_max-F_i, H_max-H_i(s))``；无上限时返回 ``None``。"""
        f_lim = self.params.max_formal_actions_per_agent_per_day
        h_lim = self.params.max_formal_actions_per_agent_per_session
        rem: list[int] = []
        if f_lim is not None:
            rem.append(max(0, f_lim - self._formal_actions_per_agent_day.get(agent, 0)))
        if h_lim is not None:
            rem.append(max(0, h_lim - self._session_formal_used(agent)))
        if not rem:
            return None
        return min(rem)

    def _session_control_used(self, agent: str) -> int:
        sess = self._current_session()
        if sess is None:
            return 0
        return self._session_control_actions_per_agent.get(sess.session_id, {}).get(agent, 0)

    def _session_control_budget_allows(self, agent: str, *, verb: str) -> bool:
        lim = self.params.max_session_control_actions_per_agent_per_session
        if lim is None:
            return True
        if self._session_control_used(agent) >= lim:
            self._audit_action(
                agent=agent,
                negotiation_op="session_control",
                verb=verb,
                valid=False,
                reason="session_control_budget_exceeded",
                extra={"limit": lim, "session_id": self._current_session().session_id if self._current_session() else None},
            )
            return False
        return True

    def _session_control_budget_commit(self, agent: str) -> None:
        lim = self.params.max_session_control_actions_per_agent_per_session
        sess = self._current_session()
        if lim is None or sess is None:
            return
        b = self._session_control_actions_per_agent.setdefault(sess.session_id, {})
        b[agent] = b.get(agent, 0) + 1

    def _formal_budget_allows_increment(self, agent: str, *, verb: str) -> bool:
        avail = self._available_formal_actions(agent)
        if avail is None:
            return True
        if avail <= 0:
            f_lim = self.params.max_formal_actions_per_agent_per_day
            h_lim = self.params.max_formal_actions_per_agent_per_session
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb=verb,
                valid=False,
                reason="formal_budget_exceeded_daily_or_per_session",
                extra={
                    "F_max": f_lim,
                    "F_used": self._formal_actions_per_agent_day.get(agent, 0),
                    "H_max": h_lim,
                    "H_used": self._session_formal_used(agent),
                },
            )
            return False
        return True

    def _formal_budget_commit_increment(self, agent: str) -> None:
        lim_d = self.params.max_formal_actions_per_agent_per_day
        if lim_d is not None:
            self._formal_actions_per_agent_day[agent] = (
                self._formal_actions_per_agent_day.get(agent, 0) + 1
            )
        lim_h = self.params.max_formal_actions_per_agent_per_session
        sess = self._current_session()
        if lim_h is not None and sess is not None:
            b = self._formal_actions_per_agent_session.setdefault(sess.session_id, {})
            b[agent] = b.get(agent, 0) + 1

    def _build_viewer_bookkeeping_summaries(
        self, *, slot_closure_reason: str, had_resolved_sessions: bool
    ) -> None:
        """§2.1 — 供给下一 scheduling / session observation 摘要（不含 agent 推断，仅 deterministic）。"""
        sess_ids_by_agent: dict[str, list[str]] = {a: [] for a in self.agent_names}
        last_reason_by_agent: dict[str, str] = {}
        for rec in self._slot_session_close_records:
            sid = rec.get("session_id", "")
            reason = rec.get("end_reason", "?")
            for p in rec.get("participants_final") or []:
                if p in sess_ids_by_agent:
                    sess_ids_by_agent[p].append(sid)
                    last_reason_by_agent[p] = str(reason)

        for a in self.agent_names:
            chunks: list[str] = []
            if not had_resolved_sessions:
                chunks.append(f"slot_outcome={slot_closure_reason}")
            elif sess_ids_by_agent[a]:
                chunks.append(f"sessions_you_were_in_at_close={sess_ids_by_agent[a]}")
                chunks.append(f"your_last_close_reason={last_reason_by_agent.get(a, '?')}")
            else:
                chunks.append(f"slot_outcome={slot_closure_reason}; role=idle_non_participant")
            lim_f = self.params.max_formal_actions_per_agent_per_day
            if lim_f is not None:
                u = self._formal_actions_per_agent_day.get(a, 0)
                chunks.append(f"formal_F_used_today={u}/{lim_f}")
            lim_h = self.params.max_formal_actions_per_agent_per_session
            if lim_h is not None:
                chunks.append(f"formal_H_cap_per_session={lim_h}")
            lim_sc = self.params.max_session_control_actions_per_agent_per_session
            if lim_sc is not None:
                chunks.append(f"session_control_cap_per_session={lim_sc}")
            lim_n = self.params.max_natural_turns_per_agent_per_session
            if lim_n is not None:
                chunks.append(f"natural_message_cap_per_agent_per_session={lim_n}")
            if self.terminal:
                chunks.append(f"world_terminal={self.terminal}")
            if self.investor_financing_path_withdrawn:
                chunks.append("investor_financing_path_withdrawn=true")
            if self.regulator_regulatory_path_withdrawn:
                chunks.append("regulator_regulatory_path_withdrawn=true")
            self._viewer_slot_bookkeeping_summary[a] = "; ".join(chunks)

    def _flush_slot_transcript_to_logs(self) -> None:
        """§2.1 post-session bookkeeping：收口 transcript、session/action 链路、摘要与合同快照。"""
        had_rs = len(self._resolved_sessions) > 0
        slot_closure_reason = (
            "scheduling_yielded_no_session" if not had_rs else "slot_session_phase_complete"
        )
        t_end = self._global_turn_number()

        self._build_viewer_bookkeeping_summaries(
            slot_closure_reason=slot_closure_reason,
            had_resolved_sessions=had_rs,
        )

        contracts_snap = [
            {
                "contract_id": cid,
                "parent_id": c.parent_id,
                "status": c.status,
                "created_by": c.created_by,
                "created_at": dict(c.created_at),
                "parties": sorted(c.parties),
                "acceptances": dict(c.acceptances),
                "visibility": sorted(c.visibility),
                "signatures": dict(c.signatures),
                "financing": dict(c.financing),
                "regulatory": dict(c.regulatory),
                "history_tail": list(c.history[-16:]),
            }
            for cid, c in sorted(self.contracts.items())
        ]

        entry: dict[str, Any] = {
            "kind": "post_session_bookkeeping",
            "day": self.day,
            "slot": self.slot,
            "t_end_global_turn": t_end,
            "slot_closure_reason": slot_closure_reason,
            "session_ids": [s.session_id for s in self._resolved_sessions],
            "closed_sessions_detail": [
                {"session_id": s.session_id, "participants": list(s.participants)}
                for s in self._resolved_sessions
            ],
            "sessions_closed_records": list(self._slot_session_close_records),
            "transcript": [
                {"session_id": sid, "agent": ag, "content": ct, "viewers_at_send": list(viewers)}
                for sid, ag, ct, viewers in self._transcript
            ],
            "negotiation_contracts_snapshot": contracts_snap,
            "budget_formal_actions_today": dict(self._formal_actions_per_agent_day),
            "viewer_bookkeeping_one_liner": dict(self._viewer_slot_bookkeeping_summary),
        }
        self.session_log.append(entry)

        for sid, agent, content, viewers in self._transcript:
            line = f"[d={self.day} k={self.slot} sid={sid}] {agent}: {content}"
            for v in viewers:
                self.visible_history.setdefault(v, []).append(line)
        self._transcript.clear()
        self._slot_session_close_records.clear()

    # ---- observations ------------------------------------------------------

    def base_observation_tail(self, viewer: str, *, system_digest: str = "") -> str:
        rem = remaining_days(self.day, self.params.D)
        parts = [
            f"[NegotiationWorld] day={self.day}/{self.params.D} remaining_days={rem} "
            f"slot={self.slot}/{self.params.s_max_per_day}",
            f"phase={self.phase.value}",
        ]
        if system_digest:
            parts.append(system_digest)
        if self.terminal:
            parts.append(f"WORLD_TERMINAL: {self.terminal}")
        return "\n".join(parts)

    def negotiation_context_addon(self, viewer: str, *, history_tail_lines: int = 8) -> str:
        """§1.0 / §2.1 — 把「对自己可见的会话历史」与「对自己可见的合同」拼进系统 digest。"""
        blocks: list[str] = []
        ext_blk = self.drain_external_event_observations(viewer)
        if ext_blk.strip():
            blocks.append("[external_events §8]\n" + ext_blk)
        hist = self.visible_history.get(viewer, [])
        if hist:
            tail = hist[-history_tail_lines:]
            blocks.append("[visible_session_history_recent]\n" + "\n".join(tail))

        contract_lines: list[str] = []
        for cid, c in sorted(self.contracts.items()):
            if viewer not in c.visibility:
                continue
            price = c.terms.get("price", "?")
            contract_lines.append(
                f"  {cid} parent={c.parent_id!r} status={c.status!r} parties={sorted(c.parties)!r} "
                f"accept={dict(c.acceptances)!r} price={price!r} "
                f"financing={c.financing!r} regulatory={c.regulatory!r}"
            )
        if contract_lines:
            blocks.append("[contracts_visible_to_you]\n" + "\n".join(contract_lines))

        summary = self._viewer_slot_bookkeeping_summary.get(viewer, "")
        if summary.strip():
            blocks.append("[last_slot_bookkeeping]\n" + summary)
        path_bits: list[str] = []
        if viewer == "investor" and self.investor_financing_path_withdrawn:
            path_bits.append("investor_financing_path_withdrawn=true")
        if viewer == "regulator" and self.regulator_regulatory_path_withdrawn:
            path_bits.append("regulator_regulatory_path_withdrawn=true")
        if path_bits:
            blocks.append("[negotiation_path_flags]\n" + "; ".join(path_bits))

        return "\n\n".join(blocks)

    def observation_for_scheduling_invite(self, viewer: str, system_digest: str) -> Observation:
        prior = (self._last_scheduling_digest.get(viewer) or "").strip()
        pre = f"[Previous slot — your scheduling outcome]\n{prior}\n\n" if prior else ""
        text = (
            f"{pre}"
            "Scheduling — Invitation round.\n"
            "Submit ONE session request using action_type='action' and JSON in argument:\n"
            '{"negotiation_op":"session_request","proposed_participants":["..."],"purpose":"..."}\n'
            'Or pass: {"negotiation_op":"sched_pass"} to skip inviting this slot.\n'
            "Rules (design §3.1): Q_i=1 — at most ONE session_request per slot; if your request fails, "
            "do not invite a different roster in the same slot (wait for a later slot).\n"
            "Design §6: scheduling JSON does not consume daily formal-action budget (F_max).\n"
            "`purpose` describes the meeting topic only, not binding price/deal terms.\n"
            f"Your name: {viewer}. Roster ids you may reference: {list(self.agent_names)}.\n"
        )
        return Observation(
            last_turn=text + "\n" + self.base_observation_tail(viewer, system_digest=system_digest),
            turn_number=self._global_turn_number(),
            available_actions=["speak", "action", "none"],
        )

    def observation_for_scheduling_response(self, viewer: str, system_digest: str) -> Observation:
        pend = self.pending_invites_for(viewer)
        header_lines = [
            "Scheduling — Response round.",
            "Invitations visible to you (other agents do not see requests they are not part of) — §3.2:",
        ]
        if pend:
            for p in pend:
                header_lines.append(
                    f"  - requester={p.requester} "
                    f"participants={sorted(p.proposed_participants)} "
                    f"purpose={p.purpose!r}"
                )
        else:
            header_lines.append("  (none)")
        header_lines.extend(
            [
                "Respond via action_type='action'. Single invite:",
                '{"negotiation_op":"session_response","requester":"<name>","accept":true|false}',
                "Multiple invites in ONE tool call (design §3.3):",
                '{"negotiation_op":"session_response_batch","responses":['
                '{"requester":"<a>","accept":true},{"requester":"<b>","accept":false}]}',
                "Rules: unknown/missing/unparseable → treated as decline; you may accept at most ONE invitation "
                "that targets you — otherwise all your accepts become ineffective this slot.",
                'If nothing to submit: {"negotiation_op":"sched_pass"}.',
            ]
        )
        return Observation(
            last_turn="\n".join(header_lines) + "\n" + self.base_observation_tail(viewer, system_digest=system_digest),
            turn_number=self._global_turn_number(),
            available_actions=["speak", "action", "none"],
        )

    def observation_for_session(self, viewer: str, system_digest: str) -> Observation:
        sess = self._current_session()
        if sess is None:
            return Observation(
                last_turn="No active session.",
                turn_number=self._global_turn_number(),
                available_actions=["none"],
            )
        others = [p for p in sess.participants if p != viewer]
        sid = sess.session_id
        N_s = self._session_total_turns.get(sid, 0)
        T_s = self._effective_T_s(sess)
        K_lim = self.params.max_turns_per_participant_per_session
        ky = self._session_agent_turns.get(sid, {}).get(viewer, 0)
        k_hint = (
            f"; your_macro_turns_K_i={ky}/{K_lim}"
            if K_lim is not None
            else f"; your_macro_turns_K_i={ky} (no per-agent cap)"
        )
        budgets = (
            f"§4.3 budgets: N_s_macro={N_s}/{T_s}{k_hint} "
            "(order: speaker_role_order ∩ session.participants; skip agents at K_i=K_s; "
            "one macro turn consumes for pass/invalid/none/formal/control alike).\n"
        )
        avail_f = self._available_formal_actions(viewer)
        avail_f_s = "∞" if avail_f is None else str(avail_f)
        sc_lim = self.params.max_session_control_actions_per_agent_per_session
        sc_used = self._session_control_used(viewer)
        sc_s = (
            f"{sc_used}/{sc_lim}"
            if sc_lim is not None
            else f"{sc_used} (no §6 session_control cap)"
        )
        budgets += (
            f"§6 budget hints: formal_actions_available≈min(F,H)={avail_f_s}; "
            f"session_control_used={sc_s}; "
            f"M_max(messages)=see max_natural_turns_per_agent_per_session in bookkeeping.\n"
        )
        rk = classify_session_roster(sess.participants)
        s7 = section7_session_hints(rk, viewer=viewer)
        text = (
            f"Active session {sess.session_id} with {list(sess.participants)}.\n"
            f"Others: {others}.\n"
            + budgets
            + s7
            + "\n"
            + "Use natural language with action_type='speak',\n"
            "or structured formal/session_control via action_type='action' with JSON:\n"
            "§5 contract (reference only if you are in visibility_set):\n"
            "- propose: "
            '{"negotiation_op":"formal","verb":"propose_contract","terms":{"price": number, '
            '"regulatory_required": 0|1, "financing_required"|"financing_contingent": 0|1 (§9 显式融资附条件), '
            '"valuation": ..., "payment": ..., "closing": ..., '
            '"compliance": ..., "penalty": ...}}\n'
            '- accept (any firm in c.parties): '
            '{"negotiation_op":"formal","verb":"accept","contract_id":"optional"}\n'
            '- reject (any firm in c.parties): '
            '{"negotiation_op":"formal","verb":"reject_contract","contract_id":"optional"}\n'
            '- amend ⇒ new contract, parent superseded: '
            '{"negotiation_op":"formal","verb":"amend_contract","contract_id":"<parent>","terms":{...}}\n'
            '- financing review ⇒ add investor to visibility: '
            '{"negotiation_op":"formal","verb":"request_financing_review","contract_id":"optional"}\n'
            '- regulatory review ⇒ add regulator to visibility: '
            '{"negotiation_op":"formal","verb":"request_regulatory_review","contract_id":"optional"}\n'
            '- share to current-session participant: '
            '{"negotiation_op":"formal","verb":"contract_share","contract_id":"...","receiver":"<name>"}\n'
            '- sign after mutual principal accept: '
            '{"negotiation_op":"formal","verb":"sign","contract_id":"optional"}\n'
            '- investor financing commit or decline (visible contract): '
            '{"negotiation_op":"formal","verb":"finance_commit","contract_id":"optional"} | '
            '{"negotiation_op":"formal","verb":"finance_decline","contract_id":"optional","reason":"optional"}\n'
            '- regulator approve or block (visible contract): '
            '{"negotiation_op":"formal","verb":"regulatory_approve","contract_id":"optional"} | '
            '{"negotiation_op":"formal","verb":"regulatory_block","contract_id":"optional","reason":"optional"}\n'
            "- terminate.negotiation (§6.4.4, consumes formal quota): principals end entire world; "
            "investor/regulator exit only financing/regulatory negotiation path.\n"
            '{"negotiation_op":"terminate_negotiation"}\n'
            "- leave session (or equivalent terminate.session): "
            '{"negotiation_op":"session_control","verb":"leave"} or '
            '{"negotiation_op":"session_control","verb":"terminate_session"}\n'
        )
        return Observation(
            last_turn=text + "\n" + self.base_observation_tail(viewer, system_digest=system_digest),
            turn_number=self._global_turn_number(),
            available_actions=["speak", "non-verbal communication", "action", "none", "leave"],
        )

    def _global_turn_number(self) -> int:
        return self._episode_atomic_turn

    @staticmethod
    def _ordered_session_participants(sess: ResolvedSession) -> tuple[str, ...]:
        ps = set(sess.participants)
        return tuple(r for r in SESSION_SPEAKER_ROLE_ORDER if r in ps)

    def _effective_T_s(self, sess: ResolvedSession) -> int:
        if self.params.max_total_turns_per_session is not None:
            return max(0, int(self.params.max_total_turns_per_session))
        return int(self.params.max_session_rounds * max(1, len(sess.participants)))

    def _eligible_speakers_under_K_s(self, sid: str, sess: ResolvedSession) -> list[str]:
        K_s = self.params.max_turns_per_participant_per_session
        bt = self._session_agent_turns.get(sid, {})
        ordered = self._ordered_session_participants(sess)
        if K_s is None:
            return list(ordered)
        return [a for a in ordered if bt.get(a, 0) < K_s]

    def _current_session(self) -> ResolvedSession | None:
        if not self._resolved_sessions or self._active_session_idx >= len(self._resolved_sessions):
            return None
        return self._resolved_sessions[self._active_session_idx]

    def current_session_id(self) -> str | None:
        """当前 active session 的 id；供 env 在 submit 可能推进 session 之前钉住回合归属。"""
        sess = self._current_session()
        return None if sess is None else sess.session_id

    def pending_invites_for(self, agent: str) -> list[SessionInviteRecord]:
        return [
            r
            for r in self._invites.values()
            if agent in r.proposed_participants and agent != r.requester
        ]

    # ---- scheduling §3 -------------------------------------------------------

    @staticmethod
    def _invite_tuple_key(inv: SessionInviteRecord) -> tuple[str, frozenset[str], int, int]:
        return (inv.requester, inv.proposed_participants, inv.day, inv.slot)

    def _scheduling_log_violation(self, kind: str, **extra: Any) -> None:
        row: dict[str, Any] = {"kind": kind, "day": self.day, "slot": self.slot}
        row.update(extra)
        self.scheduling_violation_log.append(row)

    def _impute_missing_responses_as_decline(self) -> None:
        """§3.3 — missing / unparsed ⇒ effective decline (+ log no_response)."""
        for inv in self._invites.values():
            key = self._invite_tuple_key(inv)
            bucket = self._responses.setdefault(key, {})
            for p in inv.proposed_participants:
                if p == inv.requester:
                    continue
                if p not in bucket:
                    bucket[p] = False
                    self._scheduling_log_violation(
                        "scheduling_no_response",
                        responder=str(p),
                        requester=str(inv.requester),
                    )

    def _apply_one_accept_per_invitee_constraint(self) -> None:
        """§3.3 — 同一 slot 对「他人发来的邀请」至多 accept 一处；否则全局 decline。"""
        for agent in self.agent_names:
            acc_keys: list[tuple[str, frozenset[str], int, int]] = []
            for inv in self._invites.values():
                if agent == inv.requester:
                    continue
                if agent not in inv.proposed_participants:
                    continue
                key = self._invite_tuple_key(inv)
                if self._responses.setdefault(key, {}).get(agent, False) is True:
                    acc_keys.append(key)
            if len(acc_keys) > 1:
                self._scheduling_log_violation(
                    "scheduling_multiple_accepts_same_slot_invalidated_all",
                    agent=agent,
                    n_invites=len(acc_keys),
                )
                for key in acc_keys:
                    self._responses.setdefault(key, {})[agent] = False

    def _fps_in_conflict_pairs(self, trace: SchedulingResolveTrace) -> set[frozenset[str]]:
        fs: set[frozenset[str]] = set()
        for a, b in trace.conflict_pairs:
            fs.add(a)
            fs.add(b)
        return fs

    def _store_scheduling_response_item(self, agent: str, rq: str, accept: bool) -> str:
        if rq not in self._invites:
            self._scheduling_log_violation(
                "scheduling_unknown_request_ref",
                responder=agent,
                requester_placeholder=rq,
            )
            return "unknown_requester"
        inv = self._invites[rq]
        if agent not in inv.proposed_participants or agent == rq:
            return "not_invited"
        key = self._invite_tuple_key(inv)
        bucket = self._responses.setdefault(key, {})
        if agent in bucket:
            self._scheduling_log_violation(
                "scheduling_duplicate_response_turn",
                responder=agent,
                requester=inv.requester,
            )
        bucket[agent] = accept
        return "ok"

    def submit_scheduling_response_payload(self, agent: str, payload: dict[str, Any]) -> str:
        """§3.3 — 支持单笔或 ``session_response_batch``。"""
        if self.terminal or self.phase == Phase.TERMINATED:
            return "ignored: world_terminal"
        if self.phase != Phase.SCHEDULE_RESPONSE:
            return "ignored: wrong phase"
        op = payload.get("negotiation_op")
        if op == "sched_pass":
            return "pass"

        items: list[tuple[str, bool]] = []
        if op == "session_response":
            items = [(str(payload.get("requester", "")), bool(payload.get("accept", False)))]
        elif op == "session_response_batch":
            raw = payload.get("responses") or payload.get("items") or []
            if not isinstance(raw, list):
                self._scheduling_log_violation(
                    "invalid_scheduling_payload_shape", responder=agent, negotiation_op=str(op)
                )
                return "invalid_payload"
            for it in raw:
                if not isinstance(it, dict):
                    self._scheduling_log_violation(
                        "invalid_scheduling_payload_item_skipped",
                        responder=agent,
                        negotiation_op=str(op),
                    )
                    continue
                items.append((str(it.get("requester", "")), bool(it.get("accept", False))))
            if not items:
                self._scheduling_log_violation(
                    "empty_scheduling_batch", responder=agent, negotiation_op=str(op)
                )
                return "empty_batch"
        else:
            self._scheduling_log_violation(
                "invalid_scheduling_response_op",
                responder=agent,
                negotiation_op=str(op),
            )
            return "invalid_op"

        any_ok = False
        for rq, accept in items:
            if self._store_scheduling_response_item(agent, rq, accept) == "ok":
                any_ok = True
        if not any_ok:
            self._scheduling_log_violation(
                "scheduling_batch_no_acknowledged_items", responder=agent, negotiation_op=str(op)
            )
        return "ok"

    def _rebuild_scheduling_digest(
        self, finals_tuple: tuple[frozenset[str], ...], trace: SchedulingResolveTrace
    ) -> None:
        """§290-296 personalized digest visible only per agent (combined with §3.2 privacy tail)."""
        finals_set = set(finals_tuple)
        entered_agents = {p for g in finals_tuple for p in g}
        fps_conflict = self._fps_in_conflict_pairs(trace)

        for viewer in self.agent_names:
            lines: list[str] = []
            lines.append(f"(day={self.day}, slot_k={self.slot}) personalized scheduling digest:")

            for inv in sorted(self._invites.values(), key=lambda r: r.requester):
                if inv.requester != viewer:
                    continue
                fp = frozenset(inv.proposed_participants)
                key = self._invite_tuple_key(inv)
                resp = self._responses.get(key, {})
                others = [p for p in fp if p != viewer]
                all_ok = all(resp.get(o, False) for o in others) if others else False
                if fp in finals_set:
                    desc = (
                        "ACCEPTANCE complete — session on this roster is scheduled for this slot."
                    )
                elif all_ok:
                    if any(fp < G for G in finals_set):
                        desc = (
                            "All invitees ACCEPTED — smaller roster absorbed by strictly larger overlapping "
                            "accepted session (same slot)."
                        )
                    elif fp in fps_conflict or (
                        fp not in finals_set and not any(fp < G for G in finals_set)
                    ):
                        desc = (
                            "All ACCEPTED — request dropped by resolver due to incompatible overlaps "
                            "(no subset relation)."
                        )
                    else:
                        desc = "All ACCEPTED — request not instantiated (consult resolution trace)."
                else:
                    desc = "NOT scheduled — insufficient acceptances (effective decline/no_response)."
                lines.append(f"- Your invite → {sorted(fp)}: {desc} purpose_note={inv.purpose!r}")

            for inv in sorted(self._invites.values(), key=lambda r: r.requester):
                if viewer == inv.requester:
                    continue
                if viewer not in inv.proposed_participants:
                    continue
                key = self._invite_tuple_key(inv)
                eff_accept = bool(self._responses.setdefault(key, {}).get(viewer, False))
                lines.append(
                    f"- Request from `{inv.requester}` with roster {sorted(inv.proposed_participants)}: "
                    f"effective_response={'accept' if eff_accept else 'decline_or_no_response'}"
                )

            if viewer in entered_agents:
                my_rosters = [sorted(g) for g in finals_tuple if viewer in g]
                if my_rosters:
                    lines.append(
                        "- You ENTER a negotiating session this slot — your roster cluster(s): "
                        f"{my_rosters!r}."
                    )
            else:
                lines.append(
                    "- You DO NOT enter a session this slot. "
                    "(§3.2: no unsolicited visibility of other negotiation groups.)"
                )
            self._last_scheduling_digest[viewer] = "\n".join(lines)

    def ensure_session_structure(self) -> None:
        sess = self._current_session()
        if sess is None:
            if self.phase == Phase.SESSION:
                self._enter_post_session()
            return
        if len(sess.participants) < 2:
            self._pending_session_close_reason = "insufficient_participants"
            self._advance_session_or_post()

    # ---- scheduling collect ------------------------------------------------

    def submit_invite_json(self, agent: str, payload: dict[str, Any]) -> str:
        if self.terminal or self.phase == Phase.TERMINATED:
            return "ignored: world_terminal"
        if self.phase != Phase.SCHEDULE_INVITE:
            return "ignored: wrong phase"
        op = payload.get("negotiation_op")
        if op == "sched_pass":
            return "pass"
        if op != "session_request":
            return "invalid_op"
        pp = payload.get("proposed_participants") or []
        if not isinstance(pp, list) or len(pp) < 2:
            return "invalid_participants"
        pset = frozenset(str(x) for x in pp)
        if agent not in pset:
            return "requester_not_in_set"
        bad = [x for x in pset if x not in self.agent_names]
        if bad:
            return f"unknown_agents:{bad}"
        if agent in self._invites:
            return "already_submitted"
        self._invites[agent] = SessionInviteRecord(
            requester=agent,
            proposed_participants=pset,
            purpose=str(payload.get("purpose", ""))[:500],
            slot=self.slot,
            day=self.day,
        )
        return "ok"

    def submit_response_json(self, agent: str, payload: dict[str, Any]) -> str:
        """§3.x — JSON 单列 / 批量统一入口。"""
        return self.submit_scheduling_response_payload(agent, payload)

    def finish_invite_phase(self) -> None:
        if self.terminal or self.phase == Phase.TERMINATED:
            return
        self.phase = Phase.SCHEDULE_RESPONSE

    def resolve_scheduling(self) -> None:
        """§3 — normalize responses + deterministic resolver (subset / overlap / disjoint)."""
        if self.terminal or self.phase == Phase.TERMINATED:
            return
        self._session_busy.clear()

        self._impute_missing_responses_as_decline()
        self._apply_one_accept_per_invitee_constraint()

        r_rows: list[tuple[frozenset[str], SessionInviteRecord]] = []
        for inv in sorted(
            self._invites.values(),
            key=lambda r: (r.requester, tuple(sorted(r.proposed_participants))),
        ):
            fp = frozenset(inv.proposed_participants)
            fps = tuple(sorted(fp))
            others = [p for p in fps if p != inv.requester]
            if len(fp) < 2 or not others:
                continue
            key = self._invite_tuple_key(inv)
            resp = self._responses.setdefault(key, {})
            if not all(resp.get(o, False) for o in others):
                continue
            r_rows.append((fp, inv))

        valid_fps_list = [fp for fp, _ in r_rows]
        finals_tuple, trace = resolve_valid_session_sets(valid_fps_list)
        self._last_resolve_trace = trace

        self.scheduling_resolution_log.append(
            {
                "day": self.day,
                "slot": self.slot,
                "valid_candidates": sorted([sorted(s) for s in trace.valid_sets]),
                "after_merge_unique": sorted([sorted(s) for s in trace.merged_unique]),
                "after_subset_absorption": sorted([sorted(s) for s in trace.after_absorb]),
                "conflict_pairs_writer": [
                    {"left": sorted(a), "right": sorted(b)} for a, b in trace.conflict_pairs
                ],
                "final_sessions": sorted([sorted(s) for s in finals_tuple]),
            }
        )

        self.session_meta.clear()
        self._session_total_turns.clear()
        self._session_agent_turns.clear()
        self._session_last_actor.clear()
        self._resolved_sessions = []
        for P in finals_tuple:
            sid = uuid.uuid4().hex[:12]
            pstart = tuple(sorted(P))
            self._resolved_sessions.append(
                ResolvedSession(
                    session_id=sid,
                    day=self.day,
                    slot=self.slot,
                    participants=pstart,
                )
            )
            self.session_meta[sid] = SessionRuntimeMeta(
                session_id=sid,
                day=self.day,
                slot=self.slot,
                participants_start=pstart,
                t_start_global_turn=self._episode_atomic_turn,
            )

        self._session_busy = {p for rs in self._resolved_sessions for p in rs.participants}
        self._rebuild_scheduling_digest(finals_tuple, trace)

        if self._resolved_sessions:
            self.mark_structural_progress()
            self.phase = Phase.SESSION
            self._active_session_idx = 0
            self.session_round = 0
        else:
            self._enter_post_session()

    def _enter_post_session(self) -> None:
        self._flush_slot_transcript_to_logs()
        self.phase = Phase.POST_SESSION
        self.session_round = 0

    def advance_after_post_session(self) -> None:
        if self.phase != Phase.POST_SESSION:
            return
        if self.terminal:
            return
        if self.slot < self.params.s_max_per_day:
            self.slot += 1
            self._clear_slot_state()
            self.phase = Phase.SCHEDULE_INVITE
        else:
            self.phase = Phase.END_OF_DAY

    def end_day_tick(self, event_hook: Callable[[int], None] | None = None) -> None:
        if self.phase != Phase.END_OF_DAY:
            return
        if self.terminal:
            return
        closing_day = self.day
        lim = self.params.failure_stagnation_calendar_days
        if lim is not None and lim > 0:
            if not self._progress_flag_this_calendar_day:
                self._consecutive_idle_calendar_days += 1
                if self._consecutive_idle_calendar_days >= lim:
                    self._terminate("failure")
                    return
            else:
                self._consecutive_idle_calendar_days = 0
        if event_hook:
            event_hook(closing_day)
        self.day += 1
        self._progress_flag_this_calendar_day = False
        self._formal_actions_per_agent_day.clear()
        self.slot = 1
        self._clear_slot_state()
        if self.day > self.params.D:
            self._terminate("timeout")
            return
        self.phase = Phase.SCHEDULE_INVITE

    # ---- session mechanics -------------------------------------------------

    def current_actor_in_session(self) -> str | None:
        """§4.3 — 固定 role order 轮流出牌；跳过已达 ``K_s`` 的 participant。"""
        if self.phase == Phase.TERMINATED or self.terminal:
            return None
        sess = self._current_session()
        if sess is None:
            return None
        if len(sess.participants) < 2:
            return None
        sid = sess.session_id
        ordered = self._ordered_session_participants(sess)
        if len(ordered) < 2:
            return None

        T_s = self._effective_T_s(sess)
        N_s = self._session_total_turns.get(sid, 0)
        if N_s >= T_s:
            self._pending_session_close_reason = "session_max_total_turns_T_s"
            self._advance_session_or_post()
            return None

        K_s = self.params.max_turns_per_participant_per_session
        bt = self._session_agent_turns.get(sid, {})
        if K_s is not None and all(bt.get(p, 0) >= K_s for p in sess.participants):
            self._pending_session_close_reason = "all_participants_hit_K_s"
            self._advance_session_or_post()
            return None

        eligible = self._eligible_speakers_under_K_s(sid, sess)
        if len(eligible) < 2:
            self._pending_session_close_reason = "insufficient_eligible_speakers_under_K_caps"
            self._advance_session_or_post()
            return None

        last = self._session_last_actor.get(sid)
        eligible_set = set(eligible)
        if last is None:
            for a in ordered:
                if a in eligible_set:
                    return a
            return None

        try:
            li = ordered.index(last)
        except ValueError:
            li = -1
        for step in range(len(ordered)):
            cand = ordered[(li + 1 + step) % len(ordered)]
            if cand in eligible_set:
                return cand
        return None

    def on_session_turn_completed(self, actor: str, *, session_id: str | None = None) -> None:
        """每个 session macro turn 收尾后由 env 调用一次（含 leave/none/pass）。

        若 ``submit_session_payload`` 可能推进 session（如 leave），须传入本步开始时的
        ``session_id``，以免把计数记到下一个并行 session 上。
        """
        if session_id is None:
            if self.terminal:
                return
            if self.phase != Phase.SESSION:
                return
            sess = self._current_session()
            if sess is None:
                return
            sid = sess.session_id
        else:
            sid = session_id

        self._session_total_turns[sid] = self._session_total_turns.get(sid, 0) + 1
        bucket = self._session_agent_turns.setdefault(sid, {})
        bucket[actor] = bucket.get(actor, 0) + 1
        self._session_last_actor[sid] = actor
        self._episode_atomic_turn += 1

        cur = self._current_session()
        if cur is not None and cur.session_id == sid:
            self.session_round = self._session_total_turns[sid]

    def record_session_turn(self, agent: str, action_type: str, content: str) -> None:
        """会话轮次记录（含 §2.1 natural message 预算）；formal JSON 仅占 action 信道，不占此项。"""
        if self.phase == Phase.TERMINATED or self.terminal:
            return
        sess = self._current_session()
        if sess is None:
            return

        tracked = ("speak", "non-verbal communication")
        if action_type in tracked:
            lim = self.params.max_natural_turns_per_agent_per_session
            sid = sess.session_id
            if lim is not None:
                by_agent = self._nl_turn_counts.setdefault(sid, {})
                if by_agent.get(agent, 0) >= lim:
                    self._audit_action(
                        agent=agent,
                        negotiation_op=None,
                        verb=None,
                        valid=False,
                        reason="session_natural_message_budget_exceeded",
                        extra={"limit": lim, "session_id": sid},
                    )
                    content = f"[budget_blocked nl_cap={lim}] {content}"
                else:
                    by_agent[agent] = by_agent.get(agent, 0) + 1

        viewers = tuple(sess.participants)
        self._transcript.append((sess.session_id, agent, content, viewers))
        self._audit_message_line(sess.session_id, agent, content)

    def record_session_message(self, agent: str, content: str) -> None:
        """兼容旧接口；不记入 natural budget（typed channel 请用 ``record_session_turn``）。"""
        self.record_session_turn(agent, "none", content)

    def submit_session_payload(
        self,
        agent: str,
        payload: dict[str, Any],
        *,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
    ) -> None:
        op_any = payload.get("negotiation_op")
        op = str(op_any) if op_any is not None else ""
        sess = self._current_session()

        if self.phase == Phase.TERMINATED or self.terminal:
            self._audit_action(
                agent=agent,
                negotiation_op=op,
                verb=str(payload.get("verb")) if op == "formal" else None,
                valid=False,
                reason="world_terminal",
            )
            return

        if op == "terminate_negotiation":
            if sess is None or agent not in sess.participants:
                self._audit_action(
                    agent=agent,
                    negotiation_op=op,
                    verb=None,
                    valid=False,
                    reason="terminate_requires_active_session_participant",
                )
                return
            if not self._formal_budget_allows_increment(agent, verb="terminate_negotiation"):
                return
            if agent in PRINCIPAL_PARTY_ROLES:
                self._terminate("terminated_by_agent")
                self._formal_budget_commit_increment(agent)
                self._audit_action(agent=agent, negotiation_op=op, verb=None, valid=True, reason="world_terminated")
                return
            if agent == "investor":
                self.investor_financing_path_withdrawn = True
                self._formal_budget_commit_increment(agent)
                self.append_event_records(
                    [{"kind": "negotiation_path", "note": "investor_withdrew_financing_path", "day": self.day, "slot": self.slot}]
                )
                self._audit_action(agent=agent, negotiation_op=op, verb=None, valid=True, reason="investor_financing_withdrawn")
                return
            if agent == "regulator":
                self.regulator_regulatory_path_withdrawn = True
                self._formal_budget_commit_increment(agent)
                self.append_event_records(
                    [{"kind": "negotiation_path", "note": "regulator_withdrew_regulatory_path", "day": self.day, "slot": self.slot}]
                )
                self._audit_action(agent=agent, negotiation_op=op, verb=None, valid=True, reason="regulator_review_withdrawn")
                return
            self._audit_action(
                agent=agent,
                negotiation_op=op,
                verb=None,
                valid=False,
                reason="terminate_negotiation_unhandled_role",
            )
            return

        sc_verb_raw = payload.get("verb") if op == "session_control" else None
        sc_verb = str(sc_verb_raw) if sc_verb_raw is not None else ""
        if op == "session_control" and sc_verb in ("leave", "terminate_session"):
            verb_for_log = "leave" if sc_verb == "leave" else "terminate_session"
            if sess is None or agent not in sess.participants:
                self._audit_action(
                    agent=agent,
                    negotiation_op=op,
                    verb=verb_for_log,
                    valid=False,
                    reason="leave_requires_active_session_member",
                )
                return
            if not self._session_control_budget_allows(agent, verb=verb_for_log):
                return
            self._remove_from_active_session(agent)
            self._session_control_budget_commit(agent)
            self._audit_action(
                agent=agent, negotiation_op=op, verb=verb_for_log, valid=True, reason="ok_leave_session"
            )
            return

        if op == "formal":
            verb_raw = payload.get("verb")
            verb = str(verb_raw) if verb_raw is not None else ""
            if sess is None or agent not in sess.participants:
                self._audit_action(
                    agent=agent,
                    negotiation_op=op,
                    verb=verb or None,
                    valid=False,
                    reason="formal_requires_active_session_member",
                )
                return
            if verb == "propose_contract":
                self._formal_propose(agent, payload, resources_snapshot)
            elif verb == "sign":
                self._formal_sign(agent, payload, resources_snapshot)
            elif verb == "accept":
                self._formal_accept(agent, payload)
            elif verb == "reject_contract":
                self._formal_reject_contract(agent, payload)
            elif verb == "amend_contract":
                self._formal_amend(agent, payload, resources_snapshot)
            elif verb == "request_financing_review":
                self._formal_request_financing_review(agent, payload)
            elif verb == "request_regulatory_review":
                self._formal_request_regulatory_review(agent, payload)
            elif verb == "contract_share":
                self._formal_contract_share(agent, payload)
            elif verb == "finance_commit":
                self.submit_financing_commit(
                    agent, resources_snapshot, contract_id=payload.get("contract_id")
                )
                self._reevaluate_success(resources_snapshot)
            elif verb == "finance_decline":
                self.submit_finance_decline(agent, resources_snapshot, payload)
            elif verb == "regulatory_approve":
                self.submit_regulator_approve(
                    agent, resources_snapshot, contract_id=payload.get("contract_id")
                )
                self._reevaluate_success(resources_snapshot)
            elif verb == "regulatory_block":
                self.submit_regulatory_block(agent, resources_snapshot, payload)
            else:
                self._audit_action(
                    agent=agent,
                    negotiation_op=op,
                    verb=verb or None,
                    valid=False,
                    reason="unknown_formal_verb",
                )
            return

        self._audit_action(
            agent=agent,
            negotiation_op=op,
            verb=None,
            valid=False,
            reason="unknown_or_missing_negotiation_op",
        )

    def _remove_from_active_session(self, agent: str) -> None:
        sess = self._current_session()
        if sess is None:
            return
        parts = list(sess.participants)
        if agent not in parts:
            return
        parts = [p for p in parts if p != agent]
        new_sess = ResolvedSession(
            session_id=sess.session_id,
            day=sess.day,
            slot=sess.slot,
            participants=tuple(parts),
        )
        self._resolved_sessions[self._active_session_idx] = new_sess
        if len(parts) < 2:
            self._pending_session_close_reason = "insufficient_participants_after_leave"
            self._advance_session_or_post()

    def _advance_session_or_post(self) -> None:
        if (
            self.phase == Phase.SESSION
            and self._active_session_idx < len(self._resolved_sessions)
        ):
            sess = self._resolved_sessions[self._active_session_idx]
            meta = self.session_meta.get(sess.session_id)
            if meta is not None and meta.status == "active":
                meta.t_end_global_turn = self._episode_atomic_turn
                meta.status = "closed"
            reason = self._pending_session_close_reason
            self._slot_session_close_records.append(
                {
                    "session_id": sess.session_id,
                    "day": self.day,
                    "slot": self.slot,
                    "participants_final": list(sess.participants),
                    "end_reason": reason,
                    "t_end_global_turn": self._global_turn_number(),
                }
            )
            self._pending_session_close_reason = "session_closed"

        self._active_session_idx += 1
        self.session_round = 0
        if self._active_session_idx >= len(self._resolved_sessions):
            self._enter_post_session()
        else:
            self.phase = Phase.SESSION

    def advance_session_turn(self) -> None:
        """§4.3 — ``on_session_turn_completed`` 已更新 ``N_s`` / ``K_i`` 后复检收口条件。"""
        sess = self._current_session()
        if sess is None:
            return
        sid = sess.session_id
        T_s = self._effective_T_s(sess)
        N_s = self._session_total_turns.get(sid, 0)
        if N_s >= T_s:
            self._pending_session_close_reason = "session_max_total_turns_T_s"
            self._advance_session_or_post()
            return
        K_s = self.params.max_turns_per_participant_per_session
        if K_s is not None and sess.participants:
            bt = self._session_agent_turns.get(sid, {})
            if all(bt.get(p, 0) >= K_s for p in sess.participants):
                self._pending_session_close_reason = "all_participants_hit_K_s"
                self._advance_session_or_post()
                return
        eligible = self._eligible_speakers_under_K_s(sid, sess)
        if len(eligible) < 2:
            self._pending_session_close_reason = "insufficient_eligible_speakers_under_K_caps"
            self._advance_session_or_post()

    def _contract_live(self, c: NegotiationContract) -> bool:
        """仍可被引用的合同（未成文终止 / 未被取代链路废弃）。"""
        return c.status not in ("signed", "superseded", "rejected", "failed")

    def _contract_agent_sees(self, agent: str, c: NegotiationContract) -> bool:
        """§5 — ``visible_to(C, i)``。"""
        return agent in c.visibility

    def _contract_append_history(
        self,
        c: NegotiationContract,
        event: str,
        agent: str,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        sess = self._current_session()
        row: dict[str, Any] = {
            "event": event,
            "day": self.day,
            "slot_id": self.slot,
            "session_id": sess.session_id if sess else None,
            "turn_id": self._episode_atomic_turn,
            "agent": agent,
        }
        if detail:
            row["detail"] = detail
        c.history.append(row)

    def _principal_acceptance_complete(self, c: NegotiationContract) -> bool:
        principals = sorted(PRINCIPAL_PARTY_ROLES & c.parties)
        return bool(principals) and all(c.acceptances.get(p) is True for p in principals)

    def _formal_propose(
        self,
        agent: str,
        payload: dict[str, Any],
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
    ) -> None:
        sess = self._current_session()
        if sess is None or agent not in sess.participants:
            return
        if roster_blocks_trade_contract_drafting(sess.participants):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="propose_contract",
                valid=False,
                reason="session_roster_v7_5_blocks_trade_contract_drafting",
                extra={"participants": sorted(sess.participants)},
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="propose_contract"):
            return
        cid = uuid.uuid4().hex[:14]
        res = resources_snapshot()
        terms = dict(payload.get("terms") or {})

        if self.params.enforce_formal_budget_checks:
            pr = terms.get("price", 0)
            try:
                price_check = float(pr or 0)
            except (TypeError, ValueError):
                self._audit_action(
                    agent=agent,
                    negotiation_op="formal",
                    verb="propose_contract",
                    valid=False,
                    reason="invalid_price_field",
                )
                return
            if price_check < 0:
                self._audit_action(
                    agent=agent,
                    negotiation_op="formal",
                    verb="propose_contract",
                    valid=False,
                    reason="negative_price_not_allowed",
                )
                return

        terms["_cash_firm_a_snapshot"] = float(res.get("firm_a", {}).get("cash", 0.0))
        vis = set(sess.participants)
        # § contracts §5.2 — 合同主体 = (PRINCIPAL_PARTY_ROLES ∩ session.participants)。
        # 双方 lineup 时退化成 {firm_a, firm_b}；3 公司 firms_only 时为 {firm_a, firm_b, firm_c}。
        parties = set(PRINCIPAL_PARTY_ROLES) & set(sess.participants)
        if not parties:
            parties = set(PRINCIPAL_PARTY_ROLES) & set(self.agent_names)
        acceptances_dict = {p: None for p in sorted(parties)}
        created_at = {
            "day": self.day,
            "slot_id": self.slot,
            "session_id": sess.session_id,
            "turn_id": self._episode_atomic_turn,
        }
        nc = NegotiationContract(
            contract_id=cid,
            parent_id=None,
            status="proposed",
            terms=terms,
            created_by=agent,
            created_at=dict(created_at),
            parties=set(parties),
            acceptances=dict(acceptances_dict),
            visibility=vis,
            signatures={},
            financing={"required": 0, "status": "not_required", "actor": None},
            regulatory={"required": 0, "status": "not_required", "actor": None},
            history=[],
            created_day=self.day,
            created_slot=self.slot,
        )
        self._finalize_financing_reg_flags(nc)
        self.contracts[cid] = nc
        if self.primary_contract_id is None:
            self.primary_contract_id = cid
        self._contract_append_history(nc, "contract.propose", agent, detail={"contract_id": cid})
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="propose_contract",
            valid=True,
            reason="proposed",
            extra={"contract_id": cid},
        )
        self.record_execution_event(
            "contract_proposed",
            f"合同草案已提出（contract_id={cid}，发起方={agent}）",
            contract_id=cid,
            agent=agent,
            status="proposed",
        )
        self.mark_structural_progress()

    def _formal_accept(self, agent: str, payload: dict[str, Any]) -> None:
        cid = str(payload.get("contract_id", self.primary_contract_id or ""))
        c = self.contracts.get(cid)
        if not c:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="accept",
                valid=False,
                reason="unknown_contract_id",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_agent_sees(agent, c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="accept",
                valid=False,
                reason="contract_not_visible",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_live(c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="accept",
                valid=False,
                reason="contract_not_live",
                extra={"contract_id": cid, "status": c.status},
            )
            return
        if agent not in c.parties or agent not in PRINCIPAL_PARTY_ROLES:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="accept",
                valid=False,
                reason="accept_reserved_for_principal_parties",
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="accept"):
            return
        c.acceptances[agent] = True
        self._contract_append_history(c, "contract.accept", agent, detail={"contract_id": cid})
        if self._principal_acceptance_complete(c):
            c.status = "accepted"
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="accept",
            valid=True,
            reason="accept_recorded",
            extra={"contract_id": cid, "status_after": c.status},
        )
        self.record_execution_event(
            "contract_accept",
            f"主体方 {agent} 接受合同条款（contract_id={cid}，当前状态={c.status!r}）",
            contract_id=cid,
            agent=agent,
            status_after=c.status,
        )
        self.mark_structural_progress()

    def _formal_reject_contract(self, agent: str, payload: dict[str, Any]) -> None:
        cid = str(payload.get("contract_id", self.primary_contract_id or ""))
        c = self.contracts.get(cid)
        if not c:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="reject_contract",
                valid=False,
                reason="unknown_contract_id",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_agent_sees(agent, c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="reject_contract",
                valid=False,
                reason="contract_not_visible",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_live(c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="reject_contract",
                valid=False,
                reason="contract_not_live",
                extra={"contract_id": cid},
            )
            return
        if agent not in c.parties or agent not in PRINCIPAL_PARTY_ROLES:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="reject_contract",
                valid=False,
                reason="reject_reserved_for_principal_parties",
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="reject_contract"):
            return
        c.acceptances[agent] = False
        c.status = "rejected"
        self._contract_append_history(c, "contract.reject", agent, detail={"contract_id": cid})
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="reject_contract",
            valid=True,
            reason="rejected",
            extra={"contract_id": cid},
        )
        self.mark_structural_progress()

    def _formal_amend(
        self,
        agent: str,
        payload: dict[str, Any],
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
    ) -> None:
        sess = self._current_session()
        if sess is None or agent not in sess.participants:
            return
        if roster_blocks_trade_contract_drafting(sess.participants):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="amend_contract",
                valid=False,
                reason="session_roster_v7_5_blocks_trade_contract_amend",
                extra={"participants": sorted(sess.participants)},
            )
            return
        parent_cid = str(payload.get("contract_id", self.primary_contract_id or ""))
        parent = self.contracts.get(parent_cid)
        if not parent or not self._contract_agent_sees(agent, parent):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="amend_contract",
                valid=False,
                reason="amend_parent_not_visible",
                extra={"contract_id": parent_cid},
            )
            return
        if not self._contract_live(parent):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="amend_contract",
                valid=False,
                reason="parent_not_amendable",
                extra={"contract_id": parent_cid},
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="amend_contract"):
            return
        cid = uuid.uuid4().hex[:14]
        res = resources_snapshot()
        terms = dict(parent.terms)
        terms.update(dict(payload.get("terms") or {}))

        if self.params.enforce_formal_budget_checks:
            pr = terms.get("price", 0)
            try:
                price_check = float(pr or 0)
            except (TypeError, ValueError):
                self._audit_action(
                    agent=agent,
                    negotiation_op="formal",
                    verb="amend_contract",
                    valid=False,
                    reason="invalid_price_field",
                    extra={"contract_id": parent_cid},
                )
                return
            if price_check < 0:
                self._audit_action(
                    agent=agent,
                    negotiation_op="formal",
                    verb="amend_contract",
                    valid=False,
                    reason="negative_price_not_allowed",
                )
                return

        terms["_cash_firm_a_snapshot"] = float(res.get("firm_a", {}).get("cash", 0.0))
        vis = set(sess.participants)
        acceptances_dict = {p: None for p in sorted(parent.parties)}
        created_at = {
            "day": self.day,
            "slot_id": self.slot,
            "session_id": sess.session_id,
            "turn_id": self._episode_atomic_turn,
        }
        nc = NegotiationContract(
            contract_id=cid,
            parent_id=parent.contract_id,
            status="amended",
            terms=terms,
            created_by=agent,
            created_at=dict(created_at),
            parties=set(parent.parties),
            acceptances=dict(acceptances_dict),
            visibility=vis,
            signatures={},
            financing={"required": 0, "status": "not_required", "actor": None},
            regulatory={"required": 0, "status": "not_required", "actor": None},
            history=[],
            created_day=self.day,
            created_slot=self.slot,
        )
        self._finalize_financing_reg_flags(nc)
        parent.status = "superseded"
        self._contract_append_history(
            parent, "contract.superseded", agent, detail={"child_contract_id": cid}
        )
        self.contracts[cid] = nc
        if self.primary_contract_id == parent.contract_id:
            self.primary_contract_id = cid
        self._contract_append_history(
            nc, "contract.amend", agent, detail={"parent_contract_id": parent.contract_id}
        )
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="amend_contract",
            valid=True,
            reason="amend_created",
            extra={"parent_contract_id": parent.contract_id, "contract_id": cid},
        )
        self.mark_structural_progress()

    def _formal_request_financing_review(self, agent: str, payload: dict[str, Any]) -> None:
        sess = self._current_session()
        if sess is None or agent not in sess.participants:
            return
        cid = str(payload.get("contract_id", self.primary_contract_id or ""))
        c = self.contracts.get(cid)
        if not c or not self._contract_agent_sees(agent, c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="request_financing_review",
                valid=False,
                reason="contract_not_visible",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_live(c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="request_financing_review",
                valid=False,
                reason="contract_not_live",
                extra={"contract_id": cid},
            )
            return
        if "investor" not in self.agent_names:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="request_financing_review",
                valid=False,
                reason="investor_role_absent_from_world",
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="request_financing_review"):
            return
        c.visibility.add("investor")
        c.financing["actor"] = "investor"
        self._contract_append_history(c, "review.request_financing", agent, detail={"contract_id": cid})
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="request_financing_review",
            valid=True,
            reason="investor_added_visibility",
            extra={"contract_id": cid},
        )
        self.mark_structural_progress()

    def _formal_request_regulatory_review(self, agent: str, payload: dict[str, Any]) -> None:
        sess = self._current_session()
        if sess is None or agent not in sess.participants:
            return
        cid = str(payload.get("contract_id", self.primary_contract_id or ""))
        c = self.contracts.get(cid)
        if not c or not self._contract_agent_sees(agent, c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="request_regulatory_review",
                valid=False,
                reason="contract_not_visible",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_live(c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="request_regulatory_review",
                valid=False,
                reason="contract_not_live",
                extra={"contract_id": cid},
            )
            return
        if "regulator" not in self.agent_names:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="request_regulatory_review",
                valid=False,
                reason="regulator_role_absent_from_world",
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="request_regulatory_review"):
            return
        c.visibility.add("regulator")
        c.regulatory["actor"] = "regulator"
        self._contract_append_history(c, "review.request_regulatory", agent, detail={"contract_id": cid})
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="request_regulatory_review",
            valid=True,
            reason="regulator_added_visibility",
            extra={"contract_id": cid},
        )
        self.mark_structural_progress()

    def _formal_contract_share(self, agent: str, payload: dict[str, Any]) -> None:
        sess = self._current_session()
        if sess is None or agent not in sess.participants:
            return
        cid = str(payload.get("contract_id", self.primary_contract_id or ""))
        receiver = str(payload.get("receiver", ""))
        c = self.contracts.get(cid)
        if not c or not self._contract_agent_sees(agent, c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="contract_share",
                valid=False,
                reason="contract_not_visible",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_live(c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="contract_share",
                valid=False,
                reason="contract_not_live",
                extra={"contract_id": cid},
            )
            return
        if receiver not in self.agent_names or receiver not in sess.participants:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="contract_share",
                valid=False,
                reason="share_receiver_must_be_session_participant",
                extra={"receiver": receiver},
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="contract_share"):
            return
        c.visibility.add(receiver)
        self._contract_append_history(
            c, "contract.share", agent, detail={"contract_id": cid, "receiver": receiver}
        )
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="contract_share",
            valid=True,
            reason="visibility_extended",
            extra={"contract_id": cid, "receiver": receiver},
        )
        self.mark_structural_progress()

    def _formal_sign(
        self,
        agent: str,
        payload: dict[str, Any],
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
    ) -> None:
        cid = str(payload.get("contract_id", self.primary_contract_id or ""))
        c = self.contracts.get(cid)
        if not c:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="sign",
                valid=False,
                reason="unknown_contract_id",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_agent_sees(agent, c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="sign",
                valid=False,
                reason="contract_not_visible",
                extra={"contract_id": cid},
            )
            return
        if not self._contract_live(c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="sign",
                valid=False,
                reason="contract_not_live",
                extra={"contract_id": cid},
            )
            return
        if c.status != "accepted" or not self._principal_acceptance_complete(c):
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="sign",
                valid=False,
                reason="sign_requires_prior_mutual_principal_accept",
                extra={"contract_id": cid},
            )
            return
        if agent not in PRINCIPAL_PARTY_ROLES or agent not in c.parties:
            self._audit_action(
                agent=agent,
                negotiation_op="formal",
                verb="sign",
                valid=False,
                reason="sign_reserved_for_principal_parties",
                extra={"contract_id": cid},
            )
            return
        if not self._formal_budget_allows_increment(agent, verb="sign"):
            return
        c.signatures[agent] = True
        self._contract_append_history(c, "contract.sign", agent, detail={"contract_id": cid})
        self._formal_budget_commit_increment(agent)
        self._audit_action(
            agent=agent,
            negotiation_op="formal",
            verb="sign",
            valid=True,
            reason="signed_partial",
            extra={"contract_id": cid},
        )
        self.record_execution_event(
            "contract_sign_partial",
            f"主体方 {agent} 完成签署（contract_id={cid}；是否全员签满视融资/监管条件）",
            contract_id=cid,
            agent=agent,
        )
        self.mark_structural_progress()
        self._maybe_finalize_success(c, resources_snapshot)

    def _finalize_financing_reg_flags(self, c: NegotiationContract) -> None:
        price = float(c.terms.get("price", 0) or 0)
        cash_a = float(c.terms.get("_cash_firm_a_snapshot", 0) or 0)
        need_by_budget = price > cash_a + self.params.financing_buffer
        tr = c.terms
        #: §9 — FinancingRequired = 1 iff 现金不足以覆盖名义对价 OR 合同显式含融资附条件
        explicit_financing = bool(int(tr.get("financing_required", 0) or 0)) or bool(
            int(tr.get("financing_contingent", 0) or 0)
        )
        need_fin = bool(need_by_budget or explicit_financing)
        c.financing["required"] = 1 if need_fin else 0
        prev_f = str(c.financing.get("status", "") or "")
        if not need_fin:
            c.financing["status"] = "not_required"
            c.financing["actor"] = None
        elif prev_f in ("committed", "declined"):
            c.financing["actor"] = c.financing.get("actor") or (
                "investor" if "investor" in self.agent_names else "rule_engine"
            )
        else:
            c.financing["status"] = "pending"
            c.financing["actor"] = "investor" if "investor" in self.agent_names else "rule_engine"
        r_flag = int(c.terms.get("regulatory_required", 0) or 0)
        c.regulatory["required"] = r_flag
        prev_r = str(c.regulatory.get("status", "") or "")
        if not r_flag:
            c.regulatory["status"] = "not_required"
            c.regulatory["actor"] = None
        elif prev_r in ("approved", "blocked"):
            c.regulatory["actor"] = c.regulatory.get("actor") or (
                "regulator" if "regulator" in self.agent_names else "rule_engine"
            )
        else:
            c.regulatory["status"] = "pending"
            c.regulatory["actor"] = "regulator" if "regulator" in self.agent_names else "rule_engine"

    def _auto_rule_engine_reviews(
        self,
        c: NegotiationContract,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
    ) -> None:
        """当无 investor/regulator 角色时，用 deterministic 规则自动完成融资/监管检查。"""
        if c.status != "accepted":
            return
        res = resources_snapshot()
        if (
            int(c.financing.get("required", 0)) == 1
            and c.financing.get("actor") == "rule_engine"
            and c.financing.get("status") == "pending"
        ):
            price = float(c.terms.get("price", 0) or 0)
            cash_a = float(res.get("firm_a", {}).get("cash", 0.0))
            gap = max(0.0, price - cash_a - float(self.params.financing_buffer))
            # rule: 融资缺口过大则拒绝；否则自动承诺
            if gap > max(200.0, price * 0.35):
                c.financing["status"] = "declined"
                self._contract_append_history(
                    c,
                    "financing.declined_by_rule_engine",
                    "__system__",
                    detail={"contract_id": c.contract_id, "financing_gap": gap},
                )
            else:
                c.financing["status"] = "committed"
                self._contract_append_history(
                    c,
                    "financing.committed_by_rule_engine",
                    "__system__",
                    detail={"contract_id": c.contract_id, "financing_gap": gap},
                )
        if (
            int(c.regulatory.get("required", 0)) == 1
            and c.regulatory.get("actor") == "rule_engine"
            and c.regulatory.get("status") == "pending"
        ):
            hard = int(c.terms.get("policy_hard_violation", 0) or 0)
            comp = float(c.terms.get("compliance_score", 0.75) or 0.75)
            if hard == 1 or comp < 0.45:
                c.regulatory["status"] = "blocked"
                c.status = "failed"
                self._contract_append_history(
                    c,
                    "regulatory.blocked_by_rule_engine",
                    "__system__",
                    detail={"contract_id": c.contract_id, "hard_violation": hard, "compliance_score": comp},
                )
            else:
                c.regulatory["status"] = "approved"
                self._contract_append_history(
                    c,
                    "regulatory.approved_by_rule_engine",
                    "__system__",
                    detail={"contract_id": c.contract_id, "compliance_score": comp},
                )

    def refresh_contract_contingencies_from_resources(
        self, resources: dict[str, dict[str, float]]
    ) -> None:
        """在 sign 前根据当前现金刷新融资/监管必需标记。"""
        for c in self.contracts.values():
            if not self._contract_live(c):
                continue
            ca = float(resources.get("firm_a", {}).get("cash", 0.0))
            c.terms["_cash_firm_a_snapshot"] = ca
            self._finalize_financing_reg_flags(c)

    def _maybe_finalize_success(
        self,
        c: NegotiationContract,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
    ) -> None:
        if self.terminal:
            return
        self.refresh_contract_contingencies_from_resources(resources_snapshot())
        self._auto_rule_engine_reviews(c, resources_snapshot)
        if c.status != "accepted" or not self._principal_acceptance_complete(c):
            return
        principals = sorted(PRINCIPAL_PARTY_ROLES & c.parties)
        if not principals:
            return
        # 终局成功要求 ``c.parties`` 内每名 principal 都签署。
        if not all(c.signatures.get(p, False) for p in principals):
            return
        if int(c.financing.get("required", 0)) == 1:
            if self.investor_financing_path_withdrawn:
                return
            if c.financing.get("status") == "declined":
                return
            if c.financing.get("status") != "committed":
                return
        if int(c.regulatory.get("required", 0)) == 1:
            if self.regulator_regulatory_path_withdrawn:
                return
            if c.regulatory.get("status") != "approved":
                return
        c.status = "signed"
        self._contract_append_history(c, "contract.closed_signed", "__system__", detail={})
        self.record_execution_event(
            "contract_fully_signed",
            "合同已完全生效（全部主体签署且融资/监管等附条件满足）；世界继续运行",
            contract_id=c.contract_id,
            status="signed",
        )
        self.mark_structural_progress()

    def _reevaluate_success(
        self,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
    ) -> None:
        for c in list(self.contracts.values()):
            if not self._contract_live(c):
                continue
            if c.status != "accepted":
                continue
            self._maybe_finalize_success(c, resources_snapshot)

    def submit_financing_commit(
        self,
        investor: str,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
        *,
        contract_id: Any | None = None,
    ) -> None:
        if investor != "investor":
            self._audit_action(
                agent=investor,
                negotiation_op="formal",
                verb="finance_commit",
                valid=False,
                reason="role_must_be_investor",
            )
            return

        if self.investor_financing_path_withdrawn:
            self._audit_action(
                agent=investor,
                negotiation_op="formal",
                verb="finance_commit",
                valid=False,
                reason="investor_withdrew_from_financing_path",
            )
            return

        res = resources_snapshot()
        inv = res.get("investor", {})
        dc = float(inv.get("deployable_capital", inv.get("cash", 0.0)))
        want = str(contract_id).strip() if contract_id not in (None, "") else None

        for c in self.contracts.values():
            if want is not None and c.contract_id != want:
                continue
            if not self._contract_live(c):
                continue
            if investor not in c.visibility:
                continue
            if int(c.financing.get("required", 0)) != 1:
                continue
            if c.financing.get("status") == "declined":
                continue
            if c.financing.get("status") == "committed":
                continue
            price = float(c.terms.get("price", 0) or 0)
            cash_a = float(res.get("firm_a", {}).get("cash", 0.0))
            need = max(0.0, price - cash_a - float(self.params.financing_buffer))

            if self.params.enforce_formal_budget_checks and need > dc + 1e-9:
                self._audit_action(
                    agent=investor,
                    negotiation_op="formal",
                    verb="finance_commit",
                    valid=False,
                    reason="insufficient_deployable_capital_for_financing_gap",
                    extra={"required_gap": need, "deployable_capital": dc},
                )
                return
            if not self._formal_budget_allows_increment(investor, verb="finance_commit"):
                return
            c.financing["status"] = "committed"
            self._contract_append_history(
                c, "financing.committed", investor, detail={"contract_id": c.contract_id}
            )
            self._formal_budget_commit_increment(investor)
            self._audit_action(
                agent=investor,
                negotiation_op="formal",
                verb="finance_commit",
                valid=True,
                reason="committed",
                extra={"contract_id": c.contract_id},
            )
            self.record_execution_event(
                "financing_committed",
                f"投资方确认融资承诺（contract_id={c.contract_id}）",
                contract_id=c.contract_id,
                agent=investor,
            )
            self.mark_structural_progress()
            return

        self._audit_action(
            agent=investor,
            negotiation_op="formal",
            verb="finance_commit",
            valid=False,
            reason="no_contract_pending_financing_for_visible_target",
            extra={"filter_contract_id": want},
        )

    def submit_finance_decline(
        self,
        investor: str,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
        payload: dict[str, Any],
    ) -> None:
        """§6.4.3 ``commit.finance_decline``；committed 之后不可 decline。"""
        _ = resources_snapshot
        if investor != "investor":
            self._audit_action(
                agent=investor,
                negotiation_op="formal",
                verb="finance_decline",
                valid=False,
                reason="role_must_be_investor",
            )
            return
        if self.investor_financing_path_withdrawn:
            self._audit_action(
                agent=investor,
                negotiation_op="formal",
                verb="finance_decline",
                valid=False,
                reason="investor_withdrew_from_financing_path",
            )
            return
        want = str(payload.get("contract_id")).strip() if payload.get("contract_id") not in (None, "") else None
        rsn = str(payload.get("reason", ""))[:500]

        for c in self.contracts.values():
            if want is not None and c.contract_id != want:
                continue
            if not self._contract_live(c):
                continue
            if investor not in c.visibility:
                continue
            if int(c.financing.get("required", 0)) != 1:
                continue
            st = str(c.financing.get("status", ""))
            if st == "committed":
                self._audit_action(
                    agent=investor,
                    negotiation_op="formal",
                    verb="finance_decline",
                    valid=False,
                    reason="cannot_decline_after_finance_commit",
                    extra={"contract_id": c.contract_id},
                )
                return
            if st == "declined":
                self._audit_action(
                    agent=investor,
                    negotiation_op="formal",
                    verb="finance_decline",
                    valid=True,
                    reason="already_declined",
                    extra={"contract_id": c.contract_id},
                )
                return
            if not self._formal_budget_allows_increment(investor, verb="finance_decline"):
                return
            c.financing["status"] = "declined"
            self._contract_append_history(
                c,
                "financing.declined",
                investor,
                detail={"contract_id": c.contract_id, "reason": rsn},
            )
            self._formal_budget_commit_increment(investor)
            self._audit_action(
                agent=investor,
                negotiation_op="formal",
                verb="finance_decline",
                valid=True,
                reason="declined",
                extra={"contract_id": c.contract_id},
            )
            self.mark_structural_progress()
            return

        self._audit_action(
            agent=investor,
            negotiation_op="formal",
            verb="finance_decline",
            valid=False,
            reason="no_contract_pending_finance_for_visible_decline_target",
            extra={"filter_contract_id": want},
        )

    def submit_regulator_approve(
        self,
        regulator: str,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
        *,
        contract_id: Any | None = None,
    ) -> None:
        _ = resources_snapshot
        if regulator != "regulator":
            self._audit_action(
                agent=regulator,
                negotiation_op="formal",
                verb="regulatory_approve",
                valid=False,
                reason="role_must_be_regulator",
            )
            return

        if self.regulator_regulatory_path_withdrawn:
            self._audit_action(
                agent=regulator,
                negotiation_op="formal",
                verb="regulatory_approve",
                valid=False,
                reason="regulator_withdrew_from_regulatory_path",
            )
            return

        want = str(contract_id).strip() if contract_id not in (None, "") else None

        for c in self.contracts.values():
            if want is not None and c.contract_id != want:
                continue
            if not self._contract_live(c):
                continue
            if regulator not in c.visibility:
                continue
            if int(c.regulatory.get("required", 0)) != 1:
                continue
            if c.regulatory.get("status") in ("approved", "blocked"):
                continue
            if int(c.terms.get("policy_hard_violation", 0) or 0) == 1:
                c.status = "failed"
                c.regulatory["status"] = "blocked"
                self._contract_append_history(
                    c,
                    "regulatory.blocked",
                    regulator,
                    detail={"contract_id": c.contract_id, "reason": "policy_hard_violation"},
                )
                self._audit_action(
                    agent=regulator,
                    negotiation_op="formal",
                    verb="regulatory_approve",
                    valid=False,
                    reason="policy_hard_violation_blocks_approval",
                    extra={"contract_id": c.contract_id},
                )
                self.mark_structural_progress()
                return
            if not self._formal_budget_allows_increment(regulator, verb="regulatory_approve"):
                return
            c.regulatory["status"] = "approved"
            self._contract_append_history(
                c, "regulatory.approved", regulator, detail={"contract_id": c.contract_id}
            )
            self._formal_budget_commit_increment(regulator)
            self._audit_action(
                agent=regulator,
                negotiation_op="formal",
                verb="regulatory_approve",
                valid=True,
                reason="approved",
                extra={"contract_id": c.contract_id},
            )
            self.record_execution_event(
                "regulatory_approved",
                f"监管方批准（contract_id={c.contract_id}）",
                contract_id=c.contract_id,
                agent=regulator,
            )
            self.mark_structural_progress()
            return

        self._audit_action(
            agent=regulator,
            negotiation_op="formal",
            verb="regulatory_approve",
            valid=False,
            reason="no_contract_pending_regulator_for_visible_target",
            extra={"filter_contract_id": want},
        )

    def submit_regulatory_block(
        self,
        regulator: str,
        resources_snapshot: Callable[[], dict[str, dict[str, float]]],
        payload: dict[str, Any],
    ) -> None:
        """§6.4.3 ``commit.block`` — 显式监管否决（与 hard violation 路径区分）。"""
        _ = resources_snapshot
        if regulator != "regulator":
            self._audit_action(
                agent=regulator,
                negotiation_op="formal",
                verb="regulatory_block",
                valid=False,
                reason="role_must_be_regulator",
            )
            return
        if self.regulator_regulatory_path_withdrawn:
            self._audit_action(
                agent=regulator,
                negotiation_op="formal",
                verb="regulatory_block",
                valid=False,
                reason="regulator_withdrew_from_regulatory_path",
            )
            return
        raw_cid = payload.get("contract_id")
        want = str(raw_cid).strip() if raw_cid not in (None, "") else None
        reason = str(payload.get("reason", ""))[:500]

        for c in self.contracts.values():
            if want is not None and c.contract_id != want:
                continue
            if not self._contract_live(c):
                continue
            if regulator not in c.visibility:
                continue
            if int(c.regulatory.get("required", 0)) != 1:
                continue
            if c.regulatory.get("status") == "approved":
                self._audit_action(
                    agent=regulator,
                    negotiation_op="formal",
                    verb="regulatory_block",
                    valid=False,
                    reason="cannot_block_after_approval",
                    extra={"contract_id": c.contract_id},
                )
                return
            if not self._formal_budget_allows_increment(regulator, verb="regulatory_block"):
                return
            c.status = "failed"
            c.regulatory["status"] = "blocked"
            self._contract_append_history(
                c,
                "regulatory.blocked",
                regulator,
                detail={"contract_id": c.contract_id, "reason": reason or "explicit_block"},
            )
            self._formal_budget_commit_increment(regulator)
            self._audit_action(
                agent=regulator,
                negotiation_op="formal",
                verb="regulatory_block",
                valid=True,
                reason="blocked",
                extra={"contract_id": c.contract_id},
            )
            self.mark_structural_progress()
            return

        self._audit_action(
            agent=regulator,
            negotiation_op="formal",
            verb="regulatory_block",
            valid=False,
            reason="no_contract_pending_regulatory_block",
            extra={"filter_contract_id": want},
        )

    def _terminate(self, reason: str) -> None:
        """§9 — world-level 终止（幂等），并记入 ``event_log``。"""
        if self.phase == Phase.TERMINATED:
            return
        self.record_execution_event(
            "world_terminate",
            f"谈判世界结束：terminal={reason!r}",
            terminal=reason,
        )
        self.phase = Phase.TERMINATED
        self.terminal = reason
        self.append_event_records(
            [
                {
                    "kind": "world_terminal_v9",
                    "terminal_state": reason,
                    "calendar_day": self.day,
                    "slot_index": self.slot,
                }
            ]
        )

    def terminated(self) -> bool:
        return self.phase == Phase.TERMINATED


def parse_agent_action_payload(action_argument: Any) -> dict[str, Any] | None:
    """兼容 ``argument`` 为 ``str``（JSON）或 ``dict``；若仅有旧键 ``setting1_op`` 则映射为 ``negotiation_op``。"""
    out: dict[str, Any] | None = None
    if isinstance(action_argument, dict):
        out = dict(action_argument)
    elif isinstance(action_argument, str) and action_argument.strip().startswith("{"):
        try:
            o = json.loads(action_argument)
            out = dict(o) if isinstance(o, dict) else None
        except json.JSONDecodeError:
            out = None
    if out is not None:
        legacy = out.get("setting1_op")
        if "negotiation_op" not in out and legacy is not None:
            out["negotiation_op"] = legacy
    return out
