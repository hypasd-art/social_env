"""带「超长则总结」的情景记忆：逻辑在 ``memory_summarization``，本文件只负责缓冲与合并。"""

from __future__ import annotations

import logging

from sotopia.agents.memory import EpisodicMemory
from sotopia.agents.memory_summarization import (
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_USER_TEMPLATE,
    summarize_conversation_text,
    truncate_chars,
)

log = logging.getLogger(__name__)


class SummarizingEpisodicMemory(EpisodicMemory):
    """滑动窗口 + 当最近 ``k`` 条合并文本超过字符阈值时用 LLM 压缩更早条目。

    行为对齐 AgentEvolver ``SummarizedMemory``：在读取路径（``arecent``）上触发压缩，
    将较早行合并为一条 ``[Episode memory summary]`` 前缀的摘要，并保留尾部若干条原文。
    若未配置 ``summary_model`` 或 LLM 失败，则对过长块做 ``truncate_chars``。
    """

    def __init__(
        self,
        max_entries: int = 40,
        *,
        max_recent_chars: int = 12_000,
        preserve_tail_lines: int = 6,
        summary_model: str | None = None,
        max_single_line_chars: int = 16_000,
        summary_temperature: float = 0.3,
        summary_max_tokens: int = 1024,
        system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT,
        user_template: str = DEFAULT_SUMMARY_USER_TEMPLATE,
    ) -> None:
        super().__init__(max_entries=max_entries)
        self._max_recent_chars = max(256, int(max_recent_chars))
        self._preserve_tail_lines = max(1, int(preserve_tail_lines))
        self._summary_model = summary_model
        self._max_single_line_chars = max(512, int(max_single_line_chars))
        self._summary_temperature = float(summary_temperature)
        self._summary_max_tokens = int(summary_max_tokens)
        self._system_prompt = system_prompt
        self._user_template = user_template

    def add(self, line: str) -> None:
        if not line:
            return
        if len(line) > self._max_single_line_chars:
            line = truncate_chars(line, self._max_single_line_chars)
        super().add(line)

    async def arecent(self, k: int = 8) -> str:
        items = list(self._buf)
        if not items:
            return ""
        if k <= 0:
            return ""

        take = items[-k:]
        block = "\n".join(take)
        if len(block) <= self._max_recent_chars:
            return block

        preserve = min(self._preserve_tail_lines, len(items))
        head, tail = items[:-preserve], items[-preserve:]
        if not head:
            return truncate_chars(block, self._max_recent_chars)

        head_text = "\n".join(head)
        summary_text = ""
        if self._summary_model:
            summary_text = await summarize_conversation_text(
                head_text,
                model_name=self._summary_model,
                system_prompt=self._system_prompt,
                user_template=self._user_template,
                temperature=self._summary_temperature,
                max_tokens=self._summary_max_tokens,
            )
        if not summary_text:
            summary_text = truncate_chars(head_text, min(self._max_recent_chars, 8000))

        merged = [f"[Episode memory summary]\n{summary_text}", *tail]
        self._buf.clear()
        for ln in merged:
            if ln:
                self._buf.append(ln)

        out_take = list(self._buf)[-k:]
        out = "\n".join(out_take)
        if len(out) > self._max_recent_chars:
            out = truncate_chars(out, self._max_recent_chars)
        log.info(
            "Episodic memory compressed: %s lines -> %s lines (summary_model=%s)",
            len(items),
            len(self._buf),
            self._summary_model or "-",
        )
        return out


__all__ = ["SummarizingEpisodicMemory"]
