"""轻量情景记忆（Phase 0–1）：滑动窗口文本，不做向量库。

``arecent`` / ``recent`` 在拼接最近 ``k`` 条后若超过 ``max_recent_chars``，则经
``truncate_chars`` 做 **确定性压缩**（保留头尾），避免无界增长；与
``SummarizingEpisodicMemory`` 的窗口上限语义对齐，但不调用 LLM。

后续可替换为同一接口背后的向量检索实现，而不改 ``SocialLLMAgent``。"""

from __future__ import annotations

from collections import deque

from sotopia.agents.memory_summarization import truncate_chars


class EpisodicMemory:
    def __init__(self, max_entries: int = 40, *, max_recent_chars: int | None = 12_000) -> None:
        self._buf: deque[str] = deque(maxlen=max_entries)
        self._max_recent_chars = None if max_recent_chars is None else max(256, int(max_recent_chars))

    def add(self, line: str) -> None:
        if line:
            self._buf.append(line)

    def _maybe_compress(self, text: str) -> str:
        if self._max_recent_chars is None or len(text) <= self._max_recent_chars:
            return text
        return truncate_chars(text, self._max_recent_chars)

    def recent(self, k: int = 8) -> str:
        if k <= 0:
            return ""
        take = list(self._buf)[-k:]
        return self._maybe_compress("\n".join(take))

    async def arecent(self, k: int = 8) -> str:
        """异步读取最近记忆；基类与同步 ``recent`` 等价，子类可覆盖为 LLM 压缩路径。"""
        return self.recent(k)

    def clear(self) -> None:
        self._buf.clear()


__all__ = ["EpisodicMemory"]
