"""轻量情景记忆（Phase 0–1）：滑动窗口文本，不做向量库。

后续可替换为同一接口背后的向量检索实现，而不改 ``SocialLLMAgent``。"""

from __future__ import annotations

from collections import deque


class EpisodicMemory:
    def __init__(self, max_entries: int = 40) -> None:
        self._buf: deque[str] = deque(maxlen=max_entries)

    def add(self, line: str) -> None:
        if line:
            self._buf.append(line)

    def recent(self, k: int = 8) -> str:
        if k <= 0:
            return ""
        take = list(self._buf)[-k:]
        return "\n".join(take)

    def clear(self) -> None:
        self._buf.clear()


__all__ = ["EpisodicMemory"]
