"""文本记忆压缩：超长时截断或调用 LLM 总结（对齐 AgentEvolver ``SummarizedMemory`` 思路）。

与 ``agentscope`` / ``games.agents.memory`` 解耦，仅依赖 ``litellm.acompletion``，
供 ``memory_episodic_summarizing`` 等模块复用。
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "你是一个专业的对话总结助手，能够准确提取对话中的关键信息。"
    "只输出纯文本总结，不要输出 JSON、markdown 代码围栏或格式化指令。"
)

DEFAULT_SUMMARY_USER_TEMPLATE = (
    "请总结以下对话/事件记录，保留关键信息与重要细节。\n\n"
    "{conversation_history}\n\n"
    "请提供简短总结："
)


def truncate_chars(text: str, max_chars: int, *, head_ratio: float = 0.6) -> str:
    """按字符数截断，保留开头与一小段尾部（无 LLM 时的回退）。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_n = max(1, int(max_chars * head_ratio) - 32)
    tail_n = max(0, max_chars - head_n - 32)
    head = text[:head_n]
    tail = text[-tail_n:] if tail_n else ""
    sep = "\n… [truncated] …\n"
    return head + sep + tail


async def summarize_conversation_text(
    conversation_text: str,
    *,
    model_name: str,
    system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT,
    user_template: str = DEFAULT_SUMMARY_USER_TEMPLATE,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """用 LiteLLM 对一段纯文本做总结；失败时返回空串（由调用方再截断）。"""
    text = (conversation_text or "").strip()
    if not text:
        return ""

    from litellm import acompletion

    user_content = user_template.format(conversation_history=text)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        response = await acompletion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        log.warning("memory summarization LLM call failed: %s", exc)
        return ""

    content = response.choices[0].message.content
    if content is None:
        return ""
    out = str(content).strip()
    return out


__all__ = [
    "DEFAULT_SUMMARY_SYSTEM_PROMPT",
    "DEFAULT_SUMMARY_USER_TEMPLATE",
    "summarize_conversation_text",
    "truncate_chars",
]
