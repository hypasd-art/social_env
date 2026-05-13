import re


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


def format_docstring(docstring: str) -> str:
    """Format a docstring for use in a prompt template."""
    return re.sub("\n +", "\n", docstring).strip()
