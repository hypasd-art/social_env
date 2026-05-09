from .app import app
from .benchmark import negotiation_batch  # noqa: F401 — registers Typer negotiation-batch

try:
    from .benchmark import benchmark as _benchmark_cmd  # noqa: F401
except Exception:
    # negotiation-batch 在无 Redis OM / benchmark 栈时仍可用。
    pass

__all__ = ["app", "benchmark"]


def __getattr__(name: str):
    if name == "benchmark":
        from .benchmark import benchmark as benchmark_command

        return benchmark_command
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
