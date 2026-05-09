from .negotiation_batch import negotiation_batch  # noqa: F401 — registers Typer command

__all__ = ["benchmark"]


def __getattr__(name: str):
    if name == "benchmark":
        from .benchmark import benchmark as benchmark_command

        return benchmark_command
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
