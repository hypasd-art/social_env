"""长周期运行时状态导出。

``ContractLedger`` 依赖 benchmark v2 JsonModel，延迟加载以免影响仅需 ``SystemState`` 的轻量化脚本。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .system_state import SystemState, state_from_profile_init

if TYPE_CHECKING:
    from .contracts import ContractLedger

__all__ = ["ContractLedger", "SystemState", "state_from_profile_init"]


def __getattr__(name: str):
    if name == "ContractLedger":
        from .contracts import ContractLedger as CL

        return CL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
