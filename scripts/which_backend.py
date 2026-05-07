#!/usr/bin/env python
"""诊断 sotopia 当前从哪里读数据。

输出三块信息：
1. 环境变量（你设了什么）
2. Sotopia 进程实际用的后端（is_redis_backend / is_local_backend）
3. 两边的数据量盘点
   - 本地 ~/.sotopia/data/<Class>/*.json 的文件数
   - Redis 里 <Class>:* 的 key 数（如果连得上）

用法
----
    /home/yphao/.conda/envs/social_env/bin/python scripts/which_backend.py

    # 强制切到 local 再看
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/which_backend.py

    # 也指定 Redis URL
    REDIS_OM_URL=redis://127.0.0.1:6379 \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/which_backend.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 让 .env 也能被识别（与 sotopia 进程行为一致）
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# 关注的核心模型类名（== 本地后端的目录名 == redis-om 的 key 前缀）
CORE_MODELS = [
    "AgentProfile",
    "EnvironmentProfile",
    "RelationshipProfile",
    "EnvAgentComboStorage",
    "EnvironmentList",
    "EpisodeLog",
    # V2
    "AgentProfileV2",
    "EnvironmentProfileV2",
    "EventScript",
    "Contract",
    "SystemStateSnapshot",
    "EpisodeLogV2",
]


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1) 环境变量
# ---------------------------------------------------------------------------
section("1. 环境变量")
for key in [
    "SOTOPIA_STORAGE_BACKEND",
    "REDIS_OM_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
]:
    val = os.environ.get(key)
    shown = "(unset)" if val is None else (val if "API_KEY" not in key else f"{val[:8]}...")
    default_hint = ""
    if key == "SOTOPIA_STORAGE_BACKEND" and val is None:
        default_hint = "  ← default = 'redis'"
    if key == "REDIS_OM_URL" and val is None:
        default_hint = "  ← default = 'redis://localhost:6379'"
    print(f"  {key:<26} {shown}{default_hint}")


# ---------------------------------------------------------------------------
# 2) Sotopia 进程实际用的后端
# ---------------------------------------------------------------------------
section("2. Sotopia 进程实际后端")
try:
    from sotopia.database.storage_backend import (  # noqa: E402
        get_storage_backend,
        is_local_backend,
        is_redis_backend,
    )

    backend = get_storage_backend()
    print(f"  type(backend)         {type(backend).__name__}")
    print(f"  is_redis_backend()    {is_redis_backend()}")
    print(f"  is_local_backend()    {is_local_backend()}")
    if hasattr(backend, "base_path"):
        print(f"  local data dir        {backend.base_path}")
except Exception as e:
    print(f"  [err] 无法 import storage_backend: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 3) 本地数据盘点
# ---------------------------------------------------------------------------
section("3. 本地后端 ~/.sotopia/data/ 文件数")
local_dir = Path(os.path.expanduser("~/.sotopia/data"))
if not local_dir.exists():
    print(f"  {local_dir} 不存在")
else:
    total = 0
    for cls in CORE_MODELS:
        d = local_dir / cls
        n = len(list(d.glob("*.json"))) if d.exists() else 0
        total += n
        marker = "  " if n == 0 else "✓ "
        print(f"  {marker}{cls:<26} {n}")
    extra_dirs = [d for d in local_dir.iterdir() if d.is_dir() and d.name not in CORE_MODELS]
    if extra_dirs:
        print(f"  其它目录: {[d.name for d in extra_dirs]}")
    print(f"  ---- 合计 {total} 条 ----")


# ---------------------------------------------------------------------------
# 4) Redis 数据盘点
# ---------------------------------------------------------------------------
section("4. Redis 后端 key 数（若连得上）")
redis_url = os.environ.get("REDIS_OM_URL", "redis://localhost:6379")
try:
    import redis  # type: ignore

    r = redis.Redis.from_url(redis_url, socket_connect_timeout=2)
    r.ping()
    print(f"  连接成功: {redis_url}")
    total = 0
    for cls in CORE_MODELS:
        # redis-om JsonModel 的 key 模式: <module_path>.<ClassName>:<pk>
        # 例: sotopia.database.persistent_profile.AgentProfile:01H5TNE5...
        # 这里用 *<ClassName>:* 兜底
        keys = list(r.scan_iter(match=f"*{cls}:*", count=1000))
        n = len(keys)
        total += n
        marker = "  " if n == 0 else "✓ "
        print(f"  {marker}{cls:<26} {n}")
    print(f"  ---- 合计 {total} 条 ----")
    print(f"  RediSearch indexes: {r.execute_command('FT._LIST')!r}")
except ImportError:
    print("  [skip] redis 包未装")
except Exception as e:
    print(f"  [skip] 连不上 {redis_url}: {e}")


# ---------------------------------------------------------------------------
# 5) 用 sotopia 自己的 API 实测一遍
# ---------------------------------------------------------------------------
section("5. 用 sotopia API 实测 (AgentProfile.all_pks())")
try:
    from sotopia.database import AgentProfile  # noqa: E402

    pks = list(AgentProfile.all_pks())
    print(f"  AgentProfile.all_pks() 返回 {len(pks)} 条")
    if pks:
        print(f"  示例 pk: {pks[0]}")
        a = AgentProfile.get(pks[0])
        print(f"  示例数据: {a.first_name} {a.last_name}, {a.occupation}")
except Exception as e:
    print(f"  [err] {e}")
