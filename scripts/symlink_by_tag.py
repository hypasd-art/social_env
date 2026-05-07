#!/usr/bin/env python
"""为 EpisodeLog 按 tag 生成可读的软链接，原始文件不动。

不破坏 sotopia 的 pk → 文件名 约定，只在
``~/.sotopia/data_by_tag/<tag>/<pk>.json`` 下建软链接，方便用
``ls`` / ``find`` / 文件管理器快速浏览同一组实验。

用法
----
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/symlink_by_tag.py

    # 只处理某个 tag
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/symlink_by_tag.py \\
        --tag run_v3

    # 清空旧软链接再重建
    SOTOPIA_STORAGE_BACKEND=local \\
        /home/yphao/.conda/envs/social_env/bin/python scripts/symlink_by_tag.py --clean
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")

LOCAL_DATA = Path(os.path.expanduser("~/.sotopia/data"))
EPISODE_DIR = LOCAL_DATA / "EpisodeLog"
SYMLINK_BASE = Path(os.path.expanduser("~/.sotopia/data_by_tag"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="只处理指定 tag")
    parser.add_argument("--clean", action="store_true", help="先清空 ~/.sotopia/data_by_tag/")
    args = parser.parse_args()

    if not EPISODE_DIR.exists():
        print(f"[err] {EPISODE_DIR} 不存在；本地还没有 EpisodeLog")
        return 1

    if args.clean and SYMLINK_BASE.exists():
        print(f"[clean] 删除 {SYMLINK_BASE}")
        shutil.rmtree(SYMLINK_BASE)

    SYMLINK_BASE.mkdir(parents=True, exist_ok=True)

    from sotopia.database import EpisodeLog

    pks = list(EpisodeLog.all_pks())
    print(f"[scan] {len(pks)} 条 EpisodeLog")

    by_tag: dict[str, list[str]] = {}
    for pk in pks:
        ep = EpisodeLog.get(pk)
        tag = (ep.tag or "_no_tag").strip() or "_no_tag"
        if args.tag and tag != args.tag:
            continue
        by_tag.setdefault(tag, []).append(pk)

    total = 0
    for tag, pks_in_tag in by_tag.items():
        tag_dir = SYMLINK_BASE / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        for pk in pks_in_tag:
            src = EPISODE_DIR / f"{pk}.json"
            dst = tag_dir / f"{pk}.json"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            total += 1
        print(f"  {tag:<30} {len(pks_in_tag)} files -> {tag_dir}")
    print(f"[done] 共建 {total} 条软链接")
    print(f"\n查看：ls {SYMLINK_BASE}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
