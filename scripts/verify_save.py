#!/usr/bin/env python
"""绕过 server.py 的 try/except，直接构造一个最小 EpisodeLog 并 .save()。

如果 server.py 在 push 阶段静默失败，这个脚本会把真正的异常打出来。

用法:
    SOTOPIA_STORAGE_BACKEND=redis REDIS_OM_URL=redis://localhost:6379 \\
      python scripts/verify_save.py --tag _diagnose_save_test
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="_diagnose_save_test")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="保存成功后立即删除该测试 episode，避免污染数据",
    )
    args = parser.parse_args()

    print(f"REDIS_OM_URL = {os.getenv('REDIS_OM_URL', 'redis://localhost:6379')}")
    print(f"SOTOPIA_STORAGE_BACKEND = {os.getenv('SOTOPIA_STORAGE_BACKEND', 'redis')}")
    print(f"tag = {args.tag}")

    try:
        from sotopia.database.logs import EpisodeLog
    except Exception:
        print("import EpisodeLog 失败：")
        traceback.print_exc()
        return 1

    epilog = EpisodeLog(
        environment="env_test_save",
        agents=["agent_a", "agent_b"],
        tag=args.tag,
        models=["gpt-4o-mini", "gpt-4o", "gpt-4o-mini"],
        messages=[[("Environment", "agent_a", "hi")]],
        reasoning="test",
        rewards=[0.0, 0.0],
    )
    print(f"构造成功，pk(初始) = {epilog.pk!r}")

    try:
        epilog.save()
    except Exception:
        print("\n==== epilog.save() 抛出真实异常 ====")
        traceback.print_exc()
        return 2

    print(f"\n保存成功，pk(写库后) = {epilog.pk!r}")

    try:
        roundtrip = EpisodeLog.find(EpisodeLog.tag == args.tag).all()  # type: ignore
        print(f"find(tag == {args.tag!r}).all() 返回 {len(roundtrip)} 条")
    except Exception:
        print("find 失败：")
        traceback.print_exc()
        return 3

    if args.cleanup and epilog.pk:
        try:
            EpisodeLog.delete(epilog.pk)
            print(f"已清理测试记录 pk={epilog.pk}")
        except Exception:
            print("cleanup 失败（可忽略）：")
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
