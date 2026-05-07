#!/usr/bin/env python
"""诊断 sotopia benchmark 跑完后“查不到 episode”的根因。

用法:
    SOTOPIA_STORAGE_BACKEND=redis REDIS_OM_URL=redis://localhost:6379 \\
      python scripts/diagnose_episodes.py \\
        --tag benchmark_gpt-4o_gpt-4o-mini_gpt-4o-mini_hard_trial0

会依次打印:
  [1] Redis 连接状态
  [2] EpisodeLog raw key 数量（绕过 RediSearch，直接 SCAN）
  [3] 这些 key 中匹配指定 tag 的数量（直接读 JSON 字段）
  [4] EpisodeLog.find(EpisodeLog.tag == tag).all() 的返回数量（走 RediSearch）
  [5] 现有 RediSearch 索引列表，以及 EpisodeLog 索引的字段定义
  [6] 给出明确的下一步动作建议
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import redis


def _print_section(idx: int, title: str) -> None:
    print(f"\n[{idx}] {title}")
    print("-" * (len(title) + 4))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        required=True,
        help="benchmark 写库时使用的 tag，例如 benchmark_gpt-4o_gpt-4o-mini_gpt-4o-mini_hard_trial0",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="发现 raw 多 / find 少 的不一致时，调用 Migrator().run() 重建索引",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "强制重建 EpisodeLog 索引：先 DEL schema 哈希 + FT.DROPINDEX（若存在），"
            "再调 Migrator().run()，并把 Migrator 的真实异常打出来。"
            "适用于 'No such index' 报错"
        ),
    )
    args = parser.parse_args()

    redis_url = os.getenv("REDIS_OM_URL", "redis://localhost:6379")
    backend = os.getenv("SOTOPIA_STORAGE_BACKEND", "redis").lower()
    print(f"REDIS_OM_URL = {redis_url}")
    print(f"SOTOPIA_STORAGE_BACKEND = {backend}")
    if backend != "redis":
        print(
            "本脚本仅诊断 Redis 后端。请先 `export SOTOPIA_STORAGE_BACKEND=redis` 再跑。"
        )
        return 2

    # [1] 连接
    _print_section(1, "Redis 连接")
    try:
        r = redis.from_url(redis_url, decode_responses=True)
        r.ping()
        print("PING -> PONG ✓")
    except Exception as e:
        print(f"无法连接 Redis: {e}")
        print("请先把 redis-stack-server 跑起来（参见 scripts/run.sh）。")
        return 1

    # [2] raw key 数量
    _print_section(2, "EpisodeLog raw JSON key 数量（绕过索引）")
    pattern = ":sotopia.database.logs.EpisodeLog:*"
    raw_keys: list[str] = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor=cursor, match=pattern, count=1000)
        raw_keys.extend(batch)
        if cursor == 0:
            break
    # 过滤掉索引元 key
    record_keys = [k for k in raw_keys if not k.endswith(":index")]
    print(f"匹配 {pattern} 的 key 数量: {len(record_keys)}")
    if record_keys[:3]:
        print(f"前 3 个示例 key: {record_keys[:3]}")

    # [3] 在 raw JSON 上按 tag 数
    _print_section(3, f"raw JSON 中 tag == {args.tag!r} 的数量")
    matched = 0
    sample_pks: list[str] = []
    for k in record_keys:
        try:
            doc = r.execute_command("JSON.GET", k, "$.tag")
            if not doc:
                continue
            data = json.loads(doc)
            tags = data if isinstance(data, list) else [data]
            if args.tag in tags:
                matched += 1
                if len(sample_pks) < 3:
                    pk = k.rsplit(":", 1)[-1]
                    sample_pks.append(pk)
        except Exception:
            continue
    print(f"raw 匹配 tag 的数量: {matched}")
    if sample_pks:
        print(f"前 3 个 pk: {sample_pks}")

    # [4] redis-om find()
    _print_section(4, "EpisodeLog.find(EpisodeLog.tag == tag).all() 返回数量")
    find_error: str = ""
    try:
        from sotopia.database.logs import EpisodeLog

        find_results: list[Any] = EpisodeLog.find(EpisodeLog.tag == args.tag).all()  # type: ignore
        print(f"find().all() 返回数量: {len(find_results)}")
    except Exception as e:
        find_error = str(e)
        print(f"find() 抛异常: {e}")
        find_results = []

    # [5] 索引信息
    _print_section(5, "RediSearch 索引现状")
    ep_idx_name = ":sotopia.database.logs.EpisodeLog:index"
    ep_schema_key = ":sotopia.database.logs.EpisodeLog:schema"
    index_present = False
    try:
        all_indices_raw = r.execute_command("FT._LIST") or []
        all_indices: list[str] = [
            (x.decode() if isinstance(x, (bytes, bytearray)) else x)
            for x in all_indices_raw
        ]
        print(f"全部索引: {all_indices}")
        index_present = ep_idx_name in all_indices
        if index_present:
            info = r.execute_command("FT.INFO", ep_idx_name)
            info_dict: dict[str, Any] = {}
            it = iter(info)
            for k in it:
                v = next(it)
                info_dict[k.decode() if isinstance(k, bytes) else k] = v
            print(f"num_docs: {info_dict.get('num_docs')}")
            attrs = info_dict.get("attributes") or info_dict.get("fields")
            if attrs:
                print("索引字段:")
                for a in attrs:
                    print(f"  - {a}")
        else:
            print(f"未找到索引 {ep_idx_name}")
        # schema 哈希 key 是 redis-om 用来判断 schema 是否变化的
        if r.exists(ep_schema_key):
            print(f"schema 哈希 key 存在: {ep_schema_key}")
        else:
            print(f"schema 哈希 key 不存在: {ep_schema_key}")
    except Exception as e:
        print(f"读取索引信息失败: {e}")

    # [5.5] 强制重建索引（在做决策前，避免 [6] 误判）
    if args.force_rebuild:
        _print_section(5, "force-rebuild: 清 schema 哈希 + DROPINDEX + Migrator")
        try:
            r.delete(ep_schema_key)
            print(f"已 DEL {ep_schema_key}")
        except Exception as e:
            print(f"DEL schema key 失败: {e}")
        try:
            r.execute_command("FT.DROPINDEX", ep_idx_name)
            print(f"已 DROPINDEX {ep_idx_name}")
        except Exception as e:
            print(f"DROPINDEX 跳过: {e}")
        try:
            from redis_om import Migrator

            Migrator().run()
            print("Migrator().run() 完成 ✓")
        except Exception as e:
            import traceback

            print("==== Migrator().run() 抛真实异常 ====")
            traceback.print_exc()
            print(f"summary: {e}")
        # 重新校验
        try:
            all_indices_raw = r.execute_command("FT._LIST") or []
            now_indices = [
                (x.decode() if isinstance(x, (bytes, bytearray)) else x)
                for x in all_indices_raw
            ]
            if ep_idx_name in now_indices:
                print(f"重建后已存在索引 {ep_idx_name} ✓")
                from sotopia.database.logs import EpisodeLog as _EL

                n = len(_EL.find(_EL.tag == args.tag).all())  # type: ignore
                print(f"现在 find(tag == {args.tag!r}) 返回数量: {n}")
            else:
                print(f"重建后仍未出现索引 {ep_idx_name} ✗")
        except Exception as e:
            print(f"重建后校验失败: {e}")
        return 0

    # [6] 决策
    _print_section(6, "结论与建议")
    if "No such index" in find_error or not index_present:
        print(f"→ EpisodeLog 索引缺失（'{find_error or '索引未列出'}'）。")
        print("  最可能原因:")
        print("    a) 启动时 sotopia/database/__init__.py 里 Migrator().run() 静默失败；")
        print("    b) 老 dump.rdb 只装回了数据，没装回索引，且 schema 哈希让 Migrator 误以为不需要重建。")
        if matched > 0:
            print(f"  注意: raw JSON 里有 {matched} 条匹配 tag 的记录，*数据没丢*，只是查不到。")
        print("  下一步: 重跑本脚本时加 --force-rebuild，会清 schema 哈希、DROPINDEX、再 Migrator。")
        return 0
    if matched == 0 and len(record_keys) == 0:
        print("→ Redis 里完全没有 EpisodeLog 记录。")
        print("  最可能原因: epilog.save() 抛了异常被 server.py 的 try/except 吞掉。")
        print("  下一步: 跑 scripts/verify_save.py 看真正的 save 异常。")
        return 0
    if matched > 0 and len(find_results) == 0:
        print("→ 数据写进去了，但 tag 索引没命中。")
        print("  下一步: 加 --rebuild-index 让脚本调用 Migrator().run()。仍不生效则改用 --force-rebuild。")
        if args.rebuild_index:
            try:
                from redis_om import Migrator

                Migrator().run()
                print("已调用 Migrator().run() 重建索引，重跑本脚本验证。")
            except Exception as e:
                print(f"Migrator 调用失败: {e}")
        return 0
    if matched > 0 and len(find_results) >= matched:
        print("→ 一切正常，数据已入库且 find() 能查到。")
        print("  benchmark_display 报 'No episodes found' 一般是 tag 字符串拼写不一致。")
        print("  请仔细对比写库 tag 与 display tag（包括末尾 trial0/trial1）。")
        return 0
    print("→ 出现意外组合，请把上面 [2]/[3]/[4]/[5] 的输出发给开发同学。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
