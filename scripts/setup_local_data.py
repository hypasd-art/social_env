#!/usr/bin/env python
"""把数据填进本地 JSON 后端 (~/.sotopia/data/<Class>/<pk>.json)，让 sotopia 不依赖 Redis 也能跑。

数据来源（按优先级）：
  1) social_env/export/<Class>.json   (你之前手工导出的；JSON 数组)
  2) Redis 上正在运行的实例           (作为兜底；当 export/ 缺某张表时)

用法:
    # 1) 仅从 export/ 导入；缺什么就报什么
    SOTOPIA_STORAGE_BACKEND=local python scripts/setup_local_data.py

    # 2) 自动从当前 Redis 把缺的表拉回来（Redis 必须在跑）
    SOTOPIA_STORAGE_BACKEND=local python scripts/setup_local_data.py \\
        --fallback-redis --redis-url redis://localhost:6379

    # 3) 干跑，只看哪些表会被导入、各导多少条，不真的写盘
    SOTOPIA_STORAGE_BACKEND=local python scripts/setup_local_data.py --dry-run

跑 sotopia benchmark --task hard 必备的 4 张表：
    AgentProfile / EnvironmentProfile / EnvironmentList / EnvAgentComboStorage
（可选：RelationshipProfile —— 仅 ConstraintBasedSampler 用得到，hard 任务不需要）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

EXPORT_DIR = Path(__file__).resolve().parent.parent / "export"
LOCAL_DATA_DIR = Path(os.path.expanduser("~/.sotopia/data"))

# 文件名 -> 本地后端的目录名（即 model_class.__name__）
# 这里 Key 即两者都一样；为了清晰单独列
EXPORT_FILES: dict[str, str] = {
    "AgentProfile.json": "AgentProfile",
    "EnvironmentProfile.json": "EnvironmentProfile",
    "EnvironmentList.json": "EnvironmentList",
    "EnvAgentComboStorage.json": "EnvAgentComboStorage",
    "RelationshipProfile.json": "RelationshipProfile",
    # EpisodeLog 的旧导出通常为空，且 benchmark 会自己写新的，跳过
}

REQUIRED_FOR_BENCHMARK_HARD = {
    "AgentProfile",
    "EnvironmentProfile",
    "EnvironmentList",
    "EnvAgentComboStorage",
}


def _load_export_array(path: Path) -> list[dict[str, Any]]:
    """读取 export/ 下的 JSON 数组；空文件或非数组都返回 []。"""

    if not path.exists():
        return []
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [warn] 解析 {path.name} 失败：{e}")
        return []
    if not isinstance(data, list):
        print(f"  [warn] {path.name} 顶层不是 JSON 数组，跳过")
        return []
    return data


def _write_records_to_local(model_dir_name: str, records: Iterable[dict[str, Any]], *, dry_run: bool) -> int:
    """把记录按 <pk>.json 写到本地后端目录；返回写入条数。"""

    target_dir = LOCAL_DATA_DIR / model_dir_name
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped_no_pk = 0
    for rec in records:
        pk = rec.get("pk")
        if not pk:
            skipped_no_pk += 1
            continue
        if not dry_run:
            with (target_dir / f"{pk}.json").open("w") as f:
                json.dump(rec, f, indent=2, default=str, ensure_ascii=False)
        count += 1
    if skipped_no_pk:
        print(f"  [warn] {model_dir_name}: 跳过 {skipped_no_pk} 条无 pk 的记录")
    return count


def _dump_from_redis(model_dir_name: str, redis_url: str) -> list[dict[str, Any]]:
    """连 Redis，按类名扫所有 JSON key，返回原始 JSON 列表。"""

    try:
        import redis  # type: ignore
    except ImportError:
        print("  [error] redis-py 未安装，无法兜底从 Redis 拉数据")
        return []

    name_map = {
        "AgentProfile": "sotopia.database.persistent_profile.AgentProfile",
        "EnvironmentProfile": "sotopia.database.persistent_profile.EnvironmentProfile",
        "EnvironmentList": "sotopia.database.persistent_profile.EnvironmentList",
        "EnvAgentComboStorage": "sotopia.database.env_agent_combo_storage.EnvAgentComboStorage",
        "RelationshipProfile": "sotopia.database.persistent_profile.RelationshipProfile",
    }
    full = name_map.get(model_dir_name)
    if not full:
        print(f"  [error] 未知 model 名 {model_dir_name}")
        return []

    r = redis.from_url(redis_url, decode_responses=True)
    try:
        r.ping()
    except Exception as e:
        print(f"  [error] Redis 连接失败 ({redis_url}): {e}")
        return []

    pattern = f":{full}:*"
    records: list[dict[str, Any]] = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor=cursor, match=pattern, count=1000)
        for key in batch:
            if key.endswith(":index") or key.endswith(":schema"):
                continue
            try:
                raw = r.execute_command("JSON.GET", key, "$")
                if not raw:
                    continue
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    records.append(parsed[0])
                elif isinstance(parsed, dict):
                    records.append(parsed)
            except Exception as e:
                print(f"  [warn] 读取 {key} 失败: {e}")
        if cursor == 0:
            break
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fallback-redis",
        action="store_true",
        help="export/ 缺某张表时，自动从当前 Redis 拉数据补上",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_OM_URL", "redis://localhost:6379"),
        help="Redis 地址（仅当 --fallback-redis 时使用）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不真的写盘")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="导入前先清空 ~/.sotopia/data/<Class>/ 下的旧 JSON（避免脏数据）",
    )
    args = parser.parse_args()

    backend = os.getenv("SOTOPIA_STORAGE_BACKEND", "").lower()
    if backend != "local":
        print(
            "提示: 当前 SOTOPIA_STORAGE_BACKEND != 'local'。这个脚本只是把数据落到 "
            "~/.sotopia/data，不强制本地后端；但跑 benchmark 时记得：\n"
            "  export SOTOPIA_STORAGE_BACKEND=local"
        )

    print(f"export/ 目录: {EXPORT_DIR}")
    print(f"本地后端目录: {LOCAL_DATA_DIR}")
    print(f"dry-run: {args.dry_run}, fallback-redis: {args.fallback_redis}")
    print()

    summary: dict[str, dict[str, Any]] = {}
    for filename, model_dir_name in EXPORT_FILES.items():
        path = EXPORT_DIR / filename
        records = _load_export_array(path)
        source = "export"

        if not records and args.fallback_redis:
            print(f"[{model_dir_name}] export/{filename} 为空或不存在，尝试从 Redis 拉")
            records = _dump_from_redis(model_dir_name, args.redis_url)
            source = "redis" if records else "missing"

        if not records:
            print(f"[{model_dir_name}] 无数据 (export 与 Redis 均不可用)")
            summary[model_dir_name] = {"source": "missing", "count": 0}
            continue

        print(f"[{model_dir_name}] 来源={source}, 待导入 {len(records)} 条")

        target_dir = LOCAL_DATA_DIR / model_dir_name
        if args.clear_existing and target_dir.exists() and not args.dry_run:
            for old in target_dir.glob("*.json"):
                old.unlink()
            print(f"  已清空 {target_dir}")

        n = _write_records_to_local(model_dir_name, records, dry_run=args.dry_run)
        summary[model_dir_name] = {"source": source, "count": n}

    print("\n=== 汇总 ===")
    for name, info in summary.items():
        flag = "✓" if info["count"] > 0 else "✗"
        print(f"  {flag} {name:25s}  source={info['source']:7s}  count={info['count']}")

    missing = [m for m in REQUIRED_FOR_BENCHMARK_HARD if summary.get(m, {}).get("count", 0) == 0]
    print()
    if missing:
        print(f"!! 还缺 {missing}，sotopia benchmark --task hard 起不来。")
        print("   补救方式（任选其一）：")
        print("   a) 启动 Redis 后跑：  python scripts/setup_local_data.py --fallback-redis")
        print("   b) 用 hf 镜像下 dump.rdb：bash scripts/run.sh --reset-data --no-benchmark")
        print("      然后再跑 (a)；")
        print("   c) 单独把缺的表的 json 数组放到 social_env/export/<Class>.json，再重跑本脚本。")
        return 1

    print("✓ 跑 sotopia benchmark --task hard 所需数据齐全。")
    print("  现在可以：")
    print("    export SOTOPIA_STORAGE_BACKEND=local")
    print("    sotopia benchmark --models gpt-4o --partner-model gpt-4o-mini \\")
    print("        --evaluator-model gpt-4o-mini --task hard --batch-size 5 \\")
    print("        --tag local_run_trial0 --push-to-db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
