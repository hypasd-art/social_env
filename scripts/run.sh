#!/usr/bin/env bash
# 用法：
#   bash scripts/run.sh                # 幂等启动 Redis（保留现有数据）+ 跑 benchmark
#   bash scripts/run.sh --no-benchmark # 只拉起 Redis，不跑 benchmark（用于先 display）
#   bash scripts/run.sh --display-only # 只展示已跑的 benchmark 结果，不写库
#   bash scripts/run.sh --reset-data   # 危险：清空 ~/.sotopia/redis-data 并重新下 dump.rdb
#
# 默认幂等行为：
#   - 已有容器 sotopia-redis 在跑 → 跳过重启
#   - 已有 dump.rdb → 不重新下载（避免覆盖你跑过的 episode）
#   - 停止容器前会 BGSAVE 落盘
set -euo pipefail

# -------- 参数解析 --------
RESET_DATA=0
NO_BENCH=0
DISPLAY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --reset-data)   RESET_DATA=1 ;;
    --no-benchmark) NO_BENCH=1 ;;
    --display-only) DISPLAY_ONLY=1 ;;
    -h|--help)
      sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

# -------- 环境变量 --------
if [ -f .env ]; then
  set -a; source .env; set +a
fi

DATA_DIR="${HOME}/.sotopia/redis-data"
mkdir -p "${DATA_DIR}"

# -------- 数据：只在 --reset-data 时才重下 --------
if [ "${RESET_DATA}" = "1" ]; then
  echo "[reset-data] 停止容器并删除现有数据"
  docker stop sotopia-redis >/dev/null 2>&1 || true
  docker rm -f sotopia-redis >/dev/null 2>&1 || true
  rm -f "${DATA_DIR}/dump.rdb" "${DATA_DIR}/appendonlydir" 2>/dev/null || true

  echo "[reset-data] 重新下载 dump.rdb 到 ${DATA_DIR}"
  ( cd "${DATA_DIR}" && wget --content-disposition -O dump.rdb \
      "https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true" )
  ls -lh "${DATA_DIR}/dump.rdb"
  file "${DATA_DIR}/dump.rdb"
elif [ ! -f "${DATA_DIR}/dump.rdb" ]; then
  echo "[bootstrap] ${DATA_DIR}/dump.rdb 不存在，首次拉起：下载初始 dump.rdb"
  ( cd "${DATA_DIR}" && wget --content-disposition -O dump.rdb \
      "https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true" )
fi

# -------- 容器：幂等启动 --------
if docker ps --format '{{.Names}}' | grep -qx 'sotopia-redis'; then
  echo "[redis] 容器 sotopia-redis 已在运行，复用现有实例（保留你跑过的 episode）"
elif docker ps -a --format '{{.Names}}' | grep -qx 'sotopia-redis'; then
  echo "[redis] 容器存在但未运行，启动它"
  docker start sotopia-redis >/dev/null
else
  echo "[redis] 容器不存在，创建并启动"
  docker run -d --name sotopia-redis \
    -p 6379:6379 \
    -v "${DATA_DIR}:/data" \
    redis/redis-stack-server:latest >/dev/null
fi

# -------- 等待 ready --------
for _ in $(seq 1 30); do
  if docker exec sotopia-redis redis-cli ping >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
echo -n "[redis] PING -> "; docker exec sotopia-redis redis-cli ping

# -------- 运行/展示 benchmark --------
if [ "${NO_BENCH}" = "1" ]; then
  echo "[done] --no-benchmark 已设置，跳过 benchmark。"
  exit 0
fi

BENCH_BASE=( sotopia benchmark
  --models gpt-4o
  --partner-model gpt-4o-mini
  --evaluator-model gpt-4o-mini
  --task hard
  --batch-size 50
  --tag benchmark_gpt-4o_gpt-4o-mini_gpt-4o-mini_hard_trial0 )

if [ "${DISPLAY_ONLY}" = "1" ]; then
  echo "[bench] --display-only：只展示，不写库"
  "${BENCH_BASE[@]}" --only-show-performance
else
  echo "[bench] 跑 benchmark 并 push 到 Redis"
  "${BENCH_BASE[@]}" --push-to-db
fi


# `run.sh` 已经改成幂等启动，下次不会再因为重跑脚本把数据清空。

# ## 你这次诊断结果的最终解读

# 把日志里的关键 4 行拎出来：

# ```text
# [2] EpisodeLog raw key 数量: 0     ← Redis 里没有任何 EpisodeLog json key
# [3] raw 匹配 tag 的数量: 0          ← 同上，再确认一次
# [5] num_docs: 0                    ← 索引存在，文档数为 0
# 重建后已存在索引 ... ✓ ; 现在 find ... 返回数量: 0
# ```

# 含义：**之前 100/100 跑完那批数据已经不在 Redis 里了**。索引现在是好的，但下面没有文档可查。

# 最可能就是：你那次 `bash scripts/run.sh` 在跑 benchmark 之前 `rm -f dump.rdb` + 重下了 245MB 文件覆盖回去，相当于把所有写入清零。这正是新 `run.sh` 默认行为修掉的事。

# `sotopia benchmark-display` 这条命令本来就不存在（CLI 只注册了 `benchmark` / `install`），所以那条 `No such command` 只是顺带的副伤害，不是核心问题。

# ## 接下来的标准动作（按顺序做就行）

# ### 1) 拉起 Redis（不会动你已有的 dump 或 episode）

# ```bash
# cd /mnt/userdata/yphao/FC/game_MAS/social_env
# bash scripts/run.sh --no-benchmark
# ```

# ### 2) 先跑一次最小写库验证，确认 save 链路真的通了

# ```bash
# SOTOPIA_STORAGE_BACKEND=redis REDIS_OM_URL=redis://localhost:6379 \
#   python scripts/verify_save.py --tag _diag_save_test --cleanup
# ```

# 期待看到：
# ```text
# 保存成功，pk(写库后) = '<26 位 ULID>'
# find(tag == '_diag_save_test').all() 返回 1 条
# 已清理测试记录 pk=...
# ```

# 如果这一步就 traceback 了，那才是 `epilog.save()` 真出问题，你把那段 traceback 贴上来我精修。

# ### 3) 跑完整 benchmark（这次会真正落库）

# ```bash
# bash scripts/run.sh
# ```

# 跟以前不同的是：
# - 不会再下载/覆盖 `dump.rdb`；
# - 会自动复用现有容器（如果在跑），数据不丢；
# - 跑完后所有 100 个 episode 都会留在 `~/.sotopia/redis-data/` 对应的 Redis 里。

# 跑完之后，可以确认一下：

# ```bash
# SOTOPIA_STORAGE_BACKEND=redis REDIS_OM_URL=redis://localhost:6379 \
#   python scripts/diagnose_episodes.py \
#   --tag benchmark_gpt-4o_gpt-4o-mini_gpt-4o-mini_hard_trial0
# ```

# 这次 `[2]/[3]/[5] num_docs` 应该都是 100 上下。

# ### 4) 只想再看分数（不重新跑 LLM）

# ```bash
# bash scripts/run.sh --display-only
# ```

# 这个会内部调 `sotopia benchmark --only-show-performance --tag <同一个 tag>`，把 7 个维度（BEL/REL/KNO/SEC/SOC/FIN/GOAL）+ episode 计数打成 rich 表格。

# ## 顺手提醒一下后续不要踩的坑

# - 下次升级 `dump.rdb` 或想从头重置：**显式**用 `bash scripts/run.sh --reset-data`。这条命令保留以前那种"清空+重下"的语义，方便你换数据集。
# - 之前担心过的"老 EpisodeLog schema 与代码不兼容"那些报错，目前没复现的迹象（`num_docs=0` 跟 schema 无关）。等你按上面 1-3 跑一遍确认稳定之后再说要不要清理 `dump.rdb` 里的旧 EpisodeLog 记录。