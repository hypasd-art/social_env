#!/usr/bin/env bash
set -euo pipefail

# 先测网络（哪个返回 200/302 就用哪个）
# curl -I --max-time 8 https://huggingface.co/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb
# curl -I --max-time 8 https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb

# 下载（关键：先停容器，避免 redis 自己 autosave 把它覆盖）
set -a; source .env; set +a
docker stop sotopia-redis >/dev/null 2>&1 || true

echo "Creating directory..."
mkdir -p ~/.sotopia/redis-data && cd ~/.sotopia/redis-data
rm -f dump.rdb

echo "Downloading dump.rdb..."
wget --content-disposition -O dump.rdb \
  "https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true"

echo "Downloaded dump.rdb"
ls -lh dump.rdb
file dump.rdb
head -c 5 dump.rdb && echo

docker rm -f sotopia-redis >/dev/null 2>&1 || true
docker run -d --name sotopia-redis \
  -p 6379:6379 \
  -v ~/.sotopia/redis-data:/data \
  redis/redis-stack-server:latest

# 等待 Redis 完成加载
for _ in $(seq 1 30); do
  if docker exec sotopia-redis redis-cli ping >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker exec sotopia-redis redis-cli ping
docker logs sotopia-redis 2>&1 | tail -25

sotopia benchmark \
  --models gpt-4o \
  --partner-model gpt-4o-mini \
  --evaluator-model gpt-4o-mini \
  --task hard --batch-size 10 --push-to-db