# Sotopia 运行手册（中文）

> 本文件是 [`PROJECT_OVERVIEW_zh.md`](./PROJECT_OVERVIEW_zh.md) 的"实操配套"，沉淀了环境/数据/调用过程中遇到的坑。
>
> 路径：`social_env/RUN_GUIDE_zh.md`

---

## 0. 整体决策树

```
你想做什么？
├─ 只是跑一个最小 demo，验证代码能跑
│   └─ 路线 A：local 后端 + 自造 seed 数据 + UniformSampler
│
├─ 想跑官方数据集 / experiment_eval / sotopia benchmark
│   └─ 路线 B：redis 后端（docker）+ 加载 dump.rdb
│
└─ 只是改代码，不需要执行
    └─ 不用启 redis，import 不会自动连
```

| 路线 | 数据来源 | 后端 | 入口脚本 |
|---|---|---|---|
| **A. 最小 demo** | `scripts/seed_local_demo.py` 写本地 JSON | `local` | `examples/minimalist_demo.py` |
| **B. 官方数据 + 简单批量** | redis dump.rdb（HF 下载）| `redis` | `examples/batch_demo.py`（推荐先跑通）|
| **C. 论文 benchmark** | redis dump.rdb | `redis` | `examples/experiment_eval.py` 或 `sotopia benchmark` CLI |

---

## 1. 环境准备

### 1.1 conda 环境

```bash
# 删除旧环境（如果之前装坏了）
conda deactivate
conda env remove -n social_env -y

# 重建 + 激活
conda create -n social_env python=3.11 -y
conda activate social_env

# 安装 sotopia（不带 realtime，那个要 portaudio 系统库）
cd /mnt/userdata/yphao/FC/game_MAS/social_env
pip install -e ".[test,api]"          # test 含 pytest，api 含 FastAPI
```

### 1.2 `.env` 配置

工程根 `social_env/.env`：

```bash
# OpenAI / 第三方代理
OPENAI_API_KEY=sk-xxxxxx
OPENAI_API_BASE=https://api.v3.cm/v1
OPENAI_BASE_URL=https://api.v3.cm/v1

# 后端：local 或 redis
SOTOPIA_STORAGE_BACKEND=redis        # 改成 local 走路线 A

# Redis（仅 redis 后端需要）
REDIS_OM_URL=redis://localhost:6379
```

> ⚠️ **不要写行内注释**：`SOTOPIA_STORAGE_BACKEND=redis # local` 在 `uv run --env-file` 下解析器会把整段当值。  
> ⚠️ `OPENAI_API_BASE` 末尾必须带 `/v1`，否则会 404。

每次开新 shell：
```bash
cd /mnt/userdata/yphao/FC/game_MAS/social_env
set -a; source .env; set +a
```

---

## 2. 路线 A：本地最小 demo（最快验证）

### 2.1 切到 local 后端

`.env` 改 `SOTOPIA_STORAGE_BACKEND=local`，re-source。

### 2.2 写入 seed 数据

```bash
SOTOPIA_STORAGE_BACKEND=local python scripts/seed_local_demo.py
# 输出 ~/.sotopia/data/EnvironmentProfile/<pk>.json 等
```

### 2.3 跑 demo

```bash
python examples/minimalist_demo.py
```

期望：每一轮看到 `litellm.acompletion(...)` 调用，agent1/agent2 来回对话，最后 `EpisodeLog` 打印 reward。

### 2.4 局限

- 没有 `RelationshipProfile` → ConstraintBasedSampler 不能用；
- `experiment_eval.py`、`sotopia benchmark` 跑不了；
- 索引、复杂查询能力弱（local 只是 JSON 文件夹）。

---

## 3. 路线 B：redis + 官方数据集

### 3.1 安装并启动 docker

如果 `docker --version` 没装：
```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER     # 重新登录后才生效
```

### 3.2 下载 dump.rdb

`cmu.box.com` 这台机大概率连不上，用 HF / hf-mirror。先**测网络**：

```bash
curl -I --max-time 8 https://huggingface.co/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb
curl -I --max-time 8 https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb
```

哪条返回 `HTTP/2 302/200` 就用哪条。然后：

```bash
mkdir -p ~/.sotopia/redis-data
cd ~/.sotopia/redis-data
rm -f dump.rdb

wget --content-disposition -O dump.rdb \
    'https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true'

# 验证
ls -lh dump.rdb           # 期望 30~150 MB
file dump.rdb             # 期望 "data" 或 "Redis RDB"
head -c 5 dump.rdb        # 头几字节应是 "REDIS"
```

> ❗ 文件 0 字节或者是 HTML（被代理拦截到登录页）就**不要继续**，先解决网络。

### 3.3 启动 redis-stack 容器

```bash
docker rm -f sotopia-redis 2>/dev/null

docker run -d --name sotopia-redis \
    -p 6379:6379 \
    -v ~/.sotopia/redis-data:/data \
    redis/redis-stack-server:latest

sleep 5
docker logs sotopia-redis 2>&1 | tail -20
# 期望: "DB loaded from disk: ... seconds" + "Ready to accept connections"
```

> ⚠️ 容器启动时**只读一次** `/data/dump.rdb`。换了 dump 必须 `docker rm -f` 后重建容器，不能光 restart。

### 3.4 验证 redis 后端

```bash
docker exec sotopia-redis redis-cli ping              # PONG
docker exec sotopia-redis redis-cli dbsize            # 几千+

cd /mnt/userdata/yphao/FC/game_MAS/social_env
set -a; source .env; set +a       # 已切到 redis

python -c "
from sotopia.database import EnvironmentList, EnvironmentProfile, AgentProfile
from sotopia.database.persistent_profile import RelationshipProfile
print('EnvironmentProfile  =', len(EnvironmentProfile.all()))
print('AgentProfile        =', len(AgentProfile.all()))
print('RelationshipProfile =', len(RelationshipProfile.all()))
print('hard list pk        =', EnvironmentList.get('01HAK34YPB1H1RWXQDASDKHSNS').pk)
"
```

四个数都非零 + 最后一行打印出 pk = 数据集成功加载。

---

## 4. 三种运行入口对比

| 入口 | sampler | 数据要求 | 适合 |
|---|---|---|---|
| `examples/minimalist_demo.py` | `UniformSampler` | 至少 1 env + 2 agent | "代码跑得起来吗" |
| `examples/batch_demo.py`（自定义脚本）| `UniformSampler` | 同上 | "数据/代理/redis 链路通吗"——**首推** |
| `examples/experiment_eval.py` | `ConstraintBasedSampler` | 必须有 `RelationshipProfile` + 字符串 `age_constraint` | 论文级批量实验，gin 配置 |
| `sotopia benchmark` CLI | 内部组合 + `EnvironmentList` | 必须能 `EnvironmentList.get("01HAK34Y...")`（=官方数据）| 排行榜级评测，**两边模型必须不同** |

### 4.1 batch_demo.py（推荐入口）

```bash
python examples/batch_demo.py \
    --num-episodes 2 \
    --env-model gpt-4o-mini \
    --agent1-model gpt-4o-mini \
    --agent2-model gpt-4o-mini \
    --tag first_run
```

查看产物：
```bash
python -c "
from sotopia.database import EpisodeLog
logs = list(EpisodeLog.find(EpisodeLog.tag == 'first_run').all())
print(f'episodes saved: {len(logs)}')
for ep in logs:
    print(' -', ep.pk, '| turns:', len(ep.messages), '| rewards[0]:', ep.rewards[0] if ep.rewards else None)
"
```

### 4.2 experiment_eval.py（gin 流水线）

```bash
python examples/experiment_eval.py \
    --gin_file sotopia_conf/run_async_server_in_batch.gin \
    --gin.AGENT1_MODEL='"gpt-4o-mini"' \
    --gin.AGENT2_MODEL='"gpt-4o-mini"' \
    --gin.ENV_MODEL='"gpt-4o-mini"' \
    --gin.BATCH_SIZE=2 \
    --gin.TAG='"my_first_exp"' \
    --gin.PUSH_TO_DB=True
```

> 必须用 redis 后端（ConstraintBasedSampler 依赖 `RelationshipProfile`）。

### 4.3 sotopia benchmark（排行榜）

```bash
sotopia benchmark \
    --models gpt-4o \
    --partner-model gpt-4o-mini \
    --evaluator-model gpt-4o-mini \
    --task hard --batch-size 5 --push-to-db
```

> ⚠️ **`--models` 必须不同于 `--partner-model`**，否则会被 `continue` 跳过，看到红字 `Partner model and test model, and their agent classes are the same.`，跑 0 集。  
> ⚠️ `--task hard` 90 个组合 × batch_size 5 → token 消耗大，建议先 batch_demo。

---

## 5. 关键概念速查

| 概念 | 是什么 | 在 sotopia 的作用 |
|---|---|---|
| **Redis** | 内存数据库 | 存 EnvironmentProfile / AgentProfile / EpisodeLog；用的是带 RedisJSON + RediSearch 模块的 redis-stack |
| **Docker** | 容器 | 一行命令拉起 redis-stack（不用宿主机装模块）|
| **Gin (gin-config)** | Python 配置库 | 给 `experiment_eval.py` 注入 batch / 模型等参数；只跑 demo 用不到 |
| **pytest** | 测试框架 | 跑 `tests/` 单元测试用，业务运行不依赖 |
| **UniformSampler** | env/agent 独立随机配 | 任意数据都能跑 |
| **ConstraintBasedSampler** | 先按 `relationship` 找 RelationshipProfile 对子，再按 `age_constraint`（字符串）过滤 | 必须有 RelationshipProfile + 字符串 age_constraint |
| **`age_constraint`** | EnvironmentProfile 上的 `"[(18,30),(40,60)]"` 字符串 | 限制 agent1/agent2 各自的年龄区间 |
| **`occupation_constraint`** | 同上的 occupation 字符串 | **当前 sampler 中未被使用**，只是元数据 |
| **`EnvironmentList`** | 一组 env_id + agent_index 的清单 | benchmark 的 hard task 硬编码 ID = `01HAK34YPB1H1RWXQDASDKHSNS` |

---

## 6. 数据/对象在哪儿

### 6.1 local 后端

```
~/.sotopia/data/
├── EnvironmentProfile/<pk>.json
├── AgentProfile/<pk>.json
├── EpisodeLog/<pk>.json
└── ...
```

直接 `cat` 看就行。

### 6.2 redis 后端

```
~/.sotopia/redis-data/
└── dump.rdb         # 容器启动时被加载到内存
```

容器内：
```bash
docker exec sotopia-redis redis-cli --scan --pattern '*EnvironmentProfile*' | head
docker exec sotopia-redis redis-cli JSON.GET '<key>'
```

---

## 7. 常见错误 → 修复

| 报错关键词 | 含义 | 修复 |
|---|---|---|
| `error: No pyproject.toml found` | 你不在 `social_env/` 目录 | `cd /mnt/userdata/yphao/FC/game_MAS/social_env` |
| `Failed to build pyaudio` | 缺 `portaudio.h`（系统库）| `pip install -e ".[test,api]"`，**别带 `realtime`** |
| `ModuleNotFoundError: No module named 'redis.credentials'` | `uv run` 偷偷在项目下建了 `.venv` 并搞坏了 | `rm -rf .venv` 后用 conda 的 python 直接跑 |
| `warning: Failed to parse environment file .env at position N` | `.env` 里有非法行（行内注释、空格、换行被吞）| 重写 `.env`，每行 `KEY=VALUE`，不要写注释 |
| `redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379` | 后端切 redis 但容器没起 | `docker start sotopia-redis` 或重新 docker run |
| `ValueError: No environment candidates available for sampling.` | 库里 0 条 EnvironmentProfile | local：跑 `seed_local_demo.py`；redis：dump.rdb 没加载 |
| `AssertionError: assert isinstance(age_contraint, str)` | EnvironmentProfile.age_constraint 是 None | 该 env 不能给 ConstraintBasedSampler 用，改 UniformSampler 或换 redis 数据 |
| `openai.NotFoundError: 404 Invalid URL (POST /chat/completions)` | OPENAI_API_BASE 末尾没 `/v1` | 加 `/v1` |
| 卡在 `LiteLLM completion() model= gpt-4o-mini` 不动 | 出网走代理失败、连不上 LLM | 检查 `OPENAI_API_BASE`、网络代理；尝试 `curl https://api.v3.cm/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"` |
| `EnvironmentList.get("01HAK34Y...") → NotFoundError` | redis 里没数据，dump.rdb 没真正加载 | 重新检查 `~/.sotopia/redis-data/dump.rdb` 是否存在且非 0 字节，重启容器 |
| `Partner model and test model ... are the same. Please use different models.` | benchmark 故意拦截相同模型 | `--models` 与 `--partner-model` 用不同模型；只想测试链路用 batch_demo |

---

## 8. 一次性"全检"脚本

每次开发前快速过一遍：

```bash
cd /mnt/userdata/yphao/FC/game_MAS/social_env

echo "=== conda env ==="
which python
python --version

echo "=== .env ==="
cat .env

echo "=== source .env ==="
set -a; source .env; set +a
echo "BACKEND = $SOTOPIA_STORAGE_BACKEND"
echo "REDIS   = $REDIS_OM_URL"
echo "API_BASE= $OPENAI_API_BASE"

if [ "$SOTOPIA_STORAGE_BACKEND" = "redis" ]; then
    echo "=== docker ==="
    docker ps --filter name=sotopia-redis
    docker exec sotopia-redis redis-cli ping
    docker exec sotopia-redis redis-cli dbsize
fi

echo "=== sotopia data sanity ==="
python -c "
from sotopia.database import EnvironmentProfile, AgentProfile
from sotopia.database.persistent_profile import RelationshipProfile
print('EnvironmentProfile  =', len(EnvironmentProfile.all()))
print('AgentProfile        =', len(AgentProfile.all()))
print('RelationshipProfile =', len(RelationshipProfile.all()))
"
```

四个数都满足你的预期，再去跑入口脚本。

---

## 9. 最常用命令速查

```bash
# 激活
conda activate social_env
cd /mnt/userdata/yphao/FC/game_MAS/social_env
set -a; source .env; set +a

# 容器管理
docker start sotopia-redis           # 开机后只需 start
docker stop  sotopia-redis
docker logs  sotopia-redis | tail -20
docker rm -f sotopia-redis           # 想换 dump.rdb 时用

# 跑测试
pytest tests/ -x                     # 跑代码单元测试

# 跑 demo
python examples/minimalist_demo.py
python examples/batch_demo.py --num-episodes 2 --tag try1

# 列 episode
python -c "
from sotopia.database import EpisodeLog
for ep in EpisodeLog.find(EpisodeLog.tag == 'try1').all():
    print(ep.pk, ep.environment, ep.rewards[0] if ep.rewards else None)
"
```

---

## 10. 当前已知未解决项（路线 B 进度）

- [ ] dump.rdb 在你这台机器是否真的下载成功（确认 `ls -lh ~/.sotopia/redis-data/dump.rdb` 非 0 字节）；
- [ ] 代理 `https://api.v3.cm/v1` 支持哪些模型（影响能否同时给 `--models` / `--partner-model` 配两个不同模型）；
  ```bash
  curl -s https://api.v3.cm/v1/models \
      -H "Authorization: Bearer $OPENAI_API_KEY" | python -m json.tool | head -40
  ```
- [ ] 跑通一次 `examples/batch_demo.py --num-episodes 1`（确认 redis 数据 + LLM 链路都通）。

按 §0 的决策树和 §3 的步骤执行，遇到 §7 表格里的报错按表修。



下面是三个文件的对比；最后再附一个紧急说明（你诊断结果暴露的两件事）。

## 三个入口的本质区别

| 维度 | `examples/minimalist_demo.py` | `examples/experiment_eval.py` | `sotopia/cli/benchmark/benchmark.py` |
|---|---|---|---|
| **定位** | 5 行 demo，验证环境是否能跑 | **可配置的科研实验脚本** | **官方标准 benchmark CLI** |
| **调用方式** | `python examples/minimalist_demo.py` | `python examples/experiment_eval.py --gin_file <file.gin>` | `sotopia benchmark --models ... --partner-model ... --task hard ...` |
| **配置来源** | 代码里硬编码 | **gin-config 文件**（`@gin.configurable`，所有参数都从 .gin 文件覆盖） | **typer CLI 参数** |
| **采样策略** | `UniformSampler()`（随机抽 1 个 env+随便配 agent） | `ConstraintBasedSampler`（按 env 的 `age/occupation` 约束 + 关系图找 agent，要求库里有 `RelationshipProfile`） | 直接用 **官方固定的 `EnvironmentList` + `EnvAgentComboStorage`**（hard 任务硬编码 pk `01HAK34YPB1H1RWXQDASDKHSNS`，覆盖 100 个 env-agent 组合） |
| **任务范围** | 单条对话 | 遍历库里**所有** `EnvironmentProfile`（或你指定的 env_ids 子集） | 三种预设：`hard` / `cooperative` / `competitive`（可选），每种是官方挑好的子集 |
| **Episode 数** | 1 | 取决于 env 数量 × 配置（默认每个 env 抽**第一个**未跑过的 agent 组合，一次一条） | 每个 hard env 都遍历 0/1 两个 index → 100 episode |
| **批处理** | 无 | `batch_size`（gin 配置） | `--batch-size`（默认 10） |
| **去重逻辑** | 无 | `check_existing_episodes(env_id, agent_ids, models, tag)`：**严格匹配 models** | `_list_all_env_agent_combo_not_in_db(...)`：**带 `agent_classes` + 区分 0/1 index 的更细去重**，且包了 `while True` 直到全跑完或重试 5 次 |
| **写库** | `run_async_server` 默认不写库（看你传不传 tag/`push_to_db`） | 一律 `run_async_server(...)`，没显式 push_to_db，跟你传的 gin 配 | 显式 `--push-to-db` 才写；不写就只是测试 LLM 调用 |
| **结果展示** | 只在终端打日志 | 只写 `EpisodeLog` 到 Redis，没有自带可视化 | 跑完会自动调 `benchmark_display(...)` 把 7 个维度（BEL/REL/KNO/SEC/SOC/FIN/GOAL）按模型聚合成 rich 表格，可加 `--output-to-jsonl` 导出 jsonl |
| **评测器** | 用 `run_async_server` 内置的 evaluators | 显式构造 `EpisodeLLMEvaluator(EvaluationForAgents[SotopiaDimensions])` + `RuleBasedTerminatedEvaluator(max_turn=20)` | 同 `experiment_eval.py`，但额外做"`rewards[0] is float`"的过滤删除（脏数据清洗） |
| **依赖** | 只需 OpenAI key + sampler 能从 Redis/local 取到 env | **必须有 Redis 全量数据**（含 `RelationshipProfile`、`AgentProfile`、`EnvironmentProfile`），否则 `ConstraintBasedSampler` 抛 `assert isinstance(age_constraint, str)` | **必须有 Redis 数据 + `EnvironmentList:01HAK34YPB1H1RWXQDASDKHSNS`**（hard 任务的固定 pk），否则起手就 NotFoundError |
| **日志** | RichHandler，行级日志到 stdout | RichHandler + **FileHandler 写到 `./logs/HH_MM_DD_MM_YYYY_<git_hash>.log`** | 仅 RichHandler |
| **可控性** | 几乎为 0 | 高（gin 文件可换任何参数：模型组合、env 子集、tag、batch_size …） | 中（typer 参数受限于设计的几个开关，但 partner/evaluator/task 都能改） |
| **典型用途** | 第一次装好后跑通走流程；做 import / 网络代理调试 | **写论文做对比实验**：自己控制场景子集、模型组合、评测器、tag，结果用 SQL / 自己写脚本聚合 | **复现 sotopia-pi 论文里的标准结果**：固定的 100 个 hard env，与公开 leaderboard 一致 |

## 一句话三选一

- 想"先跑通看效果" → `minimalist_demo.py`
- 想"做自己科研对比，挑数据 + 自定义评测" → `experiment_eval.py`
- 想"复现官方 leaderboard、和别人比分数" → `sotopia/cli/benchmark/benchmark.py`（即 `sotopia benchmark` 命令）

## 共用底盘

三者最后都 **走同一个 `run_async_server(...)`**（`sotopia/server.py`）。它负责：构造 `ParallelSotopiaEnv` + `LLMAgent` → 跑 `astep()` 循环 → 调 evaluator → 构造 `EpisodeLog` → 落库（如 `push_to_db=True`）。所以三个文件的差异只在"**怎么挑场景 / 怎么配模型 / 怎么聚合结果**"，**不在对话与评测内核**。
