# Sotopia 数据手册（中文）

> 本文聚焦"**数据**"：从哪里来、怎么读、怎么导出、怎么补、坑在哪。
> 配套文件：[`PROJECT_OVERVIEW_zh.md`](./PROJECT_OVERVIEW_zh.md)（项目地图）、[`RUN_GUIDE_zh.md`](./RUN_GUIDE_zh.md)（环境/运行）。
>
> 路径：`social_env/DATA_GUIDE_zh.md`

---

## 0. 数据全景

```
┌────────── 来源 ──────────┐    ┌──── 数据库（Redis 或本地）────┐    ┌────── 人可读输出 ───────┐
│ HF dump.rdb (sotopia-pi) │    │                                │    │ JSON 全量 dump            │
│ HF parquet/json          │ ─► │ EnvironmentProfile             │ ─► │ JSONL serialization       │
│ Mutual-Friend / Craigslist│    │ AgentProfile                   │    │ CSV  serialization        │
│ generate_*.py + LLM      │ ─► │ RelationshipProfile            │ ─► │ render_for_humans()       │
│ seed_local_demo.py       │    │ EnvironmentList                │    │ Streamlit UI              │
│ POST /api/...            │    │ EpisodeLog                     │    │ pandas / Excel            │
│ jsonl_to_*  反序列化      │    │ EnvAgentComboStorage           │    │                           │
└──────────────────────────┘    └────────────────────────────────┘    └───────────────────────────┘
```

数据库里有 6 类对象（即 6 张"表"）：

| 模型 | 内容 | 字段要点 |
|---|---|---|
| **EnvironmentProfile** | 一个社交场景 | scenario / agent_goals / relationship / age_constraint / occupation_constraint |
| **AgentProfile** | 一个虚拟角色 | name / age / occupation / personality / secret |
| **RelationshipProfile** | 两个 agent 之间的关系 | agent_1_id / agent_2_id / relationship / background_story |
| **EnvironmentList** | 一组 env+agent_index 清单 | benchmark `hard` 任务硬编码 PK = `01HAK34YPB1H1RWXQDASDKHSNS` |
| **EnvAgentComboStorage** | env 与 agent 的"现成组合"缓存 | env_id / agent_ids（由 sampler 写入）|
| **EpisodeLog** | 一次完整对话的记录 | environment / agents / messages / rewards / reasoning |

---

## 1. 从 HuggingFace 导入官方数据集

### 1.1 三个公开数据集

| ID | 内容 | 推荐 URL |
|---|---|---|
| **sotopia-pi**（推荐）| ACL 2024，包含 ICLR 数据 + 训练集 | `https://huggingface.co/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb` |
| **agent_vs_script** | "脚本 vs 实时" 对比 | `https://huggingface.co/datasets/cmu-lti/agent_vs_script/resolve/main/dump.rdb` |
| **sotopia**（ICLR 2024）| 经典版，仅 cmu.box.com（多数国内机器连不上）| 略 |

国内机器优先 `https://hf-mirror.com/...` 镜像。

### 1.2 下载 dump.rdb

```bash
# 先测网络（哪个返回 200/302 就用哪个）
curl -I --max-time 8 https://huggingface.co/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb
curl -I --max-time 8 https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb

# 下载（关键：先停容器，避免 redis 自己 autosave 把它覆盖）
docker stop sotopia-redis 2>/dev/null

mkdir -p ~/.sotopia/redis-data && cd ~/.sotopia/redis-data
mv dump.rdb dump.rdb.bak 2>/dev/null
rm -f dump.rdb

wget --content-disposition -O dump.rdb \
    'https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true'

# 校验
ls -lh dump.rdb           # ~245 MB
file dump.rdb             # data
head -c 5 dump.rdb        # REDIS
```

❗ 文件 0 字节 / file 输出 HTML / 头不是 REDIS → 没下成功，**不要**继续。

### 1.3 启动 redis-stack 容器

```bash
docker rm -f sotopia-redis 2>/dev/null
docker run -d --name sotopia-redis \
    -p 6379:6379 \
    -v ~/.sotopia/redis-data:/data \
    redis/redis-stack-server:latest \
    --save "" --appendonly no       # 关闭 autosave，保护 dump.rdb 不被覆盖

sleep 5
docker logs sotopia-redis 2>&1 | tail -25
```

期望日志：
```
RDB memory usage when created  ~533 Mb
Done loading RDB, keys loaded: 39000+
Ready to accept connections tcp
```

> **关键步骤——重建 RediSearch 索引**：dump.rdb 不带索引，直接 `.all()` 会返回 0。

```bash
cd /mnt/userdata/yphao/FC/game_MAS/social_env
set -a; source .env; set +a

python - <<'PY'
from redis_om import Migrator
from sotopia.database import EnvironmentProfile, AgentProfile, RelationshipProfile, EpisodeLog
from sotopia.database.persistent_profile import EnvironmentList
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage
print("Running Migrator...")
Migrator().run()
print("Done.")
PY
```

**Migrator 必须在 Python 里跑**（命令行无 `redis-om migrate`），且在 import 完所有模型之后。

### 1.4 验证

```bash
docker exec sotopia-redis redis-cli dbsize                         # 39000+
docker exec sotopia-redis redis-cli --scan --pattern '*EnvironmentProfile*' | wc -l   # 885
docker exec sotopia-redis redis-cli --scan --pattern '*AgentProfile*'      | wc -l   # 41
```

```python
from sotopia.database import EnvironmentProfile, AgentProfile, RelationshipProfile
from sotopia.database.persistent_profile import EnvironmentList
print(len(EnvironmentProfile.all()))   # 884
print(len(AgentProfile.all()))         # 40
print(len(RelationshipProfile.all()))  # 120
print(len(EnvironmentList.all()))      # 1 或 2
```

---

## 2. 把 redis 数据导出成可读文件

> Redis dump.rdb 是二进制；下面四种粒度由低到高展示数据。

### 2.1 redis-cli 看键 / 单条 JSON

```bash
docker exec sotopia-redis redis-cli --scan --pattern '*EnvironmentProfile*' | head
docker exec sotopia-redis redis-cli JSON.GET ':sotopia.database.persistent_profile.EnvironmentProfile:01H9FG...'
```

### 2.2 全量 JSON 导出（推荐）

```bash
mkdir -p ~/.sotopia/export && cd /mnt/userdata/yphao/FC/game_MAS/social_env
set -a; source .env; set +a

python - <<'PY'
import json, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from sotopia.database import (
    EnvironmentProfile, AgentProfile, RelationshipProfile, EpisodeLog,
)
from sotopia.database.persistent_profile import EnvironmentList
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage

out = Path.home() / ".sotopia" / "export"
out.mkdir(parents=True, exist_ok=True)

# 不会出现 schema 报错的几个表
for cls in [
    EnvironmentProfile, AgentProfile, RelationshipProfile,
    EnvironmentList, EnvAgentComboStorage,
]:
    items = cls.all()
    name  = cls.__name__
    file  = out / f"{name}.json"
    with file.open("w") as f:
        json.dump([item.dict() for item in items], f,
                  indent=2, ensure_ascii=False, default=str)
    print(f"{name:25s} {len(items):>6d}  →  {file}")

# EpisodeLog: 老 schema 可能验证失败，绕过 pydantic 直读
import redis
r = redis.from_url(os.environ.get("REDIS_OM_URL", "redis://localhost:6379"))
keys = [k for k in r.scan_iter(match="*EpisodeLog*", count=1000)
        if b":idx:" not in k]
raw_logs = []
for k in keys:
    try:
        doc = r.execute_command("JSON.GET", k)
        if doc:
            raw_logs.append(json.loads(doc))
    except Exception:
        pass
with (out / "EpisodeLog_raw.json").open("w") as f:
    json.dump(raw_logs, f, indent=2, ensure_ascii=False, default=str)
print(f"{'EpisodeLog (raw)':25s} {len(raw_logs):>6d}  →  {out / 'EpisodeLog_raw.json'}")
PY
```

输出在 `~/.sotopia/export/*.json`，可直接 `less` / `jq` / VSCode 浏览。

### 2.3 项目自带 serialization（研究友好的 JSONL/CSV）

`sotopia/database/serialization.py` 里已有：

```python
from sotopia.database.serialization import (
    environmentprofiles_to_jsonl, environmentprofiles_to_csv,
    agentprofiles_to_jsonl,       agentprofiles_to_csv,
    relationshipprofiles_to_jsonl, relationshipprofiles_to_csv,
    episodes_to_jsonl,             episodes_to_csv,
    # 反向：
    jsonl_to_environmentprofiles, jsonl_to_agentprofiles,
    jsonl_to_relationshipprofiles, jsonl_to_episodes,
)
```

```python
import os
os.makedirs("export", exist_ok=True)
environmentprofiles_to_jsonl(EnvironmentProfile.all(), "export/env_profiles.jsonl")
agentprofiles_to_csv         (AgentProfile.all(),       "export/agent_profiles.csv")
```

`episodes_to_jsonl` 会把 episode 串好成 `TwoAgentEpisodeWithScenarioBackgroundGoals`，自动 join 上 scenario / agent 背景 / 对话 / 评分，**直接可用作下游 finetune 数据**。

### 2.4 渲染对话稿（人审用）

```python
from sotopia.database import EpisodeLog
ep = EpisodeLog.find().all()[0]            # 取第一条
agent_lines, transcript = ep.render_for_humans()
print("=== Agents ===");      [print(l) for l in agent_lines]
print("\n=== Transcript ==="); [print(t) for t in transcript]
```

### 2.5 GUI

```bash
streamlit run ui/app.py
# 浏览器打开 http://localhost:8501
```

可分类浏览 / 编辑 EnvironmentProfile / AgentProfile / EpisodeLog。

---

## 3. ⚠️ 老版本 EpisodeLog schema 不兼容

下载到的官方 dump.rdb **是用更老版本 sotopia 写出来的**。新代码读老 EpisodeLog 时会报：

```
reasoning  → 期望 str，拿到 list[str]
rewards.N → 期望 tuple[float, dict[str,float]]，拿到 [float, float]
```

不影响**写**新 EpisodeLog（batch_demo / experiment_eval / benchmark 写出来的都是新 schema），只影响**读**老的。

### 三种处理（按需选）

| 选 | 行动 | 影响 |
|---|---|---|
| **A. 跳过 / 直读 raw**（推荐）| 用 §2.2 那段脚本里的 `EpisodeLog_raw.json` 路径，绕过 pydantic | 老 episode 仍可看；新代码不报错 |
| **B. 删掉老 EpisodeLog** | `docker exec sotopia-redis redis-cli --scan --pattern '*EpisodeLog*' \| grep -v ':idx:' \| xargs -I{} docker exec sotopia-redis redis-cli del {}` | 老对话稿丢失（已备份 `dump.rdb.gold` 就 OK）|
| **C. 写迁移脚本** | 把老 `reasoning: list[str]` join 成 str，把 `rewards: list[list]` 改成 `list[float]` 或 `[(float, {})]` | 工作量大，仅当研究必须用老 episode 数据时考虑 |

### 删除老 EpisodeLog 的安全步骤

```bash
# 先备份再删
cp ~/.sotopia/redis-data/dump.rdb ~/.sotopia/redis-data/dump.rdb.gold

docker exec sotopia-redis bash -c '
    redis-cli --scan --pattern "*EpisodeLog*" | grep -v ":idx:" | \
    while read k; do redis-cli del "$k"; done
'

# 验证
docker exec sotopia-redis redis-cli --scan --pattern '*EpisodeLog*' | wc -l    # 0
```

---

## 4. 防止 dump.rdb 被覆盖

redis-stack **默认开启 RDB autosave**（`save 3600 1 ...`）。容器跑着会周期性把内存 dump 回 `/data/dump.rdb`，**覆盖你下载的官方版**。

### 永久防护

启动容器时关掉自动保存：

```bash
docker run -d --name sotopia-redis \
    -p 6379:6379 \
    -v ~/.sotopia/redis-data:/data \
    redis/redis-stack-server:latest \
    --save "" --appendonly no
```

### 同时备份 dump.rdb

```bash
cp ~/.sotopia/redis-data/dump.rdb ~/.sotopia/redis-data/dump.rdb.gold
```

需要恢复：

```bash
docker stop sotopia-redis
cp ~/.sotopia/redis-data/dump.rdb.gold ~/.sotopia/redis-data/dump.rdb
docker rm -f sotopia-redis
docker run -d --name sotopia-redis -p 6379:6379 \
    -v ~/.sotopia/redis-data:/data redis/redis-stack-server:latest \
    --save "" --appendonly no
```

### 想保留**新写入**的 EpisodeLog

关掉 autosave 后，跑实验产生的新 EpisodeLog **只在内存**，重启容器丢失。需要时手动持久化：

```bash
docker exec sotopia-redis redis-cli BGSAVE
# 等几秒，确认完成
docker exec sotopia-redis redis-cli LASTSAVE
```

`BGSAVE` 会用当前内存生成新的 dump.rdb（覆盖原文件，所以**先 cp 一份 gold**）。

---

## 5. 用 LLM 合成新数据

### 5.1 项目内置入口

| 脚本 / 函数 | 生成什么 | 调用方式 |
|---|---|---|
| `sotopia.generation_utils.agenerate_env_profile()` | 单个 EnvironmentProfile | 直接 `await` |
| `sotopia.generation_utils.agenerate_relationship_profile()` | RelationshipProfile | 直接 `await` |
| `sotopia.agents.generate_agent_background.*` | AgentProfile 背景填充 | 直接调用 |
| `examples/generate_specific_envs.py` | env from `mutual_friends` / `craigslist_bargains` | LLM 改写公开数据 |
| `examples/generate_scenarios.py` | 多个 EnvironmentProfile 入库 | typer CLI |
| `examples/generate_script.py` | 给 env 直接生成"剧本式"对话 | 单次 LLM |

### 5.2 最小例子

```python
import asyncio
from sotopia.generation_utils import agenerate_env_profile

async def main():
    for _ in range(5):
        env = await agenerate_env_profile(
            model_name="gpt-4o-mini",
            inspiration_prompt="A negotiation between landlord and tenant about lease renewal.",
        )
        env.save()      # 直接进 redis
        print("saved", env.codename, "pk=", env.pk)

asyncio.run(main())
```

跑完就能在数据库里看到 5 条新场景，可以直接被 `UniformSampler` / `ConstraintBasedSampler` 抽到。

### 5.3 手写 seed（不调用 LLM）

参考 `scripts/seed_local_demo.py`，纯 Python 构造对象 → `.save()`。
适合做单元测试、最小复现。

---

## 6. 用 HF parquet/json 灌库（精挑细选）

如果只想用 sotopia-pi 的一部分（比如 30 条）：

```python
from datasets import load_dataset
ds = load_dataset("cmu-lti/sotopia-pi", split="train")
print(ds)
print(ds[0])

from sotopia.database import EnvironmentProfile
for row in ds.select(range(30)):
    EnvironmentProfile(
        codename=row["codename"],
        scenario=row["scenario"],
        agent_goals=row["agent_goals"],
        relationship=row["relationship"],
        age_constraint=row["age_constraint"],
        occupation_constraint=row["occupation_constraint"],
    ).save()
```

走国内镜像：`HF_ENDPOINT=https://hf-mirror.com python script.py`

---

## 7. JSONL 反向导入（恢复 / 迁移）

```python
from sotopia.database.serialization import (
    jsonl_to_environmentprofiles, jsonl_to_agentprofiles,
    jsonl_to_relationshipprofiles, jsonl_to_episodes,
)

# 别人 dump 的 jsonl → 灌回我的 redis
for ep in jsonl_to_environmentprofiles("export/env_profiles.jsonl"):
    ep.save()
```

---

## 8. local 后端的数据放在哪

```
~/.sotopia/data/
├── EnvironmentProfile/<pk>.json
├── AgentProfile/<pk>.json
├── RelationshipProfile/<pk>.json
├── EnvironmentList/<pk>.json
└── EpisodeLog/<pk>.json
```

直接 `cat`。

`local` 与 `redis` 后端互不干扰，可以并存：
- `~/.sotopia/data/` ← local 后端读写这里
- `~/.sotopia/redis-data/dump.rdb` ← redis 后端读写这里

---

## 9. 常见问题速查

| 现象 | 真因 | 修复 |
|---|---|---|
| `EnvironmentProfile.all() = 0`，但 `redis-cli --scan` 数千 key | RediSearch 索引没建 | `python -c "from redis_om import Migrator; ...; Migrator().run()"` |
| dump.rdb 只有几百字节，dbsize=4 | 容器启动时挂载目录是空的，redis 自己 autosave 写了空快照 | 关 autosave (`--save ""`)，重新下 dump.rdb，`docker rm -f` 后重启 |
| `keys loaded: 4` 而不是几万 | 同上 | 同上 |
| `pydantic ValidationError: reasoning ... list[str]` | 老 dump 的 EpisodeLog schema 不兼容 | §3 选项 A（raw 导出）或 B（删掉）|
| `ImportError: cannot import name 'EnvironmentList' from 'sotopia.database'` | 它没在 `__init__.py` re-export | 改成 `from sotopia.database.persistent_profile import EnvironmentList` |
| `cmu.box.com Connection timed out` | box.com 国内不可达 | 改用 `cmu-lti/sotopia-pi` HF 镜像 |
| `wget` 下到 HTML 而不是 dump.rdb | 代理拦到登录页 | `unset HTTPS_PROXY HTTP_PROXY` 后改 hf-mirror |
| 重启容器后老 dump.rdb 被覆盖 | 没关 autosave + 没备份 gold | §4：`--save ""` + 备份 `dump.rdb.gold` |
| `EnvAgentComboStorage = 4886` 但其它表 0 | 上次重建索引前的状态 | 跑 Migrator 后重新查 |

---

## 10. 一份完整流程的清单（首次配数据）

按顺序执行：

```bash
# 1. 网络
curl -I --max-time 8 https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb

# 2. 下载
docker stop sotopia-redis 2>/dev/null
mkdir -p ~/.sotopia/redis-data && cd ~/.sotopia/redis-data
rm -f dump.rdb
wget --content-disposition -O dump.rdb \
    'https://hf-mirror.com/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true'
ls -lh dump.rdb && file dump.rdb && head -c 5 dump.rdb && echo

# 3. 备份 gold
cp dump.rdb dump.rdb.gold

# 4. 启容器（关 autosave）
docker rm -f sotopia-redis 2>/dev/null
docker run -d --name sotopia-redis \
    -p 6379:6379 \
    -v ~/.sotopia/redis-data:/data \
    redis/redis-stack-server:latest \
    --save "" --appendonly no
sleep 5
docker logs sotopia-redis 2>&1 | tail -10

# 5. 索引 + 校验
cd /mnt/userdata/yphao/FC/game_MAS/social_env
set -a; source .env; set +a
python - <<'PY'
from redis_om import Migrator
from sotopia.database import EnvironmentProfile, AgentProfile, RelationshipProfile, EpisodeLog
from sotopia.database.persistent_profile import EnvironmentList
from sotopia.database.env_agent_combo_storage import EnvAgentComboStorage
Migrator().run()
print("EnvironmentProfile  =", len(EnvironmentProfile.all()))
print("AgentProfile        =", len(AgentProfile.all()))
print("RelationshipProfile =", len(RelationshipProfile.all()))
print("EnvironmentList     =", len(EnvironmentList.all()))
print("EnvAgentComboStorage=", len(EnvAgentComboStorage.all()))
PY

# 6. 导出归档
mkdir -p ~/.sotopia/export
# §2.2 那段脚本，跳过老 EpisodeLog

# 7. 删/绕过老 EpisodeLog
docker exec sotopia-redis bash -c '
    redis-cli --scan --pattern "*EpisodeLog*" | grep -v ":idx:" | \
    while read k; do redis-cli del "$k"; done
'

# 8. 跑实验
python examples/batch_demo.py --num-episodes 1 --tag first_real_run

# 9. 看新 episode
python -c "
from sotopia.database import EpisodeLog
for ep in EpisodeLog.find(EpisodeLog.tag == 'first_real_run').all():
    print(ep.pk, '| turns:', len(ep.messages))
"

# 10. 持久化新 episode
docker exec sotopia-redis redis-cli BGSAVE
sleep 3
docker exec sotopia-redis redis-cli LASTSAVE
```

---

## 11. 关键文件位置速查

```
social_env/
├── sotopia/
│   ├── database/
│   │   ├── __init__.py                ← 启动时按 SOTOPIA_STORAGE_BACKEND 选 redis/local
│   │   ├── persistent_profile.py      ← EnvironmentProfile / AgentProfile / EnvironmentList ...
│   │   ├── env_agent_combo_storage.py ← EnvAgentComboStorage
│   │   ├── logs.py                    ← EpisodeLog
│   │   ├── serialization.py           ← *_to_jsonl / *_to_csv / jsonl_to_*
│   │   └── storage_backend.py         ← LocalJSONBackend / RedisBackend 抽象
│   ├── generation_utils/
│   │   └── generate.py                ← agenerate_env_profile / agenerate_relationship_profile
│   └── cli/install/
│       ├── install.py                 ← `sotopia install` 命令
│       └── published_datasets.json    ← 公开数据集 URL 清单
├── examples/
│   ├── generate_specific_envs.py      ← 公开数据集 → LLM → EnvironmentProfile
│   ├── generate_scenarios.py          ← 批量入库
│   └── generate_script.py             ← 直接生成对话剧本
├── scripts/seed_local_demo.py         ← 手工 seed（无 LLM）
└── ui/                                 ← Streamlit GUI 浏览/编辑数据
```

---

## 12. 总结：路线对照

| 你想… | 路线 | 大致命令 |
|---|---|---|
| 跑 minimalist_demo | local + seed | `python scripts/seed_local_demo.py && python examples/minimalist_demo.py` |
| 跑 batch_demo / experiment_eval / benchmark | redis + dump.rdb | §10 全流程 |
| 看 episode 内容 | render_for_humans / Streamlit UI | `streamlit run ui/app.py` |
| 导出做下游 finetune | episodes_to_jsonl | `from sotopia.database.serialization import episodes_to_jsonl; episodes_to_jsonl(EpisodeLog.all(), "out.jsonl")` |
| 添加新场景（手写）| seed_local_demo 风格 | 改 EnvironmentProfile 字段 → `.save()` |
| 添加新场景（LLM 合成）| `agenerate_env_profile` | §5.2 |
| 老 EpisodeLog 报错 | §3 | A：raw 导出 / B：删 / C：写迁移 |
| dump.rdb 被覆盖 | §4 | autosave 关掉 + gold 备份 |
| Migrator 必须跑 | 加载 dump.rdb 后必跑 | §1.3 末段 |

按上面顺序处理 + 备份 dump.rdb.gold，整个数据流就能稳定下来。






















# Redis数据导出

`export/` 实际上**所有 5 张表都齐全**（之前 IDE recent files 只显示了一部分）：

| 表 | 条数 |
|---|---|
| AgentProfile | 40 |
| EnvironmentProfile | 884 |
| EnvironmentList | 1（含 `01HAK34YPB1H1RWXQDASDKHSNS`） |
| EnvAgentComboStorage | 4886 |
| RelationshipProfile | 120 |

**所以答案：能完全用 `export/` 直接跑 local 后端**，连 Redis 都可以彻底不要。

## 三步上路

```bash
cd /mnt/userdata/yphao/FC/game_MAS/social_env

# 1) 把 export/*.json 导成本地后端文件 (~/.sotopia/data/<Class>/<pk>.json)
SOTOPIA_STORAGE_BACKEND=local python scripts/setup_local_data.py

# 2) 切到 local 后端，跑 benchmark
export SOTOPIA_STORAGE_BACKEND=local
unset REDIS_OM_URL    # 可选：避免 redis-om 误以为还要连 Redis

sotopia benchmark \
  --models gpt-4o \
  --partner-model gpt-4o-mini \
  --evaluator-model gpt-4o-mini \
  --task hard \
  --batch-size 5 \
  --tag local_run_trial0 \
  --push-to-db

# 3) 跑完只看分数
sotopia benchmark \
  --models gpt-4o \
  --partner-model gpt-4o-mini \
  --evaluator-model gpt-4o-mini \
  --task hard \
  --tag local_run_trial0 \
  --only-show-performance
```

## 关键差异（跟 Redis 后端比）

| 维度 | local 后端 | Redis 后端 |
|---|---|---|
| 数据落点 | `~/.sotopia/data/<Class>/<pk>.json`，一条一文件 | Redis JSON key + RediSearch 索引 |
| 查询性能 | `find()` 全量扫目录 + 内存过滤；4886 条 EnvAgentComboStorage 这种规模没问题，10 万级会变慢 | RediSearch 二级索引，秒级 |
| 索引/Migrator | 不需要，自然不会有 `No such index` 这类报错 | 需要 Migrator 维护，前几天那个雷就是这里 |
| pk 生成 | UUID4（看 `LocalJSONBackend.generate_pk`） | ULID（redis-om 默认） |
| 数据丢失风险 | 0：JSON 文件就在你磁盘上 | 高：容器删了 / dump.rdb 被覆盖就没了 |
| 适合场景 | PoC、单机实验、调试 | 多人共享、海量历史 episode、需要 RediSearch 复杂查询 |

## 注意事项 / 可能踩的坑

1. **`sotopia benchmark` 内部会调 `find()` 去重**（`_list_all_env_agent_combo_not_in_db`）。本地后端的 `find()` 走全量扫描，4886 条 combo 每次扫一遍 = 4886 次 `json.load` + dict 比较。**会比 Redis 慢一两个数量级**，但还能跑（百毫秒级）。如果嫌慢可以加 `--task` 用更小的子集。

2. **写库 push_to_db**：local 模式下 `epilog.save()` 会在 `~/.sotopia/data/EpisodeLog/<uuid>.json` 写一个文件。**必须传 `--tag`**，否则 `EpisodeLog.find(EpisodeLog.tag == tag)` 退化成扫所有 EpisodeLog 而不是按 tag 筛。

3. **重新设置环境变量**：`export SOTOPIA_STORAGE_BACKEND=local` 这一步**每个新 shell 都要重新做**，否则 `database/__init__.py` 默认走 redis 分支会去连 6379。

4. **测试导入是否成功**：

   ```bash
   ls ~/.sotopia/data/
   # 应看到: AgentProfile  EnvAgentComboStorage  EnvironmentList  EnvironmentProfile  RelationshipProfile
   ls ~/.sotopia/data/AgentProfile | wc -l        # 应 40
   ls ~/.sotopia/data/EnvironmentProfile | wc -l  # 应 884
   ls ~/.sotopia/data/EnvironmentList             # 应有 01HAK34YPB1H1RWXQDASDKHSNS.json
   ```

5. **重新跑会跳过已完成的 episode**：`sotopia benchmark` 通过 `find(EpisodeLog.tag == tag)` + `(env_id, agents, models)` 去重。所以你可以分多批跑，中断重启也不会重复 LLM 调用。

## 想清空重来时

```bash
rm -rf ~/.sotopia/data
SOTOPIA_STORAGE_BACKEND=local python scripts/setup_local_data.py --clear-existing
```

`--clear-existing` 会把每张表目录里的 `*.json` 先删掉再重新写，避免上次跑遗留的 `EpisodeLog/<uuid>.json` 与新数据混在一起。

---

如果第 1 步跑完你想做个最小验证（不调 LLM），可以单独测 `EnvironmentList.get(...)` 是否能从本地正确读出来：

```bash
SOTOPIA_STORAGE_BACKEND=local python -c "
from sotopia.database import EnvironmentList
hard = EnvironmentList.get('01HAK34YPB1H1RWXQDASDKHSNS')
print('name:', hard.name)
print('envs:', len(hard.environments))
print('first env:', hard.environments[0])
"
```

期待打印 `name: hard_env_set`、`envs: 20`、第一个 env 是 `01H7VFHNV13MHN97GAH73E3KM8`。这样就完全确认本地后端能正常吃 `export/` 数据了。、、、

不用数据库（即 `SOTOPIA_STORAGE_BACKEND=local`）的话，所有数据都是**普通 JSON 文件**，存放路径完全确定，可以直接 `cat` / `cp` / `git add`。

## 路径布局

根目录由 `LocalJSONBackend.__init__` 决定，写死在代码里：

```165:174:social_env/sotopia/database/storage_backend.py
    def __init__(self, base_path: str | None = None) -> None:
        """Initialize local JSON backend.

        Args:
            base_path: Base directory for storing data. Defaults to ~/.sotopia/data
        """
        if base_path is None:
            base_path = os.path.expanduser("~/.sotopia/data")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
```

`save()` 把每条记录存成**一条记录一个文件**：

```200:212:social_env/sotopia/database/storage_backend.py
    def _get_file_path(self, model_class: Type[T], pk: str) -> Path:
        """Get the file path for a specific instance.
        ...
        """
        return self._get_model_dir(model_class) / f"{pk}.json"

    def save(self, model_class: Type[T], pk: str, data: dict[str, Any]) -> None:
        """Save a model instance to a JSON file.
        ...
        """
        file_path = self._get_file_path(model_class, pk)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
```

合起来就是：

```
~/.sotopia/data/
├── AgentProfile/
│   ├── 01H5TNE5PC6YGRH72RQAM862JH.json
│   ├── 01H5TNE5PBXGRD41HXQC1ZXHVN.json
│   └── ... (40 个文件)
├── EnvironmentProfile/
│   ├── 01H7VFHNV13MHN97GAH73E3KM8.json
│   └── ... (884 个文件)
├── EnvironmentList/
│   └── 01HAK34YPB1H1RWXQDASDKHSNS.json   ← hard 任务定义
├── EnvAgentComboStorage/
│   └── ... (4886 个文件)
├── RelationshipProfile/
│   └── ... (120 个文件)
└── EpisodeLog/                             ← benchmark 跑完的对话日志在这里
    ├── <uuid4>.json
    └── ...
```

每个目录名就是 Python 类名（`model_class.__name__`），不是带模块路径的全名。

## benchmark 跑完后会新增什么

看 `server.py` 里写库那一段：每跑完一条 episode 就调 `epilog.save()`，等价于在 `~/.sotopia/data/EpisodeLog/<新生成的uuid>.json` 写一个 JSON。

跑 100 条 hard episode → 你会在 `~/.sotopia/data/EpisodeLog/` 看到 100 个新文件。每个文件长这样：

```json
{
  "pk": "f1c8a2e0-1234-...",
  "environment": "01H7VFHN5WVC5HKKVBHZBA553R",
  "agents": ["01H5TNE5PAZABGW79HJ07TACCZ", "01H5TNE5P83CZ1TDBVN74NGEEJ"],
  "tag": "local_run_trial0",
  "models": ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini"],
  "messages": [
    [["Environment", "Mia Davis", "..."], ["Environment", "William Brown", "..."], ...],
    [...],   // 第 1 turn
    ...
  ],
  "reasoning": "Environment comments: terminated: ...",
  "rewards": [
    [0.0, {"believability": 9, "relationship": 4, "knowledge": 5, ...}],
    [0.0, {...}]
  ],
  "rewards_prompt": ""
}
```

`messages` 是逐回合的对话三元组 `(发言者, 接收者, 内容)`；`rewards` 是每个 agent 的最终评测维度分数。

## 怎么查 / 怎么用

### 1. 命令行直接看

```bash
# 看跑了多少条
ls ~/.sotopia/data/EpisodeLog | wc -l

# 看最新一条
ls -t ~/.sotopia/data/EpisodeLog | head -1
cat ~/.sotopia/data/EpisodeLog/<最新文件名>

# 按 tag 过滤
grep -l '"tag": "local_run_trial0"' ~/.sotopia/data/EpisodeLog/*.json | wc -l
```

### 2. 用 Python 读

```python
import os
os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"

from sotopia.database import EpisodeLog

# 读全部
all_episodes = EpisodeLog.all()
print(f"共 {len(all_episodes)} 条")

# 按 tag 筛
mine = EpisodeLog.find(EpisodeLog.tag == "local_run_trial0").all()
for ep in mine[:3]:
    profiles, lines = ep.render_for_humans()  # 把对话渲染成人类可读
    print("=" * 60)
    for line in lines:
        print(line)
```

### 3. 跑完展示分数（不调 LLM）

```bash
SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \
  --models gpt-4o --partner-model gpt-4o-mini \
  --evaluator-model gpt-4o-mini --task hard \
  --tag local_run_trial0 --only-show-performance
```

它内部就是 `EpisodeLog.find(EpisodeLog.tag == ...)`，本地后端会扫 `~/.sotopia/data/EpisodeLog/` 下所有文件，过滤出 tag 匹配的，再按 7 个维度聚合成 rich 表格。

### 4. 一键导出成可分发的单文件

```bash
SOTOPIA_STORAGE_BACKEND=local /home/yphao/.conda/envs/social_env/bin/python -c "
import json, glob, os
recs = []
for p in glob.glob(os.path.expanduser('~/.sotopia/data/EpisodeLog/*.json')):
    with open(p) as f: recs.append(json.load(f))
with open('export/EpisodeLog.json', 'w') as f:
    json.dump(recs, f, ensure_ascii=False, indent=2)
print('exported', len(recs))
"
```

导出后可以丢进 git、发给同事、传到 HuggingFace 都行。

## 跟 Redis 后端的对比（一图概括）

| 维度 | local（不用数据库） | Redis |
|---|---|---|
| **数据形态** | 一条记录一个 JSON 文件 | Redis 内存 + dump.rdb 二进制 |
| **位置** | `~/.sotopia/data/<Class>/<pk>.json` | `~/.sotopia/redis-data/dump.rdb`（不可读） |
| **可直接看** | ✅ `cat` 即可 | ❌ 必须 `redis-cli JSON.GET` |
| **备份 / 同步** | ✅ `tar`、`rsync`、`git` | ❌ 拷 `dump.rdb` 但要停容器 |
| **跨机器复用** | ✅ 拷过去就行 | ⚠️ dump.rdb 跨架构可能有兼容问题 |
| **查询性能** | 慢（全目录扫描） | 快（RediSearch 索引） |
| **数据丢失风险** | 几乎为 0（你删才会没） | 高（容器删 / 卷被覆盖就没） |
| **适合** | PoC、调试、教学、产论文小数据 | 多人共享、海量历史、复杂查询 |

## 想清掉重来 / 防止误删

```bash
# 看占多少空间
du -sh ~/.sotopia/data/*

# 只想清 EpisodeLog（保留种子数据）
rm -rf ~/.sotopia/data/EpisodeLog/*

# 完全推倒重来
rm -rf ~/.sotopia/data/
SOTOPIA_STORAGE_BACKEND=local python scripts/setup_local_data.py
```

## 一个常被忽略的细节

local 后端**不存索引信息**（`base_models.py` 里的 `find()` 直接对所有文件做 in-memory 过滤），所以**不会出现 Redis 那种 `No such index` 报错**。但代价是 `find()` 每次都要 `glob('*.json')` + 逐文件 `json.load`，当某个目录里文件超过 1 万就明显变慢。Sotopia 的 hard 任务规模（百条 EpisodeLog 量级）完全不需要担心。

如果你后面想按"哪些 episode 评分最低"之类做分析，最高效的做法是先 `EpisodeLog.all()` 一次拿到内存，然后用纯 Python `list comprehension` 自己筛选——而不是写多个 `find(...)`。





















# 数据生成指南

本节回答两个问题：**"我要怎么得到一份能跑 benchmark 的数据？"** 以及 **"已有数据怎么扩展成 V2？"**

所有命令默认 `cwd=/mnt/userdata/yphao/FC/game_MAS/social_env`，且 `SOTOPIA_STORAGE_BACKEND=local`（本地 JSON 后端）。

---

## 1. 总览：三条造数据路径

按 **"数据从哪儿来"** 分三种场景，脚本各自承担一段链路，可单独用也可串联：

| 场景 | 数据来源 | 用什么脚本 | 一句话定位 |
|---|---|---|---|
| **A1. 完全从零（手写）** | 脚本字面量 | `scripts/generate_from_scratch.py` | 8 角色 + 6 场景的手写包；不依赖 Redis、export、网络 |
| **A2. 完全从零（LLM）** | 主题关键词 + LLM | `scripts/generate_from_scratch_with_llm.py` | 给一个 theme，自动派生 brief → LLM 批量造 agent / env，再走 A1 的同款落库 |
| **B. 从 export 导入** | `social_env/export/*.json` | `scripts/setup_local_data.py` | 把 Sotopia 官方数据导进 `~/.sotopia/data/` |
| **C. 升级 / 扩展为 V2** | A 或 B 已落地的 V1 数据 | `scripts/generate_v2_seed.py`（无 LLM 升级）<br>`scripts/generate_v2_with_llm.py`（LLM 扩 V2 env） | 在 V1 基础上叠加 V2 字段、生成 EventScript / Contract / SystemStateSnapshot |

每条路径都把数据写到本地后端的统一目录布局：

```
~/.sotopia/data/
├── AgentProfile/<pk>.json
├── EnvironmentProfile/<pk>.json
├── RelationshipProfile/<pk>.json
├── EnvAgentComboStorage/<pk>.json
├── EnvironmentList/<pk>.json
├── EpisodeLog/<pk>.json              ← benchmark 跑出来的日志
├── AgentProfileV2/<pk>.json          ← V2 才有
├── EnvironmentProfileV2/<pk>.json
├── EventScript/<pk>.json
├── Contract/<pk>.json
└── SystemStateSnapshot/<pk>.json
```

---

## 2. 路径 A1：完全从零（手写，推荐第一次就跑这个）

不依赖 Redis、不依赖 `export/*.json`、不依赖网络。把 `scripts/generate_from_scratch.py` 里的字典字面量直接落库即可。

### 2.1 一条命令拿到完整数据

```bash
SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python \
    scripts/generate_from_scratch.py --clean --with-v2
```

| 参数 | 作用 |
|---|---|
| `--clean` | 先 `rm -rf ~/.sotopia/data/`，确保从零开始 |
| `--with-v2` | 同时落 V2 数据（AgentProfileV2 / EventScript / Contract / SystemStateSnapshot） |
| `--n-agents N` | 取前 N 个 archetype 作为 AgentProfile，默认 8 |
| `--n-envs M` | 取前 M 个 archetype 作为 EnvironmentProfile，默认 6 |
| `--combos-per-env K` | 每个 env 随机配 K 对 agent，默认 2 |
| `--override-hard-list` | 把生成的 EnvironmentList.pk 设成官方 hard 列表的 ULID，让 `--task hard` 直接命中 |

### 2.2 默认产出数据清单

| 类 | 数量 | 内容 |
|---|---:|---|
| `AgentProfile` | 8 | 手写角色（工程师 / 商人 / 护士 / 议员 / 大学生 / barista / 调律师 / 记者） |
| `EnvironmentProfile` | 6 | 手写场景（修洗碗机 / 股权分配 / 共用水井 / 揭丑闻 / 合租欠租 / 政策预算） |
| `RelationshipProfile` | 28 | 两两 stranger 关系（C(8,2)） |
| `EnvAgentComboStorage` | 12 | 每个 env × 2 对 agent |
| `EnvironmentList` | 1 | `name="scratch_env_set"` |
| `AgentProfileV2` / `EnvironmentProfileV2` | 8 / 6 | V2 升级版 |
| `EventScript` | 2 | day1 供应链中断、day2 央行加息 |
| `Contract` | 1 | 借贷模板（status=proposed） |
| `SystemStateSnapshot` | 1 | day=0 初始状态 |

### 2.3 立即跑 benchmark 验证

```bash
SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \
    --models gpt-4o-mini --partner-model gpt-4o-mini \
    --evaluator-model gpt-4o-mini \
    --task scratch --tag run0 --push-to-db --batch-size 4
```

- `--task scratch`：任意非 `hard/cooperative/competitive` 的字符串都会进 `else` 分支，使用全部 `EnvAgentComboStorage`，刚好覆盖刚造的 12 条。
- `--push-to-db`：**必加**，否则 episode 不落盘（参见 [§5.2 已知坑](#52-已知坑)）。

如果坚持要用 `--task hard`，重造时加 `--override-hard-list`：

```bash
python scripts/generate_from_scratch.py --clean --override-hard-list
sotopia benchmark --task hard --tag run0 --push-to-db ...
```

它绕过 `benchmark.py` 第 541 行的硬编码：

```540:542:social_env/sotopia/cli/benchmark/benchmark.py
    if task == "hard":
        hard_envs = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").environments
        agent_index = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").agent_index
```

### 2.4 怎么扩量

直接在 `scripts/generate_from_scratch.py` 里追加 `AGENT_ARCHETYPES` / `ENVIRONMENT_ARCHETYPES` 字典——每条新数据不到 30 行 Python，review 友好，不烧 token。重跑脚本即可。

---

## 2.5 路径 A2：完全从零（LLM 批量生成）

`scripts/generate_from_scratch_with_llm.py` 是路径 A1 的 LLM 版孪生兄弟：
**只换"哪儿来 agent / env"** 这一段（手写字面量 → LLM 输出），落库逻辑完全复用 A1。

### 2.5.1 它和 A1、`generate_v2_with_llm.py` 的区别

| 维度 | A1：`generate_from_scratch.py` | **A2：`generate_from_scratch_with_llm.py`** | C：`generate_v2_with_llm.py` |
|---|---|---|---|
| Agent 来源 | 手写 `AGENT_ARCHETYPES` | **LLM 调 `agenerate(AgentProfile)`** | 不造 agent |
| Env 来源 | 手写 `ENVIRONMENT_ARCHETYPES` | **LLM 调 `agenerate_env_profile`** | LLM 调 `agenerate_env_profile` |
| 是否造 Relationship / Combo / EnvironmentList | ✅ 全造 | ✅ 全造（直接 import A1 的 `save_relationships` / `save_combos` / `save_environment_list`） | ❌ 只造 env |
| 是否支持 V2 | `--with-v2` | `--with-v2`（调 A1 的 `maybe_build_v2`） | 总是写 V2 |
| 主题驱动 | 无 | **`--theme` 关键词派生 agent_briefs + scenario_briefs** | 按 `--scenario-type` 选预设 |
| 失败容忍 | 无（确定性） | ✅ async gather + 单条失败跳过，剩下的继续 | ✅ |
| 适用场景 | PoC / 回归基准 / review 友好 | **要"一键造一批主题自洽的可跑数据"**：换 theme 就换世界观 | 给已有 V1 加更多 V2 env |

一句话定位：**A2 = "用 LLM 替手写"的 A1**——同样是从零起步，同样落到 `~/.sotopia/data/`，同样能直接 `sotopia benchmark`，但 agent / env 由 theme 驱动 LLM 生成。

### 2.5.2 一条命令拿到完整数据

```bash
# 默认 theme，造 6 角色 + 4 场景，写 V2，调用 gpt-4o-mini
SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python \
    scripts/generate_from_scratch_with_llm.py --clean --with-v2

# 指定主题 + 规模 + 模型
SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python \
    scripts/generate_from_scratch_with_llm.py \
    --theme "AI startup founders facing a market crash" \
    --n-agents 8 --n-envs 5 --combos-per-env 3 \
    --model gpt-4o-mini --tag llm_run_v1 --with-v2
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--clean` | off | 先 `rm -rf ~/.sotopia/data/`，从零开始（注意：会清掉所有已有 EpisodeLog） |
| `--theme` | `default` | 命中 `THEME_PRESETS` 用预设 brief；否则按自由文本派生 6 个 agent + 4 个 scenario brief |
| `--n-agents N` | 6 | LLM 要造的 AgentProfile 数量；不足 brief 池会循环填充 |
| `--n-envs M` | 4 | LLM 要造的 EnvironmentProfile 数量 |
| `--combos-per-env K` | 2 | 每个 env 随机配 K 对 agent（套 A1 的 `save_combos`） |
| `--model` | `gpt-4o-mini` | 调用的 LLM；`agenerate` / `agenerate_env_profile` 内部走 LangChain |
| `--temperature` | 0.8 | 给一点多样性；研究复现可调到 0.2 |
| `--concurrency` | 4 | 并发 LLM 调用上限（信号量），防触发 RPM |
| `--list-name` | `scratch_llm_env_set` | `EnvironmentList.name`（不是 pk） |
| `--override-hard-list` | off | 把 `EnvironmentList.pk` 设为官方 hard ULID，让 `--task hard` 直接命中 |
| `--with-v2` | off | 同时落 V2 数据（`AgentProfileV2` / `EnvironmentProfileV2` / `EventScript` / `Contract` / `SystemStateSnapshot`） |
| `--tag` | `scratch_llm_v1` | 写入 `AgentProfile.tag`；后续 benchmark 跑出来的 `EpisodeLog.tag` 用 `<tag>_run0` 配套 |
| `--seed` | 42 | combo 随机配对的种子，`--seed` 同 → 同样的配对 |
| `--dry-run` | off | 只打印派生出来的 brief，不调 LLM、不落库；改 theme 时先用这个看预览 |

### 2.5.3 内置 theme 预设

脚本里写了四个预设主题（命中 `THEME_PRESETS` 字典就用其中预先校对过的 brief）：

| theme key | 主线 | 内置 agent / scenario brief 数 |
|---|---|---|
| `default` | 经济压力下的多元社会人 | 8 / 6 |
| `ai_startup_crash` | AI 创业公司 Series A 失败 | 6 / 5 |
| `drought_village` | 干旱村庄水井分配 | 5 / 3 |
| `post_layoff_white_collar` | 白领失业潮（AI 替代） | 5 / 3 |

不在表里的 theme（任意自由文本，例如 `"国产开源大模型团队抢人"`），会走 `derive_briefs` 通用模板自动派生 6 个 agent + 4 个 scenario brief。**先 `--dry-run` 看一下这些 brief 是否切题再开烧**。

### 2.5.4 LLM 输出怎么变成 AgentProfile

`generate_agent` 用的是 `agenerate(... PydanticOutputParser(AgentProfile))`，prompt 模板（`AGENT_TEMPLATE`）已硬编码了三处约束：

- `big_five` 强制 5 个维度的 `high|medium|low` 字符串
- `moral_values` / `schwartz_personal_values` 限定枚举
- `secret` 必须是一条具体的、不会随便分享的私事（保证场景张力）

`generate_env` 直接复用官方 `agenerate_env_profile`——它的 prompt 已被 sotopia 团队调过，能稳定吐出 `scenario` + `agent_goals` 双字段。

### 2.5.5 落库流程（与 A1 完全一致）

`main()` 的关键 6 步：

1. `wipe_local_data()`（仅 `--clean`）—— 来自 A1
2. `_generate_all()` —— `asyncio.gather` 并发 LLM 调用，失败的 brief 直接跳过
3. `save_llm_agents()` —— 给每条加 `tag`、补默认 `first_name/last_name`，再 `.save()`
4. `save_llm_envs()` —— 给每条加 `source="llm_<model>_<theme>"` 和 `codename`
5. `save_relationships() + save_combos() + save_environment_list()` —— **直接 import A1 的同名函数**，零修改复用
6. `maybe_build_v2()`（仅 `--with-v2`）—— 调用 `benchmark_v2_data_models.py` 那四个工厂函数

**所以 A2 写出来的 `~/.sotopia/data/` 目录布局跟 A1 一模一样**，老 `sotopia benchmark` 完全不感知差异。

### 2.5.6 立即跑 benchmark 验证

```bash
SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \
    --models gpt-4o-mini --partner-model gpt-4o-mini \
    --evaluator-model gpt-4o-mini \
    --task scratch --tag scratch_llm_v1_run0 --push-to-db --batch-size 4
```

注意点：

- `--task scratch`：和 A1 一样，任意非 `hard/cooperative/competitive` 字符串都会进 `else` 分支，跑全部刚造的 combo
- 想用 `--task hard`：重造时加 `--override-hard-list`，让 `EnvironmentList.pk = 01HAK34YPB1H1RWXQDASDKHSNS`
- `--push-to-db`：**必加**，否则 episode 不落盘（参见 §5.2）

### 2.5.7 已知坑 / 调试技巧

| 现象 | 排查方法 |
|---|---|
| `OPENAI_API_KEY 未设置` | 脚本入口前会兜底加载 `.env`；如果还报，确认 `.env` 在 `social_env/` 根目录、且包含 `OPENAI_API_KEY=...` |
| `[skip] agent #i failed: ...` 多 | 多半是 `PydanticOutputParser` 解析失败：模型 echo 了 schema、或者输出多余的 markdown 包裹。换更强的 `--model gpt-4o`、或把 `--temperature` 降到 0.2 |
| 跑出来场景跟 theme 无关 | 自由 theme 走的是通用模板，对模型理解能力要求高；先 `--dry-run` 检查 brief，必要时自己往 `THEME_PRESETS` 里加一项 |
| LLM 限流 (RPM) | 调小 `--concurrency` 到 1~2；或换 base_url 到企业 endpoint |
| `combos / list` 数量看着不对 | 先确认 `save_relationships/save_combos/save_environment_list` 用的是 A1 同款函数；agent 数 < 2 时脚本会主动 `[err] 至少需要 2 个 agent`，combo/list 会跳过 |
| 想保留已有数据，只追加新 LLM 数据 | **去掉 `--clean`**，但要换一个 `--seed` 和 `--list-name`，避免 combo 配对碰撞、或 EnvironmentList 同名覆盖 |

### 2.5.8 怎么扩主题

往 `THEME_PRESETS` 字典里追加一项即可：

```python
"my_theme_key": {
    "agent_briefs": [
        "...",
        "...",
    ],
    "scenario_briefs": [
        "...",
    ],
},
```

每条 brief ≈ 1 句话，LLM 生成时质量比通用模板高一档。

---

## 3. 路径 B：从 export 导入

适用于你想直接复用 Sotopia 官方公开数据（40 个 agent、800+ env、4800+ combo）。

```bash
SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python scripts/setup_local_data.py
```

跑完检查：

```bash
ls ~/.sotopia/data/AgentProfile | wc -l       # 期望 ~40
ls ~/.sotopia/data/EnvironmentProfile | wc -l # 期望几百
```

可选的 fallback 模式（`export/` 缺表时从 Redis 兜底）：

```bash
SOTOPIA_STORAGE_BACKEND=local \
    python scripts/setup_local_data.py --fallback-redis --redis-url redis://localhost:6379
```

---

## 4. 路径 C：升级为 V2

V1 数据落库之后（无论来自路径 A 还是 B），可以叠加 V2。

### 4.1 无 LLM 升级（秒级，确定性）

读现有 `AgentProfile` / `EnvironmentProfile`，按规则映射出 V2 字段（`risk_preference` / `role_type` / `scenario_type` / `max_days` 等），并生成 EventScript / Contract / SystemStateSnapshot 样例：

```bash
SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python scripts/generate_v2_seed.py \
    --n-agents 20 --n-envs 5 --tag bench_v2_seed_v1
```

> 路径 A 的 `--with-v2` 内部就是调这套逻辑，所以两条路只需选其一。

### 4.2 用 LLM 扩 V2 环境

按 `scenario_type` 用预设 inspiration prompt 调 `agenerate_env_profile`，批量生成新 V2 环境：

```bash
for s in negotiation investment commons; do
    SOTOPIA_STORAGE_BACKEND=local \
        /home/yphao/.conda/envs/social_env/bin/python scripts/generate_v2_with_llm.py \
        --scenario-type $s --n 3 --model gpt-4o-mini
done
```

会同时写老 `EnvironmentProfile`（兼容老流水线）和 V2 `EnvironmentProfileV2`。

### 4.3 验证落库

```bash
for d in AgentProfileV2 EnvironmentProfileV2 EventScript Contract SystemStateSnapshot; do
    echo "$d: $(ls ~/.sotopia/data/$d 2>/dev/null | wc -l) files"
done
```

---

## 5. 关键设计点 & 已知坑

### 5.1 关键设计点

| # | 设计点 | 说明 |
|---|---|---|
| 1 | **数据构造层入口** | `social_env/sotopia/benchmark_v2_data_models.py` 的四个工厂函数：`upgrade_agent_profile` / `upgrade_environment_profile` / `make_initial_state_snapshot` / `make_event_script_from_dict`。**驱动逻辑写在 `scripts/`，模型文件保持纯结构定义** |
| 2 | **V1 与 V2 共存** | `AgentProfileV2` / `EnvironmentProfileV2` 是独立目录，老 `sotopia benchmark` 完全不感知 V2，互不干扰 |
| 3 | **EventScript 延迟绑定** | 脚本里造的 EventScript 用 `<target_pk>` 占位，运行时由 `EventEngine`（待实现，见 `BENCHMARK_DESIGN_zh.md §B`）拿 episode 上下文解析 |
| 4 | **Contract 模板模式** | `episode_pk=""` 表示模板；运行时 `ActionDispatcher` 克隆并填上真 episode_pk |
| 5 | **SystemStateSnapshot 是 day-level** | 脚本只造 day=0；day=1..max_days 的快照由 `SocialSystemEnv` 在 `end_of_day` 自动写入 |
| 6 | **角色卡关键字段** | `personality_and_values` 控制 LLM 口吻；`secret` 制造可被探查/利用的私有信息——手写时务必两者都有 |
| 7 | **agent_goals 模板格式** | `<extra_info>…</extra_info> <主目标>`，少了 `<extra_info>` 标签 LLM 会忽略私有信息 |
| 8 | **EnvironmentList.agent_index 双向展开** | 每对 (env, combo) 展开成 `agent_index="0"` 和 `"1"` 两条，让测试模型双向各演一次，评测更对称 |

### 5.2 已知坑

| 现象 | 根因 | 解决 |
|---|---|---|
| benchmark 跑完 `No episodes found` | 默认 `push_to_db=False`，episode 只在内存里跑完 | 命令必加 `--push-to-db` |
| 跑完文件被删 | evaluator 失败导致 `rewards[0]` 退化为 `float`，原 `run_async_benchmark_in_batch` 收尾会删掉这些 episode | 已改成 `__bad_eval` 后缀 quarantine（环境变量 `SOTOPIA_QUARANTINE_BAD_EVAL=0` 可恢复老语义） |
| evaluator 返回 JSON Schema 当数据 | `gpt-5-mini` 对 `dict[str, T]` schema 的退化模式 | (1) 改用 `--evaluator-model gpt-4o-mini`；(2) `EpisodeLLMEvaluator.__acall__` 已加 3 次重试 + schema-echo 检测 |
| 文件名是 ULID 而非 tag | tag 不唯一、pk 唯一；本地后端用 pk 当文件名 | 跑 `scripts/symlink_by_tag.py` 在 `~/.sotopia/data_by_tag/<tag>/` 建软链接 |
| `~/.sotopia/data/` 与 Redis 数据不一致 | `SOTOPIA_STORAGE_BACKEND` 默认 `redis`，没显式 export 时跑 sotopia 会读 Redis | 跑 `scripts/which_backend.py` 同时盘 local + Redis；统一在命令前置 `SOTOPIA_STORAGE_BACKEND=local` |

---

## 6. 推荐工作流

按使用规模选一条：

### 6.1 PoC / 调试 / 单元测试

```bash
python scripts/generate_from_scratch.py --clean --with-v2
sotopia benchmark --task scratch --tag run0 --push-to-db ...
```

完全确定性、可重现，秒级建数据。

### 6.2 实验：手写骨干 + LLM 扩枝（生产推荐）

```bash
# Step 1: 骨干（用于回归基准）
python scripts/generate_from_scratch.py --clean --tag base

# Step 2: LLM 按场景扩 negotiation 类
python scripts/generate_v2_with_llm.py --scenario-type negotiation --n 20 --model gpt-4o-mini

# Step 3: 重新生成 combo & list（不删旧 agent/env）
python scripts/generate_from_scratch.py --combos-per-env 4 --tag mixed
```

骨干保证回归可比性，LLM 扩枝带来覆盖率，两者用 tag 区分。

### 6.2.1 一键 LLM 主题包（A2，最快出新世界观）

需要"换个主题就换一套 agent + env + combo + list + V2"的时候用：

```bash
# Step 1: 先 dry-run 看 brief 是否切题
SOTOPIA_STORAGE_BACKEND=local python scripts/generate_from_scratch_with_llm.py \
    --theme "AI startup founders facing a market crash" --dry-run

# Step 2: 真造（可能烧几分钟 + 几美分 token）
SOTOPIA_STORAGE_BACKEND=local python scripts/generate_from_scratch_with_llm.py \
    --clean --with-v2 \
    --theme "AI startup founders facing a market crash" \
    --n-agents 8 --n-envs 5 --combos-per-env 3 \
    --model gpt-4o-mini --tag ai_crash_v1

# Step 3: 直接跑 benchmark
SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \
    --models gpt-4o-mini --partner-model gpt-4o-mini \
    --evaluator-model gpt-4o-mini --task scratch \
    --tag ai_crash_v1_run0 --push-to-db --batch-size 4
```

适合：写论文需要多个主题对照实验、demo 给同事换不同世界观、快速覆盖 LLM 行为分布。

### 6.3 复用官方公开数据

```bash
python scripts/setup_local_data.py        # 路径 B
python scripts/generate_v2_seed.py        # 路径 C 无 LLM 升级
```

适合直接对齐 Sotopia 论文 setting。

---

## 7. 验证 & 检查工具

| 工具 | 用途 |
|---|---|
| `scripts/which_backend.py` | 确认当前后端是 redis 还是 local，并统计两边数据量 |
| `scripts/symlink_by_tag.py` | 在 `~/.sotopia/data_by_tag/<tag>/` 建软链接，按 tag 浏览 EpisodeLog |
| `scripts/diagnose_episodes.py` | Redis 后端时排查"为什么 find 不到 episode" |
| `scripts/verify_save.py` | 直接调 `EpisodeLog.save()` 暴露被静默吞掉的异常 |

常用命令：

```bash
# 看每类有多少条
for d in ~/.sotopia/data/*/; do
    echo "$(basename $d): $(ls $d | wc -l)"
done

# 抽看一条 EpisodeLog
cat ~/.sotopia/data/EpisodeLog/$(ls ~/.sotopia/data/EpisodeLog | head -1) | jq

# 按 tag 找 EpisodeLog（Python API）
SOTOPIA_STORAGE_BACKEND=local python -c "
from sotopia.database import EpisodeLog
eps = EpisodeLog.find(EpisodeLog.tag == 'run0').all()
print(f'tag=run0: {len(eps)} episodes')
"

# 导出 V2 全套数据（分发给同事）
tar czf v2_seed.tar.gz -C ~/.sotopia \
    data/AgentProfileV2 data/EnvironmentProfileV2 \
    data/EventScript data/Contract data/SystemStateSnapshot
```

---

## 8. 下一步

数据层已完整。再往下要落 V2 运行时（让 EventScript 真的在 `end_of_day` 触发、Contract 真的进入生命周期、SystemStateSnapshot 自动累积），需要写：

- `sotopia/envs/social_system_env.py` —— 继承 `ParallelSotopiaEnv`，按 day/intra-day-step 双层时间推进
- `sotopia/runtime/event_engine.py` —— 解析 `EventScript.effects` 的 mini-DSL 并改 `SystemState`
- `sotopia/runtime/action_dispatcher.py` —— 处理 V2 的结构化 action（含 `propose_contract` / `accept_contract` 等）

详见 `BENCHMARK_DESIGN_zh.md §B/C/D`。

## 