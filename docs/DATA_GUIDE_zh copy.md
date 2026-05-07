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





















# 数据

两个脚本都通过 dry-run。下面是**完整的造数据 SOP**（一段段复制即可）：

## 完整 SOP（推荐这个顺序）

### Step 0：把底料补足（只需一次）

```bash
cd /mnt/userdata/yphao/FC/game_MAS/social_env

SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python scripts/setup_local_data.py
```

跑完 `~/.sotopia/data/AgentProfile/` 应该有 ~40 个文件，`EnvironmentProfile/` 有几百个。

### Step 1：跑 V2 升级脚本（无 LLM，秒级）

```bash
SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python scripts/generate_v2_seed.py \
    --n-agents 20 --n-envs 5 --tag bench_v2_seed_v1
```

### Step 2：用 LLM 扩 V2 环境（按 scenario_type）

```bash
for s in negotiation investment commons; do
  SOTOPIA_STORAGE_BACKEND=local \
      /home/yphao/.conda/envs/social_env/bin/python scripts/generate_v2_with_llm.py \
      --scenario-type $s --n 3 --model gpt-4o-mini
done
```

### Step 3：验证落库

```bash
for d in AgentProfileV2 EnvironmentProfileV2 EventScript Contract SystemStateSnapshot; do
  echo "$d: $(ls ~/.sotopia/data/$d 2>/dev/null | wc -l) files"
done

cat ~/.sotopia/data/EventScript/$(ls ~/.sotopia/data/EventScript | head -1)
```

---

## 几个关键设计点（看完会少踩坑）

1. **从哪开始？** → 入口在 `social_env/sotopia/benchmark_v2_data_models.py`，里面四个工厂函数（`upgrade_agent_profile` / `upgrade_environment_profile` / `make_initial_state_snapshot` / `make_event_script_from_dict`）就是"造数据"的对外 API。**驱动脚本应该写在 `scripts/` 而不是动模型文件**——模型文件保持纯结构定义。

2. **V1 与 V2 共存**。`AgentProfileV2` 与老 `AgentProfile` 是两个独立目录，老 `sotopia benchmark` 命令完全不感知 V2。这是为了让 V1 流水线（你已经跑通的）和 V2 流水线（你正在搭）互不干扰。

3. **EventScript 的"延迟绑定"**。脚本里造的 EventScript 用 `<target_pk>` 这种占位字符串而不是真 pk——运行时 EventEngine 拿到当前 episode 的 agent 列表才解析。这是 `BENCHMARK_DESIGN_zh.md §B` 里 EventEngine 要做的事；现在先把数据结构造出来。

4. **Contract 的"模板模式"**。`episode_pk=""` 的合约是"模板"，跑 episode 时由 ActionDispatcher clone 一份并填上真 episode_pk。这样数据可以预先批量造，运行时不需要 LLM 实时生成合约结构。

5. **SystemStateSnapshot 是 day-level**。脚本现在只造 day=0；day=1..max_days 的快照是 `SocialSystemEnv` 在 `end_of_day` 自动写的，不用预先造。

6. **千万别在 `~/.sotopia/data/<Class>/` 下混用 V1 与 V2 同名实体**。两个 V2 类目录名分别是 `AgentProfileV2` / `EnvironmentProfileV2`（即 `__class__.__name__`），不会和 `AgentProfile` 冲突，已经隔离好。

---

## 常见后续动作

- 造完之后想**人工 review** EventScript？直接 `cat ~/.sotopia/data/EventScript/*.json | jq .` 即可。
- 想**导出分发**？`tar czf v2_seed.tar.gz -C ~/.sotopia data/AgentProfileV2 data/EnvironmentProfileV2 data/EventScript data/Contract data/SystemStateSnapshot`。
- 想接 V2 跑通 episode？还需要写 `SocialSystemEnv` 与 `EventEngine`，那是 `BENCHMARK_DESIGN_zh.md §B/C` 的任务，**数据已经准备好了**，等运行时层补齐就能直接用。

如果你下一步想做的是"接 EventScript 进运行时让它真的在 end_of_day 触发"，告诉我，我们再写 `EventEngine` 的最小实现。1 秒搞定，`~/.sotopia/data/` 下凭空出现 10 个目录的完整数据。下面是整个"从零造数据"的核心思路与可扩展接口。

## 核心思路：把"读数据"换成"造数据"

`scripts/setup_local_data.py` 是 `export/*.json → 本地后端`，本质上**还是从 Redis 出来的数据**。  
`scripts/generate_from_scratch.py` 改成 `Python 代码 → 本地后端`，**完全自给自足**。

| 对比 | 旧流水线 | 新流水线 |
|---|---|---|
| 数据源 | `dump.rdb` / `export/*.json`（间接来自 Sotopia 公开数据集） | 脚本里 `AGENT_ARCHETYPES` / `ENVIRONMENT_ARCHETYPES` 字面量 |
| 依赖 | Docker + Redis + RediSearch + 网络 | 只需 Python |
| 可读性 | 上千条 ULID，调试时根本找不到对应内容 | 几个命名清晰的字典，代码 review 即数据 review |
| 修改成本 | 改 LLM prompt + 重跑 LLM + 落库 | 改字典即可 |
| 适合规模 | 千条以上 | 几十~几百条；超过这个量再用 LLM 扩 |

## 一条命令拿到完整数据

```bash
cd /mnt/userdata/yphao/FC/game_MAS/social_env

SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python \
    scripts/generate_from_scratch.py --clean --with-v2
```

`--clean` 表示先把 `~/.sotopia/data/` 删空再造，确保**干净的从零开始**。

跑完拿到的全套数据：

| 类 | 数量 | 内容 |
|---|---:|---|
| `AgentProfile` | 8 | 手写的 8 个角色（Mia 工程师、William 商人、Sophia 护士、Lucas 议员…） |
| `EnvironmentProfile` | 6 | 手写场景（修洗碗机、股权分配、共用水井、记者揭丑闻、合租欠租、政策预算） |
| `RelationshipProfile` | 28 | 任意两两 stranger 关系（C(8,2)=28） |
| `EnvAgentComboStorage` | 12 | 每个 env 随机配 2 对 agent |
| `EnvironmentList` | 1 | `name="scratch_env_set"`，含 12 个 (env, agent_index) 项 |
| `AgentProfileV2` | 8 | V2 升级版（带资源/声誉/角色/风险偏好） |
| `EnvironmentProfileV2` | 6 | V2 升级版（带 scenario_type/max_days） |
| `EventScript` | 2 | 第 1 天供应链中断、第 2 天加息 |
| `Contract` | 1 | 借贷模板 |
| `SystemStateSnapshot` | 1 | day=0 初始状态 |

## 立即跑 benchmark 验证

```bash
SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \
    --models gpt-4o-mini --partner-model gpt-4o-mini \
    --evaluator-model gpt-4o-mini \
    --task scratch --tag run0
```

`--task scratch`（任意非 `hard/cooperative/competitive` 的字符串）会进入 `_benchmark_impl` 的 `else` 分支，使用所有 `EnvAgentComboStorage`，正好覆盖你刚造的 12 条。

如果你坚持要用 `--task hard`：

```bash
# 重造时让 EnvironmentList 顶替官方 hard 名单的 pk
SOTOPIA_STORAGE_BACKEND=local \
    /home/yphao/.conda/envs/social_env/bin/python \
    scripts/generate_from_scratch.py --clean --override-hard-list

SOTOPIA_STORAGE_BACKEND=local sotopia benchmark \
    --models gpt-4o-mini --partner-model gpt-4o-mini \
    --evaluator-model gpt-4o-mini --task hard --tag run0
```

`--override-hard-list` 把新建 `EnvironmentList.pk` 强制设成 `01HAK34YPB1H1RWXQDASDKHSNS`，绕过 benchmark.py 第 541 行的硬编码：

```540:545:social_env/sotopia/cli/benchmark/benchmark.py
    if task == "hard":
        hard_envs = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").environments
        agent_index = EnvironmentList.get("01HAK34YPB1H1RWXQDASDKHSNS").agent_index
```

## 怎么扩量（按需选一种）

### A. 加更多手写素材（推荐，最稳）

直接在脚本里追加 `AGENT_ARCHETYPES` / `ENVIRONMENT_ARCHETYPES` 的字典即可。每条新数据不到 30 行 Python，review 友好，不烧 token，跑完立刻可用。

### B. 用 LLM 大批量生成（已有现成入口）

之前给你的 `scripts/generate_v2_with_llm.py` 已经在用 `agenerate_env_profile`。你可以把它改成同时也写 V1 老 `EnvironmentProfile`（已经在做），跑完之后再跑一次 `generate_from_scratch.py --no-clean`（不加 `--clean`）追加更多 combo/list 即可。

### C. 混合：手写骨干 + LLM 扩枝

```bash
# 1) 先建骨干（确定性，用于回归测试）
python scripts/generate_from_scratch.py --clean --tag base

# 2) 再用 LLM 扩 negotiation 类场景
python scripts/generate_v2_with_llm.py --scenario-type negotiation --n 20 --model gpt-4o-mini

# 3) 重新生成 combo & list（不删旧 agent/env）
python scripts/generate_from_scratch.py --combos-per-env 4 --tag mixed
```

## 设计上几个关键决定

1. **角色卡的关键字段是 `personality_and_values` + `secret`**——前者控制 LLM 演的人物口吻，后者制造可被对手探到/利用的私有信息。脚本里的 8 个 archetype 都刻意造了"行为可预测但又有戏剧张力"的 secret。

2. **agent_goals 用 V1 模板格式**：`<extra_info>...</extra_info> <主目标>`。这是 sotopia 内置 prompt 模板期待的结构，少了它 LLM 会忽略私有信息。

3. **每个 env 配 `combos_per_env` 对 agent**，并在 `EnvironmentList.agent_index` 里**对每对都展开成 0/1 两条**——这样跑 benchmark 时同一个对手关系会被双向各跑一遍，让评测更对称（这是 sotopia 官方 cooperative/competitive 分支的做法）。

4. **不调 LLM**意味着可以**写到测试里**：你可以把这个脚本作为 `pytest` 的 fixture，每次跑测试前用临时目录重新造一份，完全可重现。

5. **V1 / V2 同进同出**：V2 是 V1 的"超集叠加"，所以一个驱动脚本同时落两套数据，不会有不一致问题。一段时间后想完全切到 V2，把生成 V1 的部分改成可选（`--v1` 默认 False）即可。

## 后续可能要的两条命令

```bash
# 看刚造的某个角色
cat ~/.sotopia/data/AgentProfile/$(ls ~/.sotopia/data/AgentProfile | head -1) | jq

# 看刚造的某个场景
cat ~/.sotopia/data/EnvironmentProfile/$(ls ~/.sotopia/data/EnvironmentProfile | head -1) | jq
```

要不要再写一份 `scripts/generate_from_scratch_with_llm.py`，让你只给一句"行业关键词"就能让 LLM 直接出整套 V1+V2 数据？或者下一步直接写 `EventEngine` 让 EventScript 真的在 `end_of_day` 触发？