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
