#!/usr/bin/env python
"""用大模型生成 **长期谈判题库** — V2 扩展版（合同经济学确定性化）。

与原始 ``generate_long_term_negotiation_llm.py`` 的差异：
- 使用 ``build_extended_negotiation_game_metadata_bundle``（V2 outcome rule）
- V2 合作合同：合成时确定 predetermined_payouts
- V2 买卖合同：合成时确定 reference_price + cost_price
- 新增 --initial-resources-config 指定场景专属初始资金
- 其余参数、流程与原脚本完全一致

用法::

    cd social_env
    SOTOPIA_STORAGE_BACKEND=local python scripts/generate_long_term_negotiation_llm_extended.py \\
        --model custom/deepseek-v4-pro@https://api.deepseek.com \\
        --n 6 --mode-counts firms2=3,firms3=3 --concurrency 8 \\
        --timeline-labels D2 --tag ltr_multi_firm_llm_v1 \\
        --agent-profile-out long_term_negotiation_llm_agent_profiles.multi_firm.json

    # 指定场景专属初始资金
    python scripts/generate_long_term_negotiation_llm_extended.py \\
        --model gpt-4o-mini --n 3 --modes firms2 \\
        --initial-resources-config settings/long_term_negotiation/initial_resources_example.json \\
        --tag ltr_v2_custom_funds
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "local")

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from sotopia.database import EnvironmentProfile
from sotopia.settings.long_term_negotiation import (
    NEGOTIATION_LINEUP_FIRMS_ONLY,
    SESSION_FIRMS_ONLY_ROLE_ORDER,
    SESSION_SPEAKER_ROLE_ORDER,
    NegotiationTimelineParams,
)
from sotopia.settings.long_term_negotiation.extended_scenario_loader import (
    build_extended_negotiation_game_metadata_bundle,
)

# Import helpers from the original script
from generate_long_term_negotiation_llm import (
    DEFAULT_COMPANY_LLM_ROLES,
    LOCAL_DATA_DIR,
    SCENARIO_PROMPT_BANK,
    SCENARIO_PROMPT_GUIDE,
    _LLM_MODE_TO_LINEUP_N,
    _build_mixed_inspirations,
    _lineup_n_for_mode,
    _roles_for_mode,
    _validate_scene_coverage_capacity,
    bilateral_timeline_presets_lite,
    filter_presets_lite,
    firms3_goal_padding,
    firms4_goal_padding,
    generate_one_llm_profile,
    modes_cycle_from_arg,
    parse_mode_counts,
)

ltr = sys.modules["generate_long_term_negotiation_llm"]


def _load_initial_resources_config(path: str | None) -> dict[str, dict[str, float]] | None:
    """从 JSON 文件加载场景专属初始资金配置。"""
    if not path:
        return None
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        print(f"[warn] initial_resources_config not found: {config_path}")
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] failed to parse initial_resources_config: {e}")
        return None
    if not isinstance(data, dict):
        return None
    result: dict[str, dict[str, float]] = {}
    for role, res in data.items():
        if isinstance(res, dict):
            result[str(role)] = {
                str(k): float(v) for k, v in res.items() if isinstance(v, (int, float))
            }
    return result if result else None


async def main_async_extended(
    args: argparse.Namespace,
    load_mod: Any,
    initial_resources_config: dict[str, dict[str, float]] | None = None,
) -> int:
    """与原始 ``main_async`` 功能等价，仅替换 outcome rule 构建为 V2。"""
    # Reuse the original main_async logic but patch the bundle function
    import generate_long_term_negotiation_llm as orig

    # Monkey-patch build_negotiation_game_metadata_bundle in the original module's namespace
    # so that any indirect calls also get V2 behavior
    orig_build = orig.build_negotiation_game_metadata_bundle
    try:
        # Replace in the original module for the duration of this call
        orig.build_negotiation_game_metadata_bundle = build_extended_negotiation_game_metadata_bundle

        # Also patch the reference in scenario_loader if the original imports from there
        import sotopia.settings.long_term_negotiation.scenario_loader as sl
        orig_sl_build = sl.build_negotiation_game_metadata_bundle
        sl.build_negotiation_game_metadata_bundle = build_extended_negotiation_game_metadata_bundle

        result = await orig.main_async(args, load_mod)

        # If initial_resources_config is provided, post-process the saved environments
        if initial_resources_config:
            print(
                f"[v2] applying initial_resources_config to saved environments "
                f"({len(initial_resources_config)} roles)"
            )
            # Reload manifest to find saved environments
            manifest_path = LOCAL_DATA_DIR / str(args.manifest_name)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for env_info in manifest.get("environments", []):
                    pk = env_info.get("pk")
                    if not pk:
                        continue
                    try:
                        env = EnvironmentProfile.get(pk)
                    except Exception:
                        continue
                    gm = env.game_metadata if isinstance(env.game_metadata, dict) else {}
                    gm["initial_resources_by_role"] = initial_resources_config
                    env.game_metadata = gm
                    env.save()
                    print(f"  [v2] updated EnvironmentProfile pk={pk[:8]}...")

        return result
    finally:
        orig.build_negotiation_game_metadata_bundle = orig_build
        sl.build_negotiation_game_metadata_bundle = orig_sl_build


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", default="gpt-4o-mini",
        help="agenerate_env_profile model_name（LiteLLM 路由键）",
    )
    ap.add_argument(
        "--n", type=int, default=3,
        help="生成条数上限（会与 inspiration 列表截断对齐）",
    )
    ap.add_argument(
        "--modes", default="firms3",
        help="逗号分隔、保序轮转：仅 firms2 / firms3 / firms4（均为 firms_only）",
    )
    ap.add_argument(
        "--mode-counts", default="",
        help="按模式精确指定生成条数：MODE=N[,MODE=N...]",
    )
    ap.add_argument(
        "--scenario-mix",
        default="business_coopetition,wet_market_competition,resource_scheduling_management",
        help="逗号分隔的场景类型轮转",
    )
    ap.add_argument(
        "--timeline-labels", default="",
        help="逗号分隔，仅使用这些时间轴预设标签（如 D6,D12）；空表示全部",
    )
    ap.add_argument(
        "--requirements", default="",
        help="自由文本，写入 manifest generation_spec.requirements_notes",
    )
    ap.add_argument("--tag", default="ltr_llm_benchmark")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument(
        "--inspiration", action="append", dest="inspiration", default=None,
        help="自定义灵感句（可多传）",
    )
    ap.add_argument(
        "--manifest-name", default="long_term_negotiation_llm_manifest.json",
        help="manifest 文件名（写入 ~/.sotopia/data）",
    )
    ap.add_argument(
        "--agent-profile-model", default=None,
        help="生成 AgentProfile 用的 LLM；不传则复用 --model",
    )
    ap.add_argument(
        "--agent-profile-out",
        default="long_term_negotiation_llm_agent_profiles.multi_firm.json",
        help="按环境汇总的 AgentProfile JSON",
    )
    ap.add_argument(
        "--agent-profiles-all-llm", action="store_true",
        help="兼容旧 CLI：仅 firm_a..firm_d 可走 LLM",
    )
    ap.add_argument(
        "--legacy-agent-profiles", action="store_true",
        help="不调 LLM，沿用常量画像",
    )
    ap.add_argument(
        "--deterministic-outcome-rule", action="store_true",
        help="V2 默认已确定性；此参数保留兼容",
    )
    ap.add_argument(
        "--diversity-manifest", default="",
        help="上一轮 manifest JSON，用于增量多样性",
    )
    ap.add_argument(
        "--diversity-model", default="",
        help="摘要步骤使用的模型",
    )
    ap.add_argument(
        "--diversity-max-per-scene", type=int, default=16,
        help="每个 scene_type 参与摘要的最多历史环境条数",
    )
    ap.add_argument(
        "--diversity-digest-out", default="",
        help="结构化摘要写入路径（JSON）",
    )
    ap.add_argument(
        "--incremental-diversity", action="store_true",
        help="按 scene_type 串行生成，批内不重复",
    )
    # V2-specific arguments
    ap.add_argument(
        "--initial-resources-config", default="",
        help="场景专属初始资金 JSON 文件路径（写入 game_metadata.initial_resources_by_role）",
    )
    args = ap.parse_args()

    initial_resources_config = _load_initial_resources_config(
        args.initial_resources_config or None
    )
    if initial_resources_config:
        print(
            f"[v2] loaded initial_resources_config with roles: "
            f"{list(initial_resources_config.keys())}"
        )

    load_mod = ltr._load_handwritten_generator() if hasattr(ltr, '_load_handwritten_generator') else None
    return asyncio.run(main_async_extended(args, load_mod, initial_resources_config))


if __name__ == "__main__":
    raise SystemExit(main())
