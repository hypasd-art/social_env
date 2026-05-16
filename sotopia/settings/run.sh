#!/usr/bin/env bash
# ============================================================
# 长期谈判 benchmark 一键脚本：合成 → 评测
#
# V1 + V2 已合并：
#   - 合成阶段：generate_long_term_negotiation_llm_extended.py
#     * LLM 生成 scenario / agent_goals / agent_profiles（含差异化
#       risk_preference、initial_reputation、resource_modifiers）
#     * 确定性种子 → market_state + psych_variables + V2 outcome_rule
#     * 所有差异化数据写入 game_metadata → DB
#   - 评测阶段：env.py / negotiation_metrics.py 自动检测 V1/V2 规则
#     * V2 cooperation：读取 predetermined_payouts
#     * V2 buy_sell：读取 reference_price + cost_price
#     * psych_variables 从 DB 传入 LongTermNegotiationEnv
# ============================================================
set -euo pipefail

# --------------- 环境变量 ---------------
export OPENAI_API_BASE=https://api.deepseek.com
export CUSTOM_API_KEY=sk-4b5a28b9dbd345ef9cfd67d6fb34e9e6
export SOTOPIA_STORAGE_BACKEND=local
export SOTOPIA_MAX_RENDERED_USER_CHARS=0

cd "$(dirname "$0")/../.."   # 回到 social_env 根目录

# --------------- 合成（V2 extended）---------------
echo "=== Phase 1: 场景合成（LLM extended，V2 确定性规则）==="
rm -rf ~/.sotopia/data/

PYTHONPATH=. python scripts/generate_long_term_negotiation_llm_extended.py \
  --model "custom/deepseek-v4-pro@https://api.deepseek.com" \
  --agent-profile-model "custom/deepseek-v4-pro@https://api.deepseek.com" \
  --n 3 \
  --scenario-mix business_coopetition,wet_market_competition,resource_scheduling_management \
  --mode-counts firms3=3 \
  --concurrency 8 \
  --timeline-labels D2 \
  --requirements "V2 确定性规则 + LLM 差异化 agent 经济参数 + 种子驱动 market_state/psych" \
  --tag ltr_v2_merged \
  --agent-profile-out long_term_negotiation_llm_agent_profiles.v2_merged.json

# --------------- 评测 ---------------
echo ""
echo "=== Phase 2: 批量评测（自动 V1/V2 检测）==="
mkdir -p logs runs

PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model "custom/deepseek-v4-flash@https://api.deepseek.com" \
    --evaluator-model "custom/deepseek-v4-flash@https://api.deepseek.com" \
    --run-config sotopia/settings/long_term_negotiation/run_config_examples/summarizing_memory.json \
    --batch-size 8 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --print-logs \
    --execution-trace-dir runs/execution_traces \
    --output runs/ \
    --tag ltr_v2_merged

echo ""
echo "=== Done ==="
echo "结果: runs/negotiation_eval_ltr_v2_merged_*.json"
echo "日志: logs/negotiation_batch_*.log"
echo "执行轨迹: runs/execution_traces/"
