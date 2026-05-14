# rm -rf ~/.sotopia/data/
export OPENAI_API_BASE=https://api.deepseek.com
export CUSTOM_API_KEY=sk-4b5a28b9dbd345ef9cfd67d6fb34e9e6

# SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_llm.py \
#   --model "custom/deepseek-v4-pro@https://api.deepseek.com" \
#   --agent-profile-model "custom/deepseek-v4-pro@https://api.deepseek.com" \
#   --n 6 \
#   --scenario-mix business_coopetition,wet_market_competition,resource_scheduling_management \
#   --mode-counts firms2=3,firms3=3 \
#   --concurrency 8 \
#   --timeline-labels D2 \
#   --requirements "validate firm_a..firm_d expansion" \
#   --tag ltr_multi_firm_llm_v1 \
#   --agent-profile-out long_term_negotiation_llm_agent_profiles.multi_firm.json \
#   --incremental-diversity \
#   --diversity-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json
  
# mkdir -p logs runs
# SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
#     --agent-model gpt-5-mini \
#     --evaluator-model gpt-5-mini \
#     --batch-size 8 --repeats 1 \
#     --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
#     --print-logs \
#     --artifact-root runs \
#     --output runs \
#     --tag ltr_multi_firm_llm_v1

# --- 以下为 README 长期谈判批量评测示例（--execution-trace-dir；与上文二选一执行）---
# 汇总 JSON：runs/negotiation_eval_<tag>_<时间戳>.json；文本 log：logs/negotiation_batch_<时间戳>.log
# 每场 trace：runs/execution_traces/<模型>/<时间戳>/<tag>_<runid>_<seq>_<角色或terminal_evaluator>.jsonl
# mkdir -p logs runs
export SOTOPIA_MAX_RENDERED_USER_CHARS=0
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model "custom/deepseek-v4-flash@https://api.deepseek.com" \
    --evaluator-model "custom/deepseek-v4-flash@https://api.deepseek.com" \
    --run-config sotopia/settings/long_term_negotiation/run_config_examples/summarizing_memory.json \
    --batch-size 8 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --print-logs \
    --execution-trace-dir runs/execution_traces \
    --output runs/ \
    --tag ltr_multi_firm_llm_v1

# rm -rf ~/.sotopia/data/
# SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_llm.py \
#   --model gpt-4o \
#   --agent-profile-model gpt-4o \
#   --n 6 \
#   --scenario-mix business_coopetition,wet_market_competition,resource_scheduling_management \
#   --mode-counts firms2=3,firms3=3 \
#   --concurrency 8 \
#   --timeline-labels D2 \
#   --requirements "validate firm_a..firm_d expansion" \
#   --tag ltr_multi_firm_llm_v1 \
#   --agent-profile-out long_term_negotiation_llm_agent_profiles.multi_firm.json

# # mkdir -p logs runs
# # SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
# #     --agent-model gpt-5-mini \
# #     --evaluator-model gpt-5-mini \
# #     --batch-size 8 --repeats 1 \
# #     --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
# #     --print-logs \
# #     --artifact-root runs \
# #     --output runs \
# #     --tag ltr_multi_firm_llm_v1

# # --- 以下为 README 长期谈判批量评测示例（--execution-trace-dir；与上文二选一执行）---
# # 汇总 JSON：runs/negotiation_eval_<tag>_<时间戳>.json；文本 log：logs/negotiation_batch_<时间戳>.log
# # 每场 trace：runs/execution_traces/<模型>/<时间戳>/<tag>_<runid>_<seq>_<角色或terminal_evaluator>.jsonl
# # mkdir -p logs runs
# SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
#     --agent-model gpt-5-mini \
#     --evaluator-model gpt-5-mini \
#     --batch-size 8 --repeats 1 \
#     --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
#     --print-logs \
#     --execution-trace-dir runs/execution_traces \
#     --output runs/ \
#     --tag ltr_multi_firm_llm_v1