# rm -rf ~/.sotopia/data/
# SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_llm.py \
#   --model gpt-4o \
#   --agent-profile-model gpt-4o \
#   --mode-counts firms2=1,firms3=1 \
#   --concurrency 8 \
#   --timeline-labels D2 \
#   --requirements "validate firm_a..firm_d expansion" \
#   --tag ltr_multi_firm_llm_v1 \
#   --agent-profile-out long_term_negotiation_llm_agent_profiles.multi_firm.json

mkdir -p logs runs
SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model gpt-5 \
    --evaluator-model gpt-5 \
    --batch-size 8 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --print-logs \
    --artifact-root runs \
    --output runs/ltr_multi_firm_eval.jsonl \
    --tag ltr_multi_firm_llm_v1