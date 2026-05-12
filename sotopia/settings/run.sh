cd social_env

SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python scripts/generate_long_term_negotiation_llm.py \
  --model gpt-4o-mini \
  --agent-profile-model gpt-4o-mini \
  --mode-counts firms3=3 \
  --concurrency 8 \
  --timeline-labels D1,D2 \
  --requirements "validate firm_a..firm_d expansion" \
  --tag ltr_multi_firm_llm_v1 \
  --agent-profile-out long_term_negotiation_llm_agent_profiles.multi_firm.json

SOTOPIA_STORAGE_BACKEND=local PYTHONPATH=. python -m sotopia.cli.benchmark.negotiation_batch negotiation-batch \
    --agent-model gpt-5 \
    --evaluator-model gpt-5 \
    --batch-size 8 --repeats 1 \
    --scenario-manifest ~/.sotopia/data/long_term_negotiation_llm_manifest.json \
    --print-logs \
    --artifact-root runs \
    --output runs/ltr_multi_firm_eval.jsonl \
    --tag ltr_multi_firm_llm_v1