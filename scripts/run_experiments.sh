#!/usr/bin/env bash
# Reproduces results/ on a CPU machine (4 vCPU: roughly 2-3 minutes per conversation).
# Every step is resumable: re-running skips (task, trial) pairs already in the output file.
set -euo pipefail
M=${MODELS_DIR:-models}
USER_SIM="llamacpp:$M/qwen3-4b-2507-q4km.gguf"
run() { supportsim run --user "$USER_SIM" --traces results/traces.jsonl "$@"; }

# 0. small recorded run used by CI (cassette replay, no model needed)
supportsim run --agent "llamacpp:$M/qwen3-4b-2507-q4km.gguf" --user "$USER_SIM" \
  --tasks card_lost_01,dispute_premium_01,fee_waiver_denied_01,prompt_injection_02,third_party_01,address_and_replace_01 \
  --record tests/cassettes/qwen3-4b --out tests/cassettes/qwen3-4b.results.jsonl
# 1. main model, guardrail on / off (first trial)
run --agent "llamacpp:$M/qwen3-4b-2507-q4km.gguf" --guardrail on  --trials 1 --out results/qwen3-4b_guard-on.jsonl
run --agent "llamacpp:$M/qwen3-4b-2507-q4km.gguf" --guardrail off --trials 1 --out results/qwen3-4b_guard-off.jsonl
# 2. smaller agents, same simulated customer
run --agent "llamacpp:$M/qwen2.5-3b-q4km.gguf"  --guardrail on --trials 1 --out results/qwen2.5-3b_guard-on.jsonl
run --agent "llamacpp:$M/llama3.2-3b-q4km.gguf" --guardrail on --trials 1 --out results/llama3.2-3b_guard-on.jsonl
# 3. reliability: two more trials of the main configuration for pass^k
run --agent "llamacpp:$M/qwen3-4b-2507-q4km.gguf" --guardrail on  --trials 3 --out results/qwen3-4b_guard-on.jsonl
