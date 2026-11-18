.PHONY: install test lint validate replay models eval report serve docker

install:          ## dev install (add .[local] for llama.cpp models)
	pip install -e ".[dev]"
lint:
	ruff check src tests
test:
	pytest -q
validate:         ## check every scenario's expected actions are policy-compliant
	supportsim validate
replay:           ## re-grade recorded real-model conversations (no model needed) + regression gate
	supportsim run --agent llamacpp:models/qwen3-4b-2507-q4km.gguf --replay tests/cassettes/qwen3-4b \
	  --tasks card_lost_01,dispute_premium_01,fee_waiver_denied_01,prompt_injection_02,third_party_01,address_and_replace_01 \
	  --out /tmp/replay.jsonl
	supportsim gate --baseline tests/cassettes/qwen3-4b.results.jsonl --candidate /tmp/replay.jsonl
models:           ## download the GGUF models used in results/ (~8.5 GB)
	mkdir -p models
	scripts/pull_gguf.sh ai/qwen3 4b-instruct-2507-q4_K_M models/qwen3-4b-2507-q4km.gguf
	scripts/pull_gguf.sh ai/qwen2.5 3B-Q4_K_M models/qwen2.5-3b-q4km.gguf
	scripts/pull_gguf.sh ai/llama3.2 3B-Q4_K_M models/llama3.2-3b-q4km.gguf
eval:             ## full experiment suite (hours on CPU; resumable)
	scripts/run_experiments.sh
report:
	supportsim report "Qwen3-4B, guardrail on=results/qwen3-4b_guard-on.jsonl" \
	  "Qwen3-4B, guardrail off=results/qwen3-4b_guard-off.jsonl" \
	  "Qwen2.5-3B, guardrail on=results/qwen2.5-3b_guard-on.jsonl" \
	  "Llama-3.2-3B, guardrail on=results/llama3.2-3b_guard-on.jsonl" --out results/RESULTS.md
serve:
	supportsim serve
docker:
	docker build -t support-agent-simulator .
