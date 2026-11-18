# Support Agent Simulator

**A tool-calling customer-support AI agent, tested against LLM-simulated customers with outcome-based grading, pass^k reliability, runtime policy guardrails and tracing.**

[![CI](https://github.com/juhidixit/support-agent-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/juhidixit/support-agent-simulator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`AI Agents` · `Tool Calling / Function Calling` · `LLM Evaluation` · `Simulation-Based Testing` · `Guardrails` · `Observability (OpenTelemetry)` · `FastAPI` · `Docker` · `CI/CD (GitHub Actions)` · `Python`

Production support agents fail in ways that a chat demo never shows: they skip identity checks, refund what policy forbids, give up on multi-step requests, or succeed on Monday and fail the same request on Tuesday. This project builds a support agent for a fictional bank and measures exactly those failure modes, the way agent companies evaluate their own systems: hundreds of conversations with simulated customers, each graded on what actually changed in the database.

<!-- RESULTS_SUMMARY -->

## What it does

| Component | Details |
|---|---|
| **Agent** | Tool-calling loop over any chat model (local GGUF via llama.cpp, any OpenAI-compatible API, or Claude). 13 tools: identity verification, card freeze/close/replace, disputes, provisional credit, fee waivers, address changes, human handoff. |
| **Domain** | "Aster Bank", a fictional bank with a seeded synthetic database (40 customers, 56 cards, 633 transactions) and a written support policy the agent must follow. No real customer data. |
| **Simulated customers** | An LLM plays each customer from a private brief (identity, goal, what they know, personality) and never sees tools, policy or database. |
| **Scenario suite** | 67 scenarios across 19 categories: lost/stolen cards, disputes (valid, too old, pending, duplicate, unauthorized), fee waivers (eligible and not), address changes, multi-step requests, handoffs, and adversarial cases (failed identity checks, third-party requests, prompt injection). |
| **Outcome-based grading** | A conversation passes only if the final database exactly matches the state produced by the policy-correct actions, and the agent told the customer every required fact. Deterministic: no LLM judge. |
| **Reliability metric** | pass^k (from tau-bench): the chance that *k* repeated runs of the same scenario all succeed. |
| **Guardrail** | A deterministic policy engine checks every proposed tool call before it runs (identity verified, resource owned by the verified customer, dispute window, fee-waiver limits, provisional-credit limits, verification lockout). Blocked calls return the rule ID to the agent so it can recover. Can run in audit-only mode to measure what the model would have done on its own. |
| **Tracing** | OpenTelemetry-style spans (GenAI semantic-convention attribute names) for every conversation, LLM call and tool call, with PII redaction. |
| **Service** | FastAPI app with a web chat UI that shows each tool call and guardrail block live, plus `/healthz` and Prometheus-format `/metrics`. Dockerfile and docker-compose included. |
| **CI** | GitHub Actions: lint (ruff), 28 tests on Python 3.10 and 3.12, scenario-suite validation, an **eval regression gate** that replays recorded real-model conversations through the current code, and a Docker build + health-check smoke test. |

## Architecture

```mermaid
flowchart LR
    U[Simulated customer<br/>LLM + private brief] -- message --> A[Support agent<br/>tool-calling loop]
    A -- reply --> U
    A -- proposed tool call --> G{Policy guardrail}
    G -- allowed --> T[Bank tools]
    G -- "BLOCKED (rule id)" --> A
    T --> DB[(Synthetic bank DB<br/>per conversation)]
    T -- result --> A
    DB --> E[Grader: final DB == expected DB<br/>+ required facts stated]
    A -. spans .-> TR[Tracing + PII redaction]
    E --> M[pass rate, pass^k, violations,<br/>latency, tokens, failure types]
```

Each conversation gets its own copy of the database. The grader replays the scenario's expected actions on a fresh copy, then compares the two states field by field, so "the agent said it froze the card" never counts unless the card is actually frozen, and an extra, unrequested action fails the scenario.

<!-- RESULTS -->

## Quick start

```bash
git clone https://github.com/juhidixit/support-agent-simulator && cd support-agent-simulator
pip install -e ".[dev]"
make test validate replay           # unit tests, scenario checks, replay of recorded conversations

# Run the evaluation with a local open-weight model (CPU is fine; ~6.5 GB of downloads)
pip install -e ".[local]"
make models
supportsim run --agent llamacpp:models/qwen3-4b-2507-q4km.gguf --tasks all --trials 3 \
  --out results/my_run.jsonl --traces results/traces.jsonl
supportsim report "my run=results/my_run.jsonl"

# Or with a hosted model
export OPENAI_API_KEY=...       # any OpenAI-compatible endpoint: openai:<model>@<base_url>
supportsim run --agent openai:gpt-4.1-mini --user openai:gpt-4.1-mini --trials 3 --out results/gpt.jsonl
export ANTHROPIC_API_KEY=...    # pip install -e ".[anthropic]"
supportsim run --agent anthropic:claude-opus-5-5 --user openai:gpt-4.1-mini --out results/claude.jsonl
```

Chat with the agent yourself:

```bash
supportsim chat --agent llamacpp:models/qwen3-4b-2507-q4km.gguf              # terminal
SUPPORTSIM_AGENT=llamacpp:models/qwen3-4b-2507-q4km.gguf supportsim serve    # web UI on :8000
docker compose up                                                             # same, in Docker
```

Test identities are in `src/supportsim/data/bank_db.json` (for example `ava.murphy@example.com`, born 1985-12-19, ZIP 80203).

## CLI

| Command | Purpose |
|---|---|
| `supportsim run` | Simulate and grade conversations (`--trials`, `--guardrail on/off`, `--tasks` by id/category/tag, `--record`/`--replay` cassettes, `--workers` for API models). Resumable. |
| `supportsim report` | Markdown table: pass rate with 95% bootstrap CI, pass^k, violations, latency, tokens, per-category and failure-type breakdowns. |
| `supportsim gate` | Regression gate: exits 1 if a candidate run's paired success drop exceeds a threshold with a CI below zero, or if it executes any new policy violation. |
| `supportsim validate` | Checks every scenario's expected actions are allowed by the guardrail and produce the intended state change. |
| `supportsim chat` / `serve` | Terminal chat / FastAPI web UI. |

## Repository layout

```
src/supportsim/
  domain/        env.py (tools + per-conversation DB), generate.py (seeded synthetic data), models.py
  data/          bank_db.json, policy.md (the agent's written policy), tasks.json (67 scenarios)
  agent.py       tool-calling loop          user_sim.py   simulated customer
  guardrails.py  policy engine              evaluate.py   state-diff grading + failure taxonomy
  metrics.py     pass^k, bootstrap CIs      runner.py     orchestration (resumable JSONL)
  tracing.py     spans + PII redaction      report.py     markdown report + regression gate
  taskgen.py     scenario generator         providers/    llama.cpp, OpenAI-compatible, Anthropic, record/replay
  server/        FastAPI app + web UI
tests/           28 tests incl. scripted-model agent tests, API tests, recorded cassettes
results/         raw per-conversation results (JSONL) and RESULTS.md
```

## Design decisions

- **Grade outcomes, not transcripts.** Comparing database state is deterministic and catches both missing and extra actions. LLM judges are useful for tone, but they are not needed to know whether a dispute was filed.
- **Policy in code and in prose.** The agent reads the policy as text; the guardrail enforces the same rules as code. Running with the guardrail in audit mode measures how often the model alone breaks policy, which is the number to watch before trusting a model with real money movement.
- **Blocked calls are recoverable.** The guardrail returns a machine-readable rule ID, so a blocked agent can verify the customer and retry instead of failing the conversation.
- **Refusal scenarios are reported separately.** A scenario whose correct outcome is "change nothing" can be passed by an agent that does nothing, so results split action, refusal and handoff scenarios.
- **Deterministic CI with real model outputs.** Model responses are recorded to cassettes keyed by a hash of the full request; CI replays them through the current agent, tools, guardrail and grader. Any change to prompts or tools changes the hash and fails loudly instead of reusing stale answers.

## Limitations

- Customers are simulated by a 4B-parameter model, which sometimes over-shares, repeats itself or drifts from its brief; some failures are simulator errors rather than agent errors. Larger simulator models reduce this.
- The bank, policy and data are synthetic and much simpler than a real bank's.
- Results are from small local models on CPU; hosted frontier models will score differently. The harness supports them directly.

## License

MIT
