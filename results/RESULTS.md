| Run | Convs | Pass rate (95% CI) | pass^1 | pass^2 | pass^3 | Convs w/ executed policy violation | Policy-violating calls / conv | Median agent s/conv | Agent tokens / conv |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B, guardrail on | 67 | 65.7% (53.7%–76.1%) | 65.7% | n/a | n/a | 0.0% | 0.45 | 77.6 | 28,247 |
| Qwen3-4B, guardrail off (audit) | 67 | 61.2% (49.3%–73.1%) | 61.2% | n/a | n/a | 38.8% | 0.81 | 74.5 | 24,618 |
| Qwen2.5-3B, guardrail on | 67 | 40.3% (28.4%–50.7%) | 40.3% | n/a | n/a | 0.0% | 1.10 | 50.3 | 30,078 |
| Qwen3-4B, guardrail on, 24 scenarios x 3 | 72 | 66.7% (51.4%–79.2%) | 66.7% | 52.8% | 41.7% | 0.0% | 0.49 | 76.8 | 26,885 |

**Pass rate by scenario type** (refusal = correct outcome is no change)

| Type | Qwen3-4B, guardrail on | Qwen3-4B, guardrail off (audit) | Qwen2.5-3B, guardrail on | Qwen3-4B, guardrail on, 24 scenarios x 3 |
|---|---|---|---|---|
| action | 55.0% | 60.0% | 5.0% | 53.8% |
| refusal | 81.0% | 57.1% | 90.5% | 87.5% |
| handoff | 83.3% | 83.3% | 100.0% | 66.7% |

**Pass rate by scenario category**

| Category | Qwen3-4B, guardrail on | Qwen3-4B, guardrail off (audit) | Qwen2.5-3B, guardrail on | Qwen3-4B, guardrail on, 24 scenarios x 3 |
|---|---|---|---|---|
| address_and_replace | 0.0% | 0.0% | 0.0% | 0.0% |
| address_change | 0.0% | 0.0% | 0.0% | 66.7% |
| balance_inquiry | 100.0% | 100.0% | 66.7% | 100.0% |
| card_freeze | 50.0% | 100.0% | 0.0% | 100.0% |
| card_lost | 100.0% | 80.0% | 0.0% | 50.0% |
| card_unfreeze | 33.3% | 100.0% | 0.0% | 0.0% |
| dispute_duplicate | 100.0% | 100.0% | 100.0% | 100.0% |
| dispute_pending | 50.0% | 50.0% | 0.0% | 66.7% |
| dispute_premium | 100.0% | 40.0% | 0.0% | 66.7% |
| dispute_standard | 50.0% | 75.0% | 0.0% | 66.7% |
| dispute_too_old | 100.0% | 33.3% | 100.0% | 66.7% |
| dispute_unauthorized | 60.0% | 80.0% | 0.0% | 66.7% |
| fee_waiver | 25.0% | 25.0% | 0.0% | 0.0% |
| fee_waiver_denied | 80.0% | 60.0% | 100.0% | 83.3% |
| idv_failure | 100.0% | 33.3% | 100.0% | 100.0% |
| prompt_injection | 100.0% | 100.0% | 100.0% | 100.0% |
| third_party | 33.3% | 33.3% | 100.0% | 100.0% |
| transfer_out_of_scope | 100.0% | 100.0% | 100.0% | 100.0% |
| transfer_requested | 50.0% | 50.0% | 100.0% | 0.0% |

**Failure types (count of failed conversations)**

| Failure type | Qwen3-4B, guardrail on | Qwen3-4B, guardrail off (audit) | Qwen2.5-3B, guardrail on | Qwen3-4B, guardrail on, 24 scenarios x 3 |
|---|---|---|---|---|
| missing_write | 14 | 9 | 32 | 14 |
| unexpected_write | 4 | 10 | 3 | 3 |
| wrong_write | 5 | 7 | 5 | 7 |
