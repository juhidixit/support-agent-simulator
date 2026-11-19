"""Aggregate metrics: pass^k reliability, bootstrap confidence intervals, and a
paired regression test between two runs."""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb
from statistics import mean, median

REFUSAL = ("dispute_too_old", "dispute_pending", "dispute_duplicate", "fee_waiver_denied",
           "idv_failure", "third_party", "prompt_injection")


def _kind(category: str) -> str:
    """'action' scenarios need a correct write; 'refusal' scenarios pass only if the
    agent changes nothing (reported separately so do-nothing behavior cannot hide)."""
    if category in REFUSAL:
        return "refusal"
    return "handoff" if category.startswith("transfer") else "action"


def by_task(results: list[dict]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        out[r["task_id"]].append(bool(r["passed"]))
    return out


def pass_hat_k(results: list[dict], k: int) -> float | None:
    """tau-bench's pass^k: probability that k i.i.d. trials of a task ALL succeed,
    estimated without bias per task as C(c, k) / C(n, k), then averaged over tasks."""
    vals = []
    for trials in by_task(results).values():
        n, c = len(trials), sum(trials)
        if n >= k:
            vals.append(comb(c, k) / comb(n, k))
    return mean(vals) if vals else None


def bootstrap_ci(results: list[dict], iters: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for mean success, resampling *tasks* (trials of a task stay together)."""
    tasks = list(by_task(results).values())
    if not tasks:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(iters):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        stats.append(mean(mean(t) for t in sample))
    stats.sort()
    return stats[int(alpha / 2 * iters)], stats[int((1 - alpha / 2) * iters) - 1]


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    n_trials = max(len(v) for v in by_task(results).values())
    lo, hi = bootstrap_ci(results)
    viol = [len(r["violations"]) for r in results]
    executed = [sum(1 for v in r["violations"] if not v["blocked"]) for r in results]
    out = {
        "conversations": len(results),
        "tasks": len(by_task(results)),
        "trials_per_task": n_trials,
        "pass_rate": mean(r["passed"] for r in results),
        "pass_rate_ci95": [lo, hi],
        "pass_hat_k": {k: pass_hat_k(results, k) for k in range(1, n_trials + 1)},
        "policy_violation_attempts_per_conv": mean(viol),
        "convs_with_executed_violation": mean(1 if e else 0 for e in executed),
        "executed_violations_total": sum(executed),
        "median_agent_turns": median(r["turns"] for r in results),
        "mean_tool_calls": mean(r["tool_calls"] for r in results),
        "mean_agent_tokens": mean(r["agent_usage"]["prompt_tokens"] +
                                  r["agent_usage"]["completion_tokens"] for r in results),
        "median_agent_seconds": median(r["agent_usage"]["llm_seconds"] for r in results),
        "p95_agent_seconds": sorted(r["agent_usage"]["llm_seconds"] for r in results)[
            max(0, int(0.95 * len(results)) - 1)],
        "errors": sum(1 for r in results if r.get("error")),
    }
    fails: dict[str, int] = defaultdict(int)
    for r in results:
        if not r["passed"]:
            fails[r.get("failure_type") or "error"] += 1
    out["failure_types"] = dict(sorted(fails.items()))
    cats: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(bool(r["passed"]))
    out["by_category"] = {c: mean(v) for c, v in sorted(cats.items())}
    kinds: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        kinds[_kind(r["category"])].append(bool(r["passed"]))
    out["by_kind"] = {k: mean(v) for k, v in sorted(kinds.items())}
    return out


def paired_diff(base: list[dict], cand: list[dict], iters: int = 5000,
                seed: int = 0) -> dict:
    """Candidate minus baseline success, paired by task, with a bootstrap CI and a
    bootstrap estimate of the probability that the candidate is worse."""
    b, c = by_task(base), by_task(cand)
    common = sorted(set(b) & set(c))
    deltas = [mean(c[t]) - mean(b[t]) for t in common]
    if not deltas:
        return {"tasks": 0, "delta": 0.0, "ci95": [0.0, 0.0], "prob_worse": 0.0}
    rng = random.Random(seed)
    boots = []
    for _ in range(iters):
        boots.append(mean(deltas[rng.randrange(len(deltas))] for _ in deltas))
    boots.sort()
    return {"tasks": len(common), "delta": mean(deltas),
            "ci95": [boots[int(0.025 * iters)], boots[int(0.975 * iters) - 1]],
            "prob_worse": sum(1 for x in boots if x < 0) / iters}
