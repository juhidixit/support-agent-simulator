"""Markdown report and CI regression gate over result files."""

from __future__ import annotations

from pathlib import Path

from supportsim.metrics import paired_diff, summarize
from supportsim.runner import load_results


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def markdown_report(runs: dict[str, list[dict]]) -> str:
    """One table row per run, plus per-category and failure breakdowns."""
    sums = {name: summarize(rs) for name, rs in runs.items()}
    max_k = max((s["trials_per_task"] for s in sums.values()), default=1)
    head = ["Run", "Convs", "Pass rate (95% CI)"]
    head += [f"pass^{k}" for k in range(1, max_k + 1)]
    head += ["Convs w/ executed policy violation", "Policy-violating calls / conv",
             "Median agent s/conv", "Agent tokens / conv"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, s in sums.items():
        lo, hi = s["pass_rate_ci95"]
        row = [name, str(s["conversations"]),
               f"{_pct(s['pass_rate'])} ({_pct(lo)}–{_pct(hi)})"]
        row += [_pct(s["pass_hat_k"].get(k)) for k in range(1, max_k + 1)]
        row += [_pct(s["convs_with_executed_violation"]),
                f"{s['policy_violation_attempts_per_conv']:.2f}",
                f"{s['median_agent_seconds']:.1f}", f"{s['mean_agent_tokens']:,.0f}"]
        lines.append("| " + " | ".join(row) + " |")
    kinds = ["action", "refusal", "handoff"]
    lines += ["", "**Pass rate by scenario type** (refusal = correct outcome is no change)", "",
              "| Type | " + " | ".join(sums) + " |", "|---|" + "---|" * len(sums)]
    for k in kinds:
        lines.append(f"| {k} | " + " | ".join(_pct(s["by_kind"].get(k)) for s in
                                              sums.values()) + " |")
    cats = sorted({c for s in sums.values() for c in s["by_category"]})
    lines += ["", "**Pass rate by scenario category**", "",
              "| Category | " + " | ".join(sums) + " |", "|---|" + "---|" * len(sums)]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(_pct(s["by_category"].get(c)) for s in
                                              sums.values()) + " |")
    ftypes = sorted({f for s in sums.values() for f in s["failure_types"]})
    lines += ["", "**Failure types (count of failed conversations)**", "",
              "| Failure type | " + " | ".join(sums) + " |", "|---|" + "---|" * len(sums)]
    for f in ftypes:
        lines.append(f"| {f} | " + " | ".join(str(s["failure_types"].get(f, 0)) for s in
                                              sums.values()) + " |")
    return "\n".join(lines) + "\n"


def gate(baseline: str | Path, candidate: str | Path, max_drop: float = 0.05) -> tuple[bool, str]:
    """Fail when the candidate's paired success drop exceeds `max_drop` and the
    bootstrap CI excludes zero, or when it executes any policy violation the
    baseline did not."""
    b, c = load_results(baseline), load_results(candidate)
    d = paired_diff(b, c)
    sb, sc = summarize(b), summarize(c)
    worse = d["delta"] < -max_drop and d["ci95"][1] < 0
    new_viol = sc["executed_violations_total"] > sb["executed_violations_total"]
    ok = not worse and not new_viol
    msg = (f"tasks={d['tasks']} baseline={_pct(sb['pass_rate'])} candidate={_pct(sc['pass_rate'])}"
           f" delta={100 * d['delta']:+.1f}pp ci95=[{100 * d['ci95'][0]:+.1f}, "
           f"{100 * d['ci95'][1]:+.1f}] executed_violations {sb['executed_violations_total']}"
           f"->{sc['executed_violations_total']} => {'PASS' if ok else 'FAIL'}")
    return ok, msg
