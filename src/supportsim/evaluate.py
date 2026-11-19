"""Outcome-based grading.

A conversation passes when (1) the final database equals the database produced
by replaying the task's expected actions, and (2) every required fact (e.g. a
balance) appears in the agent's messages. Grading never asks an LLM for an
opinion, so the same transcript always gets the same score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from supportsim.domain.env import BankEnv, normalize_db
from supportsim.tasks import Task


def expected_env(task: Task) -> BankEnv:
    env = BankEnv()
    for a in task.actions:
        env.execute(a.name, dict(a.args))
    return env


def diff_db(expected: dict, actual: dict, prefix: str = "") -> list[str]:
    """Human-readable list of differences between two database snapshots."""
    out: list[str] = []
    keys = sorted(set(expected) | set(actual), key=str)
    for k in keys:
        path = f"{prefix}.{k}" if prefix else str(k)
        if k not in actual:
            out.append(f"missing {path}")
        elif k not in expected:
            out.append(f"unexpected {path}")
        elif isinstance(expected[k], dict) and isinstance(actual[k], dict):
            out.extend(diff_db(expected[k], actual[k], path))
        elif expected[k] != actual[k]:
            out.append(f"{path}: expected {expected[k]!r}, got {actual[k]!r}")
    return out


def _norm_text(s: str) -> str:
    return re.sub(r"[,$]", "", s.lower())


@dataclass
class Grade:
    passed: bool
    db_match: bool
    outputs_ok: bool
    diff: list[str] = field(default_factory=list)
    missing_outputs: list[str] = field(default_factory=list)
    failure_type: str | None = None


def classify(exp_changes: set[str], act_changes: set[str]) -> str:
    """Bucket a DB mismatch: the agent skipped a required write, made an extra one,
    or made the right kind of write with the wrong details."""
    missing, extra = exp_changes - act_changes, act_changes - exp_changes
    if missing and not extra:
        return "missing_write"
    if extra and not missing:
        return "unexpected_write"
    return "wrong_write"


def _changed_fields(a: dict, b: dict) -> set[str]:
    return {d.split(":")[0].replace("missing ", "").replace("unexpected ", "")
            for d in diff_db(a, b)}


def grade(task: Task, env: BankEnv, agent_texts: list[str]) -> Grade:
    base = normalize_db(BankEnv().db)
    exp = normalize_db(expected_env(task).db)
    act = normalize_db(env.db)
    diff = diff_db(exp, act)
    joined = _norm_text(" ".join(agent_texts))
    missing = [o for o in task.outputs if _norm_text(o) not in joined]
    passed = not diff and not missing
    ftype = None
    if diff:
        ftype = classify(_changed_fields(base, exp), _changed_fields(base, act))
    elif missing:
        ftype = "missing_info"
    return Grade(passed, not diff, not missing, diff, missing, ftype)
