"""Runs simulated conversations and writes one JSON line per (task, trial)."""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from supportsim.agent import GREETING, SupportAgent
from supportsim.domain.env import BankEnv
from supportsim.evaluate import grade
from supportsim.providers.base import Provider
from supportsim.tasks import Task
from supportsim.tracing import Tracer
from supportsim.user_sim import STOP, SimulatedUser


def run_conversation(task: Task, agent_provider: Provider, user_provider: Provider, *,
                     trial: int = 0, guardrail: bool = True, max_turns: int = 14,
                     tracer: Tracer | None = None) -> dict:
    tracer = tracer or Tracer()
    env = BankEnv()
    agent = SupportAgent(agent_provider, env, guardrail=guardrail, tracer=tracer, seed=trial)
    user = SimulatedUser(user_provider, task.instruction, seed=1000 * trial + 1)
    transcript = [{"role": "agent", "text": GREETING}]
    agent_texts = [GREETING]
    error, end_reason = None, "max_turns"
    t0 = time.perf_counter()
    with tracer.span("conversation", **{"task.id": task.id, "task.category": task.category,
                                        "trial": trial, "guardrail": guardrail,
                                        "gen_ai.request.model": agent_provider.name}) as root:
        try:
            agent_text = GREETING
            for _ in range(max_turns):
                with tracer.span("llm.call", **{"llm.role": "simulated_user",
                                                "gen_ai.request.model": user_provider.name}):
                    user_text = user.reply(agent_text)
                prev = [m["text"] for m in transcript if m["role"] == "customer"]
                if prev and user_text == prev[-1]:  # simulator stuck repeating itself
                    end_reason = "customer_repeat"
                    break
                if STOP in user_text:
                    before = user_text.replace(STOP, "").strip()
                    if before:
                        transcript.append({"role": "customer", "text": before})
                    end_reason = "customer_done"
                    break
                transcript.append({"role": "customer", "text": user_text})
                reply = agent.respond(user_text)
                agent_text = reply.text
                agent_texts.append(reply.text)
                transcript.append({"role": "agent", "text": reply.text})
                if reply.ended:
                    end_reason = "handoff"
                    break
        except Exception as e:  # a crash is a failed conversation, not a failed run
            error = f"{type(e).__name__}: {e}"
            end_reason = "error"
            traceback.print_exc()
        g = grade(task, env, agent_texts)
        root.set(**{"eval.passed": g.passed, "eval.failure_type": g.failure_type})
        trace_id = root.trace_id
    return {
        "task_id": task.id, "category": task.category, "trial": trial,
        "agent_model": agent_provider.name, "user_model": user_provider.name,
        "guardrail": guardrail, "passed": g.passed and error is None,
        "db_match": g.db_match, "outputs_ok": g.outputs_ok, "diff": g.diff[:20],
        "missing_outputs": g.missing_outputs,
        "failure_type": g.failure_type if error is None else "error",
        "end_reason": end_reason, "error": error,
        "turns": sum(1 for m in transcript if m["role"] == "customer"),
        "tool_calls": len(agent.tool_log),
        "violations": [{"step": e["step"], "tool": e["name"], "rule": e["guard_rule"],
                        "blocked": e["blocked"]} for e in agent.tool_log if e["guard_rule"]],
        "tool_log": agent.tool_log, "transcript": transcript,
        "agent_usage": agent.usage, "user_usage": user.usage,
        "wall_seconds": round(time.perf_counter() - t0, 2), "trace_id": trace_id,
    }


def done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["task_id"], r["trial"]))
    return keys


def run_suite(tasks: list[Task], agent_provider: Provider, user_provider: Provider, *,
              out: str | Path, trials: int = 1, guardrail: bool = True, workers: int = 1,
              trace_path: str | Path | None = None,
              on_result: Callable[[dict], None] | None = None) -> list[dict]:
    """Run every task `trials` times, appending results to `out` (resumable)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(out)
    tracer = Tracer(trace_path)
    jobs = [(t, k) for k in range(trials) for t in tasks if (t.id, k) not in skip]

    def one(job: tuple[Task, int]) -> dict:
        task, k = job
        r = run_conversation(task, agent_provider, user_provider, trial=k,
                             guardrail=guardrail, tracer=tracer)
        with out.open("a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        if on_result:
            on_result(r)
        return r

    if workers <= 1:
        return [one(j) for j in jobs]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
