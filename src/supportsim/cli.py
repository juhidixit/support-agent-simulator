"""Command line: ``supportsim {run,report,gate,validate,chat,serve}``."""

from __future__ import annotations

import argparse
import json
import sys

from supportsim.tasks import Task, load_tasks


def _select(tasks: list[Task], spec: str | None) -> list[Task]:
    if not spec or spec == "all":
        return tasks
    keys = set(spec.split(","))
    return [t for t in tasks if t.id in keys or t.category in keys or set(t.tags) & keys]


def _provider(spec: str, slot: str, record: str | None, replay: str | None):
    from supportsim.providers import make_provider
    from supportsim.providers.replay import ReplayProvider

    if replay:
        name = spec.split(":", 1)[1].rsplit("/", 1)[-1].removesuffix(".gguf")
        return ReplayProvider(f"{replay}.{slot}.jsonl", None, name=name)
    p = make_provider(spec, slot=slot)
    return ReplayProvider(f"{record}.{slot}.jsonl", p) if record else p


def cmd_run(a: argparse.Namespace) -> int:
    from supportsim.runner import run_suite

    tasks = _select(load_tasks(a.tasks_file), a.tasks)
    agent = _provider(a.agent, "agent", a.record, a.replay)
    user = _provider(a.user or a.agent, "user", a.record, a.replay)
    print(f"running {len(tasks)} tasks x {a.trials} trials, agent={agent.name} "
          f"user={user.name} guardrail={'on' if a.guardrail == 'on' else 'off'}", flush=True)

    def show(r: dict) -> None:
        mark = "PASS" if r["passed"] else f"FAIL ({r['failure_type']})"
        print(f"  {r['task_id']:<26} trial {r['trial']}  {mark:<24} turns={r['turns']} "
              f"tools={r['tool_calls']} {r['wall_seconds']:.0f}s", flush=True)

    run_suite(tasks, agent, user, out=a.out, trials=a.trials, guardrail=a.guardrail == "on",
              workers=a.workers, trace_path=a.traces, on_result=show)
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from supportsim.report import markdown_report
    from supportsim.runner import load_results

    runs = {}
    for item in a.runs:
        name, _, path = item.rpartition("=")
        runs[name or path] = load_results(path)
    md = markdown_report(runs)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
    print(md)
    if a.json:
        from supportsim.metrics import summarize

        print(json.dumps({k: summarize(v) for k, v in runs.items()}, indent=1))
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    from supportsim.report import gate

    ok, msg = gate(a.baseline, a.candidate, a.max_drop)
    print(msg)
    return 0 if ok else 1


def cmd_validate(a: argparse.Namespace) -> int:
    """Replays every task's expected actions with the guardrail on: each action must
    be allowed and succeed, and the resulting state must differ from the start
    state exactly when the task expects a write. Guards against broken tasks."""
    from supportsim.domain.env import BankEnv, hash_db
    from supportsim.guardrails import PolicyGuard

    guard, bad = PolicyGuard(), 0
    base = hash_db(BankEnv().db)
    for t in load_tasks(a.tasks_file):
        env = BankEnv()
        for act in t.actions:
            d = guard.check(env, act.name, act.args)
            if not d.allowed:
                print(f"{t.id}: guard blocks expected {act.name} ({d.rule})")
                bad += 1
            env.execute(act.name, dict(act.args))
        changed = env.state_hash() != base
        if changed != ("write" in t.tags or "handoff" in t.tags):
            print(f"{t.id}: write tag mismatch (state changed={changed})")
            bad += 1
    n = len(load_tasks(a.tasks_file))
    print(f"validated {n} tasks: {'OK' if not bad else f'{bad} problems'}")
    return 1 if bad else 0


def cmd_chat(a: argparse.Namespace) -> int:
    from supportsim.agent import GREETING, SupportAgent
    from supportsim.domain.env import BankEnv
    from supportsim.providers import make_provider

    agent = SupportAgent(make_provider(a.agent), BankEnv(), guardrail=a.guardrail == "on")
    print(f"agent: {GREETING}")
    for line in sys.stdin:
        n = len(agent.tool_log)
        reply = agent.respond(line.strip())
        for e in agent.tool_log[n:]:
            flag = f" [BLOCKED {e['guard_rule']}]" if e["blocked"] else ""
            print(f"  tool: {e['name']}({json.dumps(e['arguments'])}){flag}")
        print(f"agent: {reply.text}")
        if reply.ended:
            break
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("supportsim.server.app:app", host=a.host, port=a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="supportsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate conversations and grade them")
    r.add_argument("--agent", required=True, help="provider spec, e.g. llamacpp:model.gguf")
    r.add_argument("--user", help="provider spec for the simulated customer (default: agent)")
    r.add_argument("--tasks", default="all", help="'all', ids, categories or tags (comma list)")
    r.add_argument("--tasks-file")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--guardrail", choices=["on", "off"], default="on")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument("--traces", help="append OpenTelemetry-style spans to this JSONL file")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--record", help="cassette prefix: record model responses")
    g.add_argument("--replay", help="cassette prefix: replay recorded responses only")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="markdown summary of one or more runs")
    p.add_argument("runs", nargs="+", help="results.jsonl or name=results.jsonl")
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)

    q = sub.add_parser("gate", help="fail (exit 1) if candidate regresses vs baseline")
    q.add_argument("--baseline", required=True)
    q.add_argument("--candidate", required=True)
    q.add_argument("--max-drop", type=float, default=0.05)
    q.set_defaults(fn=cmd_gate)

    v = sub.add_parser("validate", help="check the scenario suite is self-consistent")
    v.add_argument("--tasks-file")
    v.set_defaults(fn=cmd_validate)

    c = sub.add_parser("chat", help="chat with the agent in the terminal")
    c.add_argument("--agent", required=True)
    c.add_argument("--guardrail", choices=["on", "off"], default="on")
    c.set_defaults(fn=cmd_chat)

    s = sub.add_parser("serve", help="start the FastAPI service and web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
