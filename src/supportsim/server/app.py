"""FastAPI service: chat sessions with the support agent, per-session traces,
health and Prometheus-style metrics, and a small web UI.

Configure the model with ``SUPPORTSIM_AGENT`` (a provider spec, e.g.
``llamacpp:/models/qwen3-4b.gguf`` or ``openai:gpt-4.1-mini``) and the guardrail
with ``SUPPORTSIM_GUARDRAIL=on|off``.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import Counter
from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from supportsim.agent import GREETING, SupportAgent
from supportsim.domain.env import BankEnv
from supportsim.evaluate import diff_db
from supportsim.tracing import Tracer

app = FastAPI(title="Support Agent Simulator", version="0.1.0")

_sessions: dict[str, dict] = {}
_lock = threading.Lock()
METRICS: Counter = Counter()
LATENCY: list[float] = []
MAX_SESSIONS = 200


def get_provider():
    """Overridable in tests via ``app.dependency_overrides`` style monkeypatching."""
    from supportsim.providers import make_provider

    spec = os.environ.get("SUPPORTSIM_AGENT")
    if not spec:
        raise HTTPException(503, "set SUPPORTSIM_AGENT to a provider spec to enable chat")
    return make_provider(spec)


class NewMessage(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "sessions": len(_sessions)}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    lat = sorted(LATENCY) or [0.0]
    lines = [f"supportsim_{k}_total {v}" for k, v in sorted(METRICS.items())]
    lines += [f'supportsim_turn_latency_seconds{{quantile="0.5"}} {lat[len(lat) // 2]:.3f}',
              f'supportsim_turn_latency_seconds{{quantile="0.95"}} '
              f'{lat[max(0, int(0.95 * len(lat)) - 1)]:.3f}']
    return "\n".join(lines) + "\n"


@app.post("/api/sessions")
def create_session() -> dict:
    provider = app.state.provider_factory()
    sid = uuid.uuid4().hex[:12]
    tracer = Tracer()
    guard = os.environ.get("SUPPORTSIM_GUARDRAIL", "on") != "off"
    agent = SupportAgent(provider, BankEnv(), guardrail=guard, tracer=tracer)
    with _lock:
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))
        _sessions[sid] = {"agent": agent, "tracer": tracer, "lock": threading.Lock(),
                          "ended": False}
    METRICS["sessions"] += 1
    return {"session_id": sid, "greeting": GREETING, "guardrail": guard}


def _session(sid: str) -> dict:
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, "unknown session")
    return s


@app.post("/api/sessions/{sid}/messages")
def send_message(sid: str, msg: NewMessage) -> dict:
    s = _session(sid)
    if s["ended"]:
        raise HTTPException(409, "conversation was handed to a human")
    agent: SupportAgent = s["agent"]
    with s["lock"]:
        n = len(agent.tool_log)
        t0 = time.perf_counter()
        reply = agent.respond(msg.text)
        LATENCY.append(time.perf_counter() - t0)
        s["ended"] = reply.ended
    new = agent.tool_log[n:]
    METRICS["turns"] += 1
    METRICS["tool_calls"] += len(new)
    METRICS["guardrail_blocks"] += sum(1 for e in new if e["blocked"])
    METRICS["handoffs"] += int(reply.ended)
    return {"reply": reply.text, "ended": reply.ended,
            "tool_calls": [{"name": e["name"], "arguments": e["arguments"],
                            "blocked": e["blocked"], "rule": e["guard_rule"],
                            "error": e["error"]} for e in new]}


@app.get("/api/sessions/{sid}/trace")
def trace(sid: str) -> list[dict]:
    return sorted(_session(sid)["tracer"].spans, key=lambda x: x["start_time_unix_nano"])


@app.get("/api/sessions/{sid}/changes")
def changes(sid: str) -> list[str]:
    """What this conversation changed in the bank's database."""
    return diff_db(BankEnv().db, _session(sid)["agent"].env.db)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return resources.files("supportsim.server").joinpath("static/index.html").read_text()


app.state.provider_factory = get_provider
