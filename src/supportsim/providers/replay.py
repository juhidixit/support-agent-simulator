"""Record/replay ("cassette") wrapper for deterministic CI.

In ``record`` mode every call is forwarded to the real provider and the response
is appended to a JSONL cassette keyed by a hash of the full request. In ``replay``
mode responses come from the cassette only, so the whole evaluation pipeline can
run in CI with real model outputs, no GPU and no API key. A prompt or tool change
alters the hash and fails loudly instead of silently reusing a stale answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from supportsim.providers.base import Completion


class CassetteMiss(KeyError):
    pass


def request_key(model: str, messages: list[dict], tools: list[dict] | None,
                temperature: float, seed: int | None) -> str:
    clean = [{k: v for k, v in m.items() if k not in ("provider_blocks", "tool_calls")}
             | ({"tool_calls": [{"name": t["name"], "arguments": t["arguments"]}
                                for t in m["tool_calls"]]} if m.get("tool_calls") else {})
             for m in messages]
    # tool-call ids are random; drop them so a replayed conversation hashes the same
    for m in clean:
        m.pop("tool_call_id", None)
    blob = json.dumps({"model": model, "messages": clean, "tools": tools,
                       "temperature": temperature, "seed": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class ReplayProvider:
    def __init__(self, cassette: str | Path, inner=None, *, name: str | None = None) -> None:
        self.path = Path(cassette)
        self.inner = inner
        self.name = name or (inner.name if inner else "replay")
        self.store: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.store[rec["key"]] = rec["completion"]

    def complete(self, messages: list[dict], tools: list[dict] | None = None, *,
                 temperature: float = 0.0, seed: int | None = None,
                 max_tokens: int = 512) -> Completion:
        key = request_key(self.name, messages, tools, temperature, seed)
        if key in self.store:
            return Completion.from_dict(self.store[key])
        if self.inner is None:
            raise CassetteMiss(f"no recorded response for request {key[:12]} in {self.path}; "
                               "re-record with --record")
        comp = self.inner.complete(messages, tools, temperature=temperature, seed=seed,
                                   max_tokens=max_tokens)
        self.store[key] = comp.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "completion": comp.to_dict()}) + "\n")
        return comp
