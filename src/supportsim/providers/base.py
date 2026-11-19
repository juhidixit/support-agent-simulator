"""Provider interface shared by every model backend.

Messages use one internal, OpenAI-like shape so the agent loop never cares which
backend is behind it::

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [{"id": str, "name": str, "arguments": dict}],   # assistant only
     "tool_call_id": str, "name": str}                               # tool only
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific (e.g. raw blocks)

    def to_dict(self) -> dict:
        return {"content": self.content, "tool_calls": [t.to_dict() for t in self.tool_calls],
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "latency_s": self.latency_s, "extra": self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> Completion:
        return cls(content=d["content"], tool_calls=[ToolCall(**t) for t in d["tool_calls"]],
                   prompt_tokens=d["prompt_tokens"], completion_tokens=d["completion_tokens"],
                   latency_s=d["latency_s"], extra=d.get("extra", {}))


class Provider(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict] | None = None, *,
                 temperature: float = 0.0, seed: int | None = None,
                 max_tokens: int = 512) -> Completion: ...


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def _json_objects(text: str) -> list[tuple[int, int, Any]]:
    """Find top-level JSON objects embedded in free text."""
    dec = json.JSONDecoder()
    out, i = [], 0
    while (j := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        out.append((j, end, obj))
        i = end
    return out


def _as_call(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    return ToolCall(new_call_id(), obj["name"], args if isinstance(args, dict) else {})


def parse_tool_calls(text: str, tool_names: set[str]) -> tuple[str, list[ToolCall]]:
    """Extract tool calls from raw model text.

    Handles Hermes/Qwen style ``<tool_call>{...}</tool_call>`` blocks and Llama 3.x
    style bare JSON (``{"name": ..., "parameters": {...}}``, optionally after
    ``<|python_tag|>``). Returns (visible text, calls)."""
    calls: list[ToolCall] = []
    if "<tool_call>" in text:
        for block in _TOOL_TAG.findall(text):
            for _s, _e, obj in _json_objects(block):
                if (c := _as_call(obj)) is not None:
                    calls.append(c)
        visible = text.split("<tool_call>")[0]
        return visible.strip(), calls
    cleaned = text.replace("<|python_tag|>", "")
    spans = []
    for s, e, obj in _json_objects(cleaned):
        c = _as_call(obj)
        if c is not None and c.name in tool_names:
            calls.append(c)
            spans.append((s, e))
    if calls:
        visible = cleaned
        for s, e in reversed(spans):
            visible = visible[:s] + visible[e:]
        return visible.strip(), calls
    return text.strip(), []
