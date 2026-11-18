"""Any OpenAI-compatible Chat Completions endpoint: OpenAI, vLLM, Ollama, LM Studio,
llama.cpp server, Together, Groq...

Configure with ``--base-url`` and the ``OPENAI_API_KEY`` environment variable.
"""

from __future__ import annotations

import json
import os
import time

import httpx

from supportsim.providers.base import Completion, ToolCall


def _wire_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{"id": t["id"], "type": "function",
                                        "function": {"name": t["name"],
                                                     "arguments": json.dumps(t["arguments"])}}
                                       for t in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


class OpenAICompatProvider:
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, timeout: float = 120.0) -> None:
        self.name = model
        self.model = model
        key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                   headers={"Authorization": f"Bearer {key}"})

    def complete(self, messages: list[dict], tools: list[dict] | None = None, *,
                 temperature: float = 0.0, seed: int | None = None,
                 max_tokens: int = 512) -> Completion:
        body: dict = {"model": self.model, "messages": _wire_messages(messages),
                      "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        for attempt in range(4):
            r = self.client.post("/chat/completions", json=body)
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        dt = time.perf_counter() - t0
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc["function"]["arguments"]}
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args))
        usage = data.get("usage") or {}
        return Completion(content=(msg.get("content") or "").strip(), tool_calls=calls,
                          prompt_tokens=usage.get("prompt_tokens", 0),
                          completion_tokens=usage.get("completion_tokens", 0), latency_s=dt)
