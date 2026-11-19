"""Claude through the official ``anthropic`` Python SDK (optional dependency).

Install with ``pip install -e ".[anthropic]"`` and set ``ANTHROPIC_API_KEY``.
Assistant turns keep the raw content blocks returned by the API (including any
thinking blocks) and send them back unchanged on the next request.
"""

from __future__ import annotations

import time

from supportsim.providers.base import Completion, ToolCall

DEFAULT_MODEL = "claude-opus-5-5"


def _to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant":
            blocks = m.get("provider_blocks")
            if not blocks:
                blocks = ([{"type": "text", "text": m["content"]}] if m.get("content") else [])
                blocks += [{"type": "tool_use", "id": t["id"], "name": t["name"],
                            "input": t["arguments"]} for t in m.get("tool_calls") or []]
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m["content"]}
            # all results for one assistant turn go in a single user message
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return system, out


class AnthropicProvider:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low") -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = model
        self.effort = effort

    def complete(self, messages: list[dict], tools: list[dict] | None = None, *,
                 temperature: float = 0.0, seed: int | None = None,
                 max_tokens: int = 4096) -> Completion:
        system, msgs = _to_anthropic(messages)
        kwargs: dict = {"model": self.model, "max_tokens": max(max_tokens, 4096),
                        "system": system, "messages": msgs,
                        "output_config": {"effort": self.effort}}
        if tools:
            kwargs["tools"] = [{"name": t["function"]["name"],
                                "description": t["function"]["description"],
                                "input_schema": t["function"]["parameters"]} for t in tools]
        t0 = time.perf_counter()
        resp = self.client.messages.create(**kwargs)
        dt = time.perf_counter() - t0
        if resp.stop_reason == "refusal":
            return Completion(content="I'm sorry, I can't help with that here.", latency_s=dt,
                              prompt_tokens=resp.usage.input_tokens,
                              completion_tokens=resp.usage.output_tokens)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content
                 if b.type == "tool_use"]
        return Completion(content=text, tool_calls=calls, latency_s=dt,
                          prompt_tokens=resp.usage.input_tokens,
                          completion_tokens=resp.usage.output_tokens,
                          extra={"provider_blocks": [b.model_dump(exclude_none=True)
                                                     for b in resp.content]})
