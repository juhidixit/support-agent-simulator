"""Local open-weight models through llama.cpp (GGUF), CPU or Metal.

The prompt is rendered with the model's *own* chat template (read from GGUF
metadata) including its native tool-calling format, so Qwen, Llama and others are
prompted the way they were trained. llama.cpp reuses the KV cache for the shared
prompt prefix between turns, which keeps multi-turn agent loops affordable on CPU.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from supportsim.providers.base import Completion, parse_tool_calls


def _template_messages(messages: list[dict]) -> list[dict]:
    """Internal messages -> the shape HF chat templates expect."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": [{"type": "function", "id": t["id"],
                                        "function": {"name": t["name"],
                                                     "arguments": t["arguments"]}}
                                       for t in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "name": m.get("name", ""),
                        "tool_call_id": m.get("tool_call_id", "")})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


class LlamaCppProvider:
    def __init__(self, model_path: str, *, n_ctx: int = 8192, n_threads: int | None = None,
                 name: str | None = None, verbose: bool = False) -> None:
        import jinja2
        from llama_cpp import Llama

        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         verbose=verbose, n_gpu_layers=-1)
        self.name = name or model_path.rsplit("/", 1)[-1].removesuffix(".gguf")
        meta = self.llm.metadata
        tmpl = meta.get("tokenizer.chat_template")
        if not tmpl:
            raise ValueError(f"{model_path} has no chat template in its metadata")
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                 loader=jinja2.BaseLoader())
        env.filters["tojson"] = lambda x, indent=None, **_: json.dumps(x, indent=indent,
                                                                      ensure_ascii=False)
        env.globals["raise_exception"] = self._raise
        env.globals["strftime_now"] = lambda fmt: datetime(2026, 11, 1).strftime(fmt)
        self.template = env.from_string(tmpl)
        vocab = self.llm._model
        self.bos = self.llm.detokenize([vocab.token_bos()], special=True).decode(errors="ignore")
        self.eos = self.llm.detokenize([vocab.token_eos()], special=True).decode(errors="ignore")
        self.stops = sorted({s for s in (self.eos, "<|im_end|>", "<|eot_id|>", "<|eom_id|>",
                                         "<|end|>", "<end_of_turn>") if s})

    @staticmethod
    def _raise(msg: str) -> None:
        raise ValueError(msg)

    def render(self, messages: list[dict], tools: list[dict] | None) -> str:
        return self.template.render(messages=_template_messages(messages), tools=tools or None,
                                    add_generation_prompt=True, bos_token=self.bos,
                                    eos_token=self.eos, date_string="01 Nov 2026")

    def complete(self, messages: list[dict], tools: list[dict] | None = None, *,
                 temperature: float = 0.0, seed: int | None = None,
                 max_tokens: int = 512) -> Completion:
        prompt = self.render(messages, tools)
        tokens = self.llm.tokenize(prompt.encode(), add_bos=False, special=True)
        if len(tokens) + max_tokens > self.llm.n_ctx():
            raise RuntimeError(f"context overflow: {len(tokens)} prompt tokens")
        t0 = time.perf_counter()
        out = self.llm.create_completion(prompt=tokens, max_tokens=max_tokens,
                                         temperature=temperature, top_p=0.95 if temperature else 1,
                                         seed=seed if seed is not None else 0, stop=self.stops)
        dt = time.perf_counter() - t0
        text = out["choices"][0]["text"]
        names = {t["function"]["name"] for t in tools or []}
        visible, calls = parse_tool_calls(text, names) if tools else (text.strip(), [])
        return Completion(content=visible, tool_calls=calls,
                          prompt_tokens=out["usage"]["prompt_tokens"],
                          completion_tokens=out["usage"]["completion_tokens"], latency_s=dt,
                          extra={"raw": text})
