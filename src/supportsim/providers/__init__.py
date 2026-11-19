"""Model backends. ``make_provider`` turns a CLI spec string into a provider:

- ``llamacpp:/path/model.gguf``                   local open-weight model (llama.cpp)
- ``openai:gpt-4.1-mini``                         OpenAI (``OPENAI_API_KEY``)
- ``openai:qwen2.5:7b@http://localhost:11434/v1`` any OpenAI-compatible server
- ``anthropic:claude-opus-5-5``                   Claude (``ANTHROPIC_API_KEY``)
"""

from __future__ import annotations

from functools import cache

from supportsim.providers.base import Completion, Provider, ToolCall

__all__ = ["Completion", "Provider", "ToolCall", "make_provider"]


@cache
def _llamacpp(path: str, slot: str):
    import os

    from supportsim.providers.llamacpp import LlamaCppProvider

    threads = int(os.environ.get("SUPPORTSIM_THREADS", "0")) or None
    return LlamaCppProvider(path, n_threads=threads)


def make_provider(spec: str, *, slot: str = "agent"):
    """`slot` keeps separate llama.cpp instances (and KV caches) for the agent and
    the simulated user even when both use the same model file."""
    kind, _, rest = spec.partition(":")
    if kind == "llamacpp":
        return _llamacpp(rest, slot)
    if kind == "openai":
        from supportsim.providers.openai_compat import OpenAICompatProvider

        model, _, base = rest.partition("@")
        return OpenAICompatProvider(model, base or "https://api.openai.com/v1")
    if kind == "anthropic":
        from supportsim.providers.anthropic_provider import DEFAULT_MODEL, AnthropicProvider

        return AnthropicProvider(rest or DEFAULT_MODEL)
    raise ValueError(f"unknown provider spec {spec!r}")
