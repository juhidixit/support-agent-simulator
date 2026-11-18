"""The support agent: a tool-calling loop around any provider.

One call to :meth:`SupportAgent.respond` handles one customer message. The model
may call several tools (each checked by the policy guardrail before it runs)
before it writes the reply that goes back to the customer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from supportsim.domain.env import WRITE_TOOLS, BankEnv, ToolError, load_policy
from supportsim.guardrails import PolicyGuard
from supportsim.providers.base import Provider
from supportsim.tracing import NULL_TRACER, Tracer

GREETING = "Hi! I'm the Aster Bank virtual assistant. How can I help you today?"

AGENT_RULES = """
## How to work
- You talk to the customer in plain text. To act, call one of the provided tools.
- Tool results are JSON. If a result contains "error", read it and adjust; never claim an
  action succeeded unless its tool result confirms it.
- Ask for the information you need (email or phone, date of birth, ZIP code) before acting.
"""


@dataclass
class AgentReply:
    text: str
    ended: bool = False  # True when the conversation was handed to a human


@dataclass
class SupportAgent:
    provider: Provider
    env: BankEnv
    guardrail: bool = True
    tracer: Tracer = NULL_TRACER
    temperature: float = 0.0
    seed: int = 0
    max_steps: int = 10
    max_tokens: int = 400
    messages: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0,
                                                 "llm_calls": 0, "llm_seconds": 0.0})

    def __post_init__(self) -> None:
        self.guard = PolicyGuard()
        if not self.messages:
            self.messages = [{"role": "system", "content": load_policy() + AGENT_RULES},
                             {"role": "assistant", "content": GREETING}]
        self.tools = self.env.tool_schemas()

    # ------------------------------------------------------------------ tools
    def _run_tool(self, name: str, args: Any) -> tuple[Any, dict]:
        entry: dict = {"step": len(self.tool_log), "name": name, "arguments": args,
                       "guard_rule": None, "blocked": False, "error": None}
        with self.tracer.span("tool.call", **{"gen_ai.tool.name": name,
                                              "tool.arguments": args}) as sp:
            decision = self.guard.check(self.env, name, args if isinstance(args, dict) else {})
            if not decision.allowed:
                entry["guard_rule"] = decision.rule
                sp.set(**{"guardrail.rule": decision.rule,
                          "guardrail.action": "block" if self.guardrail else "audit"})
            if not decision.allowed and self.guardrail:
                entry["blocked"] = True
                result: Any = {"error": f"BLOCKED by policy ({decision.rule}): "
                                        f"{decision.message}"}
            else:
                try:
                    result = self.env.execute(name, args)
                except ToolError as e:
                    entry["error"] = str(e)
                    result = {"error": str(e)}
            entry["write"] = name in WRITE_TOOLS and not entry["blocked"] and not entry["error"]
            sp.set(**{"tool.blocked": entry["blocked"], "tool.error": entry["error"],
                      "tool.result": result})
        self.tool_log.append(entry)
        return result, entry

    # ------------------------------------------------------------------- loop
    def respond(self, user_text: str) -> AgentReply:
        self.messages.append({"role": "user", "content": user_text})
        with self.tracer.span("agent.turn", **{"user.message": user_text}) as turn:
            for _ in range(self.max_steps):
                with self.tracer.span("llm.call", **{"gen_ai.request.model": self.provider.name,
                                                     "gen_ai.operation.name": "chat",
                                                     "llm.role": "agent"}) as sp:
                    comp = self.provider.complete(self.messages, self.tools,
                                                  temperature=self.temperature, seed=self.seed,
                                                  max_tokens=self.max_tokens)
                    sp.set(**{"gen_ai.usage.input_tokens": comp.prompt_tokens,
                              "gen_ai.usage.output_tokens": comp.completion_tokens,
                              "llm.output": comp.content,
                              "llm.tool_calls": [t.name for t in comp.tool_calls]})
                self.usage["prompt_tokens"] += comp.prompt_tokens
                self.usage["completion_tokens"] += comp.completion_tokens
                self.usage["llm_calls"] += 1
                self.usage["llm_seconds"] += comp.latency_s
                msg: dict = {"role": "assistant", "content": comp.content}
                if comp.tool_calls:
                    msg["tool_calls"] = [t.to_dict() for t in comp.tool_calls]
                if "provider_blocks" in comp.extra:
                    msg["provider_blocks"] = comp.extra["provider_blocks"]
                self.messages.append(msg)
                if not comp.tool_calls:
                    turn.set(**{"agent.reply": comp.content})
                    return AgentReply(comp.content)
                ended = False
                for call in comp.tool_calls:
                    result, entry = self._run_tool(call.name, call.arguments)
                    self.messages.append({"role": "tool", "tool_call_id": call.id,
                                          "name": call.name,
                                          "content": json.dumps(result, default=str)})
                    if call.name == "transfer_to_human" and entry["write"]:
                        ended = True
                if ended:
                    text = comp.content or "I've transferred you to a member of our team."
                    turn.set(**{"agent.reply": text, "agent.handoff": True})
                    return AgentReply(text, ended=True)
            text = "Sorry, I'm having trouble completing that. Let me try again."
            self.messages.append({"role": "assistant", "content": text})
            turn.set(**{"agent.reply": text, "agent.step_limit": True})
            return AgentReply(text)
