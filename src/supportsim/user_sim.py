"""LLM-simulated customer.

The simulator only sees its private instruction and the agent's replies (never the
tools, policy or database), and talks until its goal is met, refused or handed off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from supportsim.providers.base import Provider

STOP = "###STOP###"

USER_SYSTEM = """You write the CUSTOMER's side of a support chat with Aster Bank's \
virtual assistant. You are the customer, not the assistant: you ask for help, you never \
offer help, never ask the assistant to verify itself, and never mention tools or policies.

Who you are and what you want (private, reveal gradually):
{instruction}

Rules:
- Output ONLY the customer's next message: one or two short sentences, plain text, no \
"Customer:" prefix.
- Give a detail (email/phone, date of birth, ZIP, card, charge) only when the assistant \
asks for it or it is needed for your request.
- Use only facts from the description above. If asked for something you don't know (card \
number, transaction id, account number), say you don't know it. Never invent details.
- Do not ask for a human agent unless the description above tells you to.
- As soon as your request is completed, or the assistant clearly refuses or cannot help \
further, or you are transferred to a human, output exactly {stop} and nothing else. Do not \
keep thanking the assistant.
"""


def _render(history: list[tuple[str, str]]) -> str:
    lines = [f"{'Assistant' if who == 'agent' else 'Customer (you)'}: {text}"
             for who, text in history]
    return ("Chat so far:\n" + "\n".join(lines) +
            "\n\nWrite the customer's next message (or " + STOP + " if you are done).")


@dataclass
class SimulatedUser:
    """Small open models confuse roles when the chat is mirrored, so the simulator
    sees the conversation as a transcript and writes the next customer line."""

    provider: Provider
    instruction: str
    temperature: float = 0.7
    seed: int = 0
    max_tokens: int = 120
    history: list[tuple[str, str]] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0,
                                                 "llm_calls": 0, "llm_seconds": 0.0})

    def __post_init__(self) -> None:
        self.system = USER_SYSTEM.format(instruction=self.instruction, stop=STOP)

    def reply(self, agent_text: str) -> str:
        """Agent text in, next customer message out (may contain STOP)."""
        self.history.append(("agent", agent_text or "(no reply)"))
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": _render(self.history)}]
        comp = self.provider.complete(messages, None, temperature=self.temperature,
                                      seed=self.seed, max_tokens=self.max_tokens)
        self.usage["prompt_tokens"] += comp.prompt_tokens
        self.usage["completion_tokens"] += comp.completion_tokens
        self.usage["llm_calls"] += 1
        self.usage["llm_seconds"] += comp.latency_s
        self.seed += 1
        text = comp.content.strip()
        for prefix in ("Customer (you):", "Customer:", "customer:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        text = text.split("\nAssistant:")[0].strip().strip('"').strip()
        self.history.append(("customer", text))
        return text
