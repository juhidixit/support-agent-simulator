"""Agent loop tests with a scripted fake model (no LLM needed)."""

from supportsim.agent import SupportAgent
from supportsim.domain.env import BankEnv
from supportsim.providers.base import Completion, ToolCall, parse_tool_calls
from supportsim.runner import run_conversation
from supportsim.tasks import load_tasks
from supportsim.tracing import Tracer, redact
from supportsim.user_sim import STOP


class Scripted:
    """Returns queued completions in order."""

    name = "scripted"

    def __init__(self, steps):
        self.steps = list(steps)

    def complete(self, messages, tools=None, **_):
        return self.steps.pop(0)


def call(name, **args):
    return Completion("", [ToolCall(f"id_{name}", name, args)])


def say(text):
    return Completion(text)


def test_guardrail_blocks_unverified_write_and_agent_sees_reason():
    env = BankEnv()
    card = next(iter(env.db["cards"]))
    agent = SupportAgent(Scripted([call("freeze_card", card_id=card), say("ok")]), env)
    agent.respond("freeze my card")
    assert env.db["cards"][card]["status"] == BankEnv().db["cards"][card]["status"]
    assert agent.tool_log[0]["blocked"] and agent.tool_log[0]["guard_rule"] == "IDV-REQUIRED"
    assert "BLOCKED" in agent.messages[-2]["content"]


def test_guardrail_off_executes_but_still_records_violation():
    env = BankEnv()
    card = next(k for k, v in env.db["cards"].items() if v["status"] == "active")
    agent = SupportAgent(Scripted([call("freeze_card", card_id=card), say("done")]), env,
                         guardrail=False)
    agent.respond("freeze my card")
    assert env.db["cards"][card]["status"] == "frozen"
    assert agent.tool_log[0]["guard_rule"] == "IDV-REQUIRED" and not agent.tool_log[0]["blocked"]


def test_full_scripted_conversation_passes_grading():
    task = next(t for t in load_tasks() if t.category == "card_freeze")
    steps = [call(a.name, **a.args) for a in task.actions] + [say("Your card is frozen.")]
    user = Scripted([say("Please freeze my card."), say(STOP)])
    r = run_conversation(task, Scripted(steps), user)
    assert r["passed"] and r["end_reason"] == "customer_done" and r["violations"] == []


def test_transfer_ends_conversation():
    task = next(t for t in load_tasks() if t.category == "transfer_requested")
    agent = Scripted([call("transfer_to_human", reason="customer_request", summary="x")])
    r = run_conversation(task, agent, Scripted([say("Human please.")]))
    assert r["end_reason"] == "handoff" and r["passed"]


def test_parse_hermes_and_llama_tool_calls():
    names = {"find_customer"}
    text, calls = parse_tool_calls(
        'Let me check.\n<tool_call>\n{"name": "find_customer", "arguments": '
        '{"email": "a@b.com"}}\n</tool_call>', names)
    assert text == "Let me check." and calls[0].arguments == {"email": "a@b.com"}
    text, calls = parse_tool_calls(
        '<|python_tag|>{"name": "find_customer", "parameters": {"phone": "1"}}', names)
    assert calls[0].name == "find_customer" and calls[0].arguments == {"phone": "1"}
    text, calls = parse_tool_calls('Here is JSON {"name": "x"} for you', names)
    assert calls == []


def test_tracing_redacts_pii():
    tr = Tracer()
    with tr.span("t", email="a@b.com", note="call 555-123-4567", dob="1990-01-01"):
        pass
    attrs = tr.spans[0]["attributes"]
    assert attrs["email"] == "[REDACTED]" and "555" not in attrs["note"]
    assert redact({"x": "born 1990-01-01"})["x"] == "born [DATE]"
