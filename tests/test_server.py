from fastapi.testclient import TestClient

from supportsim.domain.env import BankEnv
from supportsim.providers.base import Completion, ToolCall
from supportsim.server.app import app


class Fake:
    name = "fake"

    def __init__(self):
        self.n = 0

    def complete(self, messages, tools=None, **_):
        self.n += 1
        card = next(iter(BankEnv().db["cards"]))
        if self.n == 1:
            return Completion("", [ToolCall("c1", "freeze_card", {"card_id": card})])
        return Completion("I need to verify you first.")


def test_chat_flow_with_guardrail_block():
    app.state.provider_factory = Fake
    c = TestClient(app)
    assert c.get("/healthz").json()["status"] == "ok"
    sid = c.post("/api/sessions").json()["session_id"]
    r = c.post(f"/api/sessions/{sid}/messages", json={"text": "freeze my card"}).json()
    assert r["reply"] == "I need to verify you first."
    assert r["tool_calls"][0]["blocked"] and r["tool_calls"][0]["rule"] == "IDV-REQUIRED"
    assert c.get(f"/api/sessions/{sid}/changes").json() == []
    names = {s["name"] for s in c.get(f"/api/sessions/{sid}/trace").json()}
    assert {"agent.turn", "llm.call", "tool.call"} <= names
    assert "supportsim_guardrail_blocks_total 1" in c.get("/metrics").text
    assert c.get("/").status_code == 200
    assert c.post("/api/sessions/nope/messages", json={"text": "hi"}).status_code == 404
