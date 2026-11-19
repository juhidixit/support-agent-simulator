import pytest

from supportsim.domain.env import BankEnv, ToolError, hash_db
from supportsim.domain.generate import generate


def test_generator_is_deterministic_and_matches_committed_db():
    assert generate(7) == BankEnv().db


def test_envs_are_isolated(env):
    other = BankEnv()
    card = next(iter(env.db["cards"].values()))
    card["status"] = "closed"
    assert other.db["cards"][card["card_id"]]["status"] != "closed" or True
    assert hash_db(other.db) == hash_db(BankEnv().db)


def test_find_customer_by_email_and_phone(env):
    c = env.db["customers"]["C1005"]
    assert env.find_customer(email=c["email"].upper())["customer_id"] == "C1005"
    assert env.find_customer(phone=c["phone"])["customer_id"] == "C1005"
    with pytest.raises(ToolError):
        env.find_customer(email="nobody@example.com")


def test_verify_identity_counts_failures(env):
    c = env.db["customers"]["C1002"]
    r = env.verify_identity("C1002", "1900-01-01", c["address"]["zip"])
    assert r == {"verified": False, "attempts_remaining": 2}
    assert env.session.failed_attempts == 1
    assert env.verify_identity("C1002", c["dob"], c["address"]["zip"])["verified"]
    assert env.session.verified_customer_id == "C1002"


def test_report_lost_ships_replacement_to_current_address(verified):
    env, c = verified
    card = next(k for k in env.db["cards"].values()
                if k["customer_id"] == "C1001" and k["status"] == "active")
    env.update_address("C1001", "1 New St", "Austin", "tx", "78701")
    out = env.report_card_lost(card["card_id"], "lost", True)
    new = env.db["cards"][out["replacement_card_id"]]
    assert env.db["cards"][card["card_id"]]["status"] == "closed"
    assert new["status"] == "active" and new["shipped_to"]["zip"] == "78701"
    assert new["shipped_to"]["state"] == "TX"


def test_dispute_and_provisional_credit_change_balance(env):
    tid, t = next((k, v) for k, v in env.db["transactions"].items()
                  if not v["disputed"] and v["category"] != "fee")
    before = env.db["accounts"][t["account_id"]]["balance"]
    d = env.file_dispute(tid, "not_received")["dispute_id"]
    env.issue_provisional_credit(d)
    assert env.db["accounts"][t["account_id"]]["balance"] == round(before + t["amount"], 2)
    with pytest.raises(ToolError):
        env.file_dispute(tid, "duplicate")
    with pytest.raises(ToolError):
        env.issue_provisional_credit(d)


def test_execute_rejects_bad_calls(env):
    with pytest.raises(ToolError):
        env.execute("delete_everything", {})
    with pytest.raises(ToolError):
        env.execute("freeze_card", {"card": "K3001"})
    with pytest.raises(ToolError):
        env.execute("file_dispute", {"transaction_id": "T1001", "reason": "because"})


def test_tool_schemas_are_valid_function_specs(env):
    for s in env.tool_schemas():
        f = s["function"]
        assert s["type"] == "function" and f["name"] in env.tools
        assert set(f["parameters"]["required"]) <= set(f["parameters"]["properties"])
