from datetime import date, timedelta

from supportsim.domain.models import TODAY
from supportsim.guardrails import PolicyGuard

G = PolicyGuard()


def _card(env, cid, status="active"):
    return next(k for k in env.db["cards"].values()
                if k["customer_id"] == cid and k["status"] == status)


def test_account_tools_require_verification(env):
    card = _card(env, "C1001")
    d = G.check(env, "freeze_card", {"card_id": card["card_id"]})
    assert not d.allowed and d.rule == "IDV-REQUIRED"


def test_cannot_act_on_another_customers_card(verified):
    env, _ = verified
    other = _card(env, "C1002")
    d = G.check(env, "freeze_card", {"card_id": other["card_id"]})
    assert not d.allowed and d.rule == "IDV-OWNER"


def test_lockout_after_three_failures(env):
    for _ in range(3):
        env.verify_identity("C1003", "1900-01-01", "00000")
    assert G.check(env, "verify_identity", {"customer_id": "C1003", "dob": "x",
                                            "zip": "y"}).rule == "IDV-LOCKOUT"
    assert G.check(env, "transfer_to_human", {"reason": "other", "summary": "s"}).rule \
        == "XFER-IDV"


def _txn_for(env, cid, pred):
    cards = {k["card_id"] for k in env.db["cards"].values() if k["customer_id"] == cid}
    return next((t for t in env.db["transactions"].values()
                 if t["card_id"] in cards and pred(t)), None)


def test_dispute_rules(verified):
    env, _ = verified
    cutoff = (TODAY - timedelta(days=60)).isoformat()
    old = _txn_for(env, "C1001", lambda t: t["category"] != "fee" and t["date"] < cutoff)
    if old:
        assert G.check(env, "file_dispute", {"transaction_id": old["transaction_id"],
                                             "reason": "duplicate"}).rule == "DSP-WINDOW"
    fee = _txn_for(env, "C1001", lambda t: t["category"] == "fee")
    if fee:
        assert G.check(env, "file_dispute", {"transaction_id": fee["transaction_id"],
                                             "reason": "duplicate"}).rule == "DSP-FEE"
    ok = _txn_for(env, "C1001", lambda t: t["category"] != "fee" and t["date"] >= cutoff
                  and t["status"] == "posted" and not t["disputed"])
    assert G.check(env, "file_dispute", {"transaction_id": ok["transaction_id"],
                                         "reason": "not_received"}).allowed
    d = G.check(env, "file_dispute", {"transaction_id": ok["transaction_id"],
                                      "reason": "unauthorized"})
    card = env.db["cards"][ok["card_id"]]
    assert d.allowed == (card["status"] != "active")


def test_pending_transactions_cannot_be_disputed(env):
    t = next(t for t in env.db["transactions"].values() if t["status"] == "pending")
    owner = env.db["cards"][t["card_id"]]["customer_id"]
    c = env.db["customers"][owner]
    env.verify_identity(owner, c["dob"], c["address"]["zip"])
    assert G.check(env, "file_dispute", {"transaction_id": t["transaction_id"],
                                         "reason": "incorrect_amount"}).rule == "DSP-PENDING"


def test_fee_waiver_limit_for_standard_tier(env):
    year_ago = date(TODAY.year - 1, TODAY.month, TODAY.day).isoformat()
    for c in env.db["customers"].values():
        cards = {k["card_id"] for k in env.db["cards"].values()
                 if k["customer_id"] == c["customer_id"]}
        fee = next((t for t in env.db["transactions"].values() if t["card_id"] in cards
                    and t["category"] == "fee"
                    and t["date"] >= (TODAY - timedelta(days=90)).isoformat()), None)
        if fee is None or c["tier"] != "standard":
            continue
        env.session.verified_customer_id = c["customer_id"]
        used = sum(1 for d in c["fee_waivers"] if d > year_ago)
        d = G.check(env, "waive_fee", {"transaction_id": fee["transaction_id"]})
        assert d.allowed == (used == 0), c["customer_id"]


def test_provisional_credit_only_premium_under_limit(env):
    for t in env.db["transactions"].values():
        if t["category"] == "fee" or t["disputed"] or t["status"] != "posted":
            continue
        owner = env.db["cards"][t["card_id"]]["customer_id"]
        tier = env.db["customers"][owner]["tier"]
        env.session.verified_customer_id = owner
        did = env.file_dispute(t["transaction_id"], "not_received")["dispute_id"]
        d = G.check(env, "issue_provisional_credit", {"dispute_id": did})
        assert d.allowed == (tier == "premium" and t["amount"] <= 500)
