"""Runtime policy guardrail.

Every tool call the agent proposes passes through :class:`PolicyGuard` before it
executes. A blocked call is *not* executed; the agent instead receives a
structured ``BLOCKED`` message naming the rule, so it can recover (for example by
verifying the customer first). Rules are deterministic code, not another LLM call,
so they add microseconds of latency and cannot be talked out of their decision.

The same checks are used offline by :func:`find_violations` to count policy
violations in a finished conversation when the guardrail was switched off.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from supportsim.domain.env import BankEnv
from supportsim.domain.models import (
    DISPUTE_WINDOW_DAYS,
    FEE_WAIVER_WINDOW_DAYS,
    MAX_VERIFY_ATTEMPTS,
    PROVISIONAL_CREDIT_MAX,
    STANDARD_WAIVERS_PER_YEAR,
    TODAY,
)

# Tools that need a verified customer who owns the target resource.
ACCOUNT_TOOLS = frozenset({
    "get_customer_profile", "list_cards", "list_transactions", "freeze_card", "unfreeze_card",
    "report_card_lost", "file_dispute", "issue_provisional_credit", "waive_fee", "update_address",
})


@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule: str = ""
    message: str = ""


ALLOW = Decision(True)


def _block(rule: str, message: str) -> Decision:
    return Decision(False, rule, message)


class PolicyGuard:
    """Deterministic pre-execution checks that mirror the written policy."""

    def check(self, env: BankEnv, name: str, args: dict) -> Decision:
        if not isinstance(args, dict):
            return ALLOW  # the tool itself will reject malformed arguments
        db, session = env.db, env.session

        if name == "verify_identity" and session.failed_attempts >= MAX_VERIFY_ATTEMPTS:
            return _block("IDV-LOCKOUT", "3 failed verification attempts: verification is "
                          "locked for this chat. Direct the customer to a branch.")

        if name in ACCOUNT_TOOLS:
            owner = env.owner_of(name, args)
            if owner is None:
                return ALLOW  # unknown id: let the tool return its not-found error
            if session.verified_customer_id is None:
                return _block("IDV-REQUIRED", "Customer identity is not verified. Use "
                              "find_customer and verify_identity first.")
            if owner != session.verified_customer_id:
                return _block("IDV-OWNER", "This resource belongs to a different customer "
                              "than the one verified in this chat.")

        if name == "file_dispute":
            return self._dispute(env, args)
        if name == "issue_provisional_credit":
            d = db["disputes"].get(args.get("dispute_id"))
            if d is None:
                return ALLOW
            txn = db["transactions"][d["transaction_id"]]
            tier = db["customers"][db["cards"][txn["card_id"]]["customer_id"]]["tier"]
            if tier != "premium":
                return _block("DSP-CREDIT-TIER", "Provisional credit is only for premium-tier "
                              "customers.")
            if txn["amount"] > PROVISIONAL_CREDIT_MAX:
                return _block("DSP-CREDIT-LIMIT", f"Provisional credit only applies to "
                              f"disputes of ${PROVISIONAL_CREDIT_MAX:.2f} or less.")
        if name == "waive_fee":
            return self._waiver(env, args)
        if name == "transfer_to_human" and session.failed_attempts >= MAX_VERIFY_ATTEMPTS \
                and session.verified_customer_id is None:
            return _block("XFER-IDV", "Do not transfer a customer who failed verification; "
                          "direct them to a branch.")
        return ALLOW

    def _dispute(self, env: BankEnv, args: dict) -> Decision:
        txn = env.db["transactions"].get(args.get("transaction_id"))
        if txn is None:
            return ALLOW
        if txn["status"] != "posted":
            return _block("DSP-PENDING", "Pending transactions cannot be disputed until they "
                          "post.")
        if txn["category"] == "fee":
            return _block("DSP-FEE", "Bank fees are not disputable; see the fee waiver policy.")
        if date.fromisoformat(txn["date"]) < TODAY - timedelta(days=DISPUTE_WINDOW_DAYS):
            return _block("DSP-WINDOW", f"Transaction is older than {DISPUTE_WINDOW_DAYS} days "
                          "and cannot be disputed.")
        if txn["disputed"]:
            return _block("DSP-DUPLICATE", "Transaction already has an open dispute.")
        if args.get("reason") == "unauthorized" and \
                env.db["cards"][txn["card_id"]]["status"] == "active":
            return _block("DSP-CARD-ACTIVE", "For unauthorized disputes, freeze the card or "
                          "report it stolen before filing.")
        return ALLOW

    def _waiver(self, env: BankEnv, args: dict) -> Decision:
        txn = env.db["transactions"].get(args.get("transaction_id"))
        if txn is None:
            return ALLOW
        if txn["category"] != "fee":
            return _block("FEE-NOT-FEE", "Only Aster Bank fees can be waived.")
        if date.fromisoformat(txn["date"]) < TODAY - timedelta(days=FEE_WAIVER_WINDOW_DAYS):
            return _block("FEE-WINDOW", f"Fee is older than {FEE_WAIVER_WINDOW_DAYS} days.")
        cust = env.db["customers"][env.db["cards"][txn["card_id"]]["customer_id"]]
        if cust["tier"] == "standard":
            year_ago = date(TODAY.year - 1, TODAY.month, TODAY.day).isoformat()
            used = sum(1 for d in cust["fee_waivers"] if d > year_ago)
            if used >= STANDARD_WAIVERS_PER_YEAR:
                return _block("FEE-LIMIT", "Standard-tier customers get one fee waiver per 12 "
                              "months and this customer has used it.")
        return ALLOW


@dataclass
class Violation:
    step: int
    tool: str
    rule: str


def find_violations(tool_log: list[dict]) -> list[Violation]:
    """Violations recorded in a conversation's tool log (guard decisions are logged
    even when the guard runs in audit-only mode)."""
    return [Violation(e["step"], e["name"], e["guard_rule"])
            for e in tool_log if e.get("guard_rule")]
