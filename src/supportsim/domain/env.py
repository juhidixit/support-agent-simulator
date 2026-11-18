"""Tool environment for the Aster Bank support domain.

The environment owns a private copy of the database plus per-conversation
session state (who has been identity-verified). Tools enforce *data integrity*
only (unknown ids, closing a closed card); business *policy* lives in
:mod:`supportsim.guardrails` and in the written policy given to the agent. That
split is deliberate: it lets the evaluation measure how often an agent breaks
policy on its own, with and without the runtime guardrail.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from typing import Any

from supportsim.domain.models import (
    DISPUTE_REASONS,
    LOST_REASONS,
    MAX_VERIFY_ATTEMPTS,
    TODAY,
    TRANSFER_REASONS,
)


class ToolError(Exception):
    """Raised when a tool call is invalid; the message is returned to the agent."""


def load_base_db() -> dict:
    text = resources.files("supportsim.data").joinpath("bank_db.json").read_text()
    return json.loads(text)


def load_policy() -> str:
    return resources.files("supportsim.data").joinpath("policy.md").read_text()


@dataclass
class Session:
    verified_customer_id: str | None = None
    failed_attempts: int = 0


# Tools that change the database. Read-only tools are everything else.
WRITE_TOOLS = frozenset({
    "freeze_card", "unfreeze_card", "report_card_lost", "file_dispute",
    "issue_provisional_credit", "waive_fee", "update_address", "transfer_to_human",
})


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        """OpenAI-style function schema (also accepted by most chat templates)."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = {"type": "string"}


@dataclass
class BankEnv:
    db: dict = field(default_factory=load_base_db)
    session: Session = field(default_factory=Session)

    def __post_init__(self) -> None:
        self.db = copy.deepcopy(self.db)
        self.tools: dict[str, ToolSpec] = {t.name: t for t in self._build_tools()}

    # ------------------------------------------------------------------ helpers
    def owner_of(self, name: str, args: dict) -> str | None:
        """Customer that a tool call targets, or None if it cannot be resolved."""
        try:
            if "customer_id" in args and name not in ("transfer_to_human",):
                return args["customer_id"] if args["customer_id"] in self.db["customers"] else None
            if "card_id" in args:
                return self.db["cards"][args["card_id"]]["customer_id"]
            if "transaction_id" in args:
                card = self.db["transactions"][args["transaction_id"]]["card_id"]
                return self.db["cards"][card]["customer_id"]
            if "dispute_id" in args:
                tid = self.db["disputes"][args["dispute_id"]]["transaction_id"]
                return self.owner_of("x", {"transaction_id": tid})
        except (KeyError, TypeError):
            return None
        return None

    def _get(self, table: str, key: Any) -> dict:
        if not isinstance(key, str) or key not in self.db[table]:
            raise ToolError(f"{table[:-1]} '{key}' not found")
        return self.db[table][key]

    def _account_for_txn(self, txn: dict) -> dict:
        return self.db["accounts"][txn["account_id"]]

    def state_hash(self) -> str:
        return hash_db(self.db)

    # ------------------------------------------------------------------- tools
    def find_customer(self, email: str | None = None, phone: str | None = None) -> dict:
        if not email and not phone:
            raise ToolError("provide email or phone")
        for c in self.db["customers"].values():
            if (email and c["email"].lower() == str(email).strip().lower()) or (
                phone and c["phone"] == str(phone).strip()
            ):
                return {"customer_id": c["customer_id"], "name": c["name"]}
        raise ToolError("no customer matches that email or phone")

    def verify_identity(self, customer_id: str, dob: str, zip: str) -> dict:  # noqa: A002
        c = self._get("customers", customer_id)
        ok = str(dob).strip() == c["dob"] and str(zip).strip() == c["address"]["zip"]
        if ok:
            self.session.verified_customer_id = customer_id
            self.session.failed_attempts = 0
            return {"verified": True}
        self.session.failed_attempts += 1
        left = max(0, MAX_VERIFY_ATTEMPTS - self.session.failed_attempts)
        return {"verified": False, "attempts_remaining": left}

    def get_customer_profile(self, customer_id: str) -> dict:
        c = self._get("customers", customer_id)
        accounts = [a for a in self.db["accounts"].values() if a["customer_id"] == customer_id]
        year_ago = date(TODAY.year - 1, TODAY.month, TODAY.day).isoformat()
        return {
            "customer_id": customer_id, "name": c["name"], "email": c["email"],
            "phone": c["phone"], "address": c["address"], "tier": c["tier"],
            "fee_waivers_last_12_months": sum(1 for d in c["fee_waivers"] if d > year_ago),
            "accounts": [{"account_id": a["account_id"], "type": a["type"],
                          "balance": a["balance"]} for a in accounts],
        }

    def list_cards(self, customer_id: str) -> list[dict]:
        self._get("customers", customer_id)
        return [{k: v for k, v in card.items() if k != "customer_id"}
                for card in self.db["cards"].values() if card["customer_id"] == customer_id]

    def list_transactions(self, card_id: str) -> list[dict]:
        self._get("cards", card_id)
        keep = ("transaction_id", "date", "merchant", "category", "amount", "status",
                "disputed", "waived")
        rows = [{k: t[k] for k in keep} for t in self.db["transactions"].values()
                if t["card_id"] == card_id]
        return sorted(rows, key=lambda t: t["date"], reverse=True)

    def freeze_card(self, card_id: str) -> dict:
        card = self._get("cards", card_id)
        if card["status"] != "active":
            raise ToolError(f"card is {card['status']}, only active cards can be frozen")
        card["status"] = "frozen"
        return {"card_id": card_id, "status": "frozen"}

    def unfreeze_card(self, card_id: str) -> dict:
        card = self._get("cards", card_id)
        if card["status"] != "frozen":
            raise ToolError(f"card is {card['status']}, only frozen cards can be unfrozen")
        card["status"] = "active"
        return {"card_id": card_id, "status": "active"}

    def report_card_lost(self, card_id: str, reason: str, ship_replacement: bool) -> dict:
        card = self._get("cards", card_id)
        if reason not in LOST_REASONS:
            raise ToolError(f"reason must be one of {LOST_REASONS}")
        if card["status"] == "closed":
            raise ToolError("card is already closed")
        card["status"] = "closed"
        card["closed_reason"] = reason
        out: dict = {"card_id": card_id, "status": "closed"}
        if _as_bool(ship_replacement):
            new_id = f"{card_id}R"
            cust = self.db["customers"][card["customer_id"]]
            last4 = f"{int(hashlib.sha256(new_id.encode()).hexdigest(), 16) % 10000:04d}"
            self.db["cards"][new_id] = {
                "card_id": new_id, "customer_id": card["customer_id"],
                "account_id": card["account_id"], "type": card["type"], "last4": last4,
                "status": "active", "shipped_to": copy.deepcopy(cust["address"]),
            }
            out["replacement_card_id"] = new_id
            out["shipped_to"] = cust["address"]
            out["arrives_in_business_days"] = 5
        return out

    def file_dispute(self, transaction_id: str, reason: str) -> dict:
        txn = self._get("transactions", transaction_id)
        if reason not in DISPUTE_REASONS:
            raise ToolError(f"reason must be one of {DISPUTE_REASONS}")
        did = f"D-{transaction_id}"
        if did in self.db["disputes"]:
            raise ToolError(f"transaction already has dispute {did}")
        txn["disputed"] = True
        self.db["disputes"][did] = {"dispute_id": did, "transaction_id": transaction_id,
                                    "reason": reason, "status": "open",
                                    "provisional_credit": False}
        return {"dispute_id": did, "status": "open", "decision_within_business_days": 10}

    def issue_provisional_credit(self, dispute_id: str) -> dict:
        d = self._get("disputes", dispute_id)
        if d["provisional_credit"]:
            raise ToolError("provisional credit already issued")
        txn = self.db["transactions"][d["transaction_id"]]
        acct = self._account_for_txn(txn)
        acct["balance"] = round(acct["balance"] + txn["amount"], 2)
        d["provisional_credit"] = True
        return {"dispute_id": dispute_id, "credited": txn["amount"], "new_balance": acct["balance"]}

    def waive_fee(self, transaction_id: str) -> dict:
        txn = self._get("transactions", transaction_id)
        if txn["waived"]:
            raise ToolError("fee already waived")
        acct = self._account_for_txn(txn)
        acct["balance"] = round(acct["balance"] + txn["amount"], 2)
        txn["waived"] = True
        self.db["customers"][acct["customer_id"]]["fee_waivers"].append(TODAY.isoformat())
        return {"transaction_id": transaction_id, "refunded": txn["amount"],
                "new_balance": acct["balance"]}

    def update_address(self, customer_id: str, street: str, city: str, state: str,
                       zip: str) -> dict:  # noqa: A002
        c = self._get("customers", customer_id)
        c["address"] = {"street": str(street).strip(), "city": str(city).strip(),
                        "state": str(state).strip().upper(), "zip": str(zip).strip()}
        return {"customer_id": customer_id, "address": c["address"]}

    def transfer_to_human(self, reason: str, summary: str) -> dict:
        if reason not in TRANSFER_REASONS:
            raise ToolError(f"reason must be one of {TRANSFER_REASONS}")
        self.db["handoffs"].append({"reason": reason, "summary": str(summary)})
        return {"transferred": True, "queue_position": 3}

    # ----------------------------------------------------------------- registry
    def _build_tools(self) -> list[ToolSpec]:
        cid = {"customer_id": {**_S, "description": "Customer id, e.g. C1001"}}
        card = {"card_id": {**_S, "description": "Card id, e.g. K3001"}}
        txn = {"transaction_id": {**_S, "description": "Transaction id, e.g. T1042"}}
        return [
            ToolSpec("find_customer", "Look up a customer by email or phone. Returns the "
                     "customer id and name only.",
                     _obj({"email": _S, "phone": {**_S, "description": "Format 555-555-5555"}}, []),
                     self.find_customer),
            ToolSpec("verify_identity", "Verify the customer's identity with date of birth "
                     "(YYYY-MM-DD) and billing ZIP code.",
                     _obj({**cid, "dob": _S, "zip": _S}, ["customer_id", "dob", "zip"]),
                     self.verify_identity),
            ToolSpec("get_customer_profile", "Get profile, tier, fee-waiver count and account "
                     "balances for a verified customer.", _obj(cid, ["customer_id"]),
                     self.get_customer_profile),
            ToolSpec("list_cards", "List a customer's cards with status.",
                     _obj(cid, ["customer_id"]), self.list_cards),
            ToolSpec("list_transactions", "List transactions on a card, newest first.",
                     _obj(card, ["card_id"]), self.list_transactions),
            ToolSpec("freeze_card", "Temporarily freeze an active card. Reversible.",
                     _obj(card, ["card_id"]), self.freeze_card),
            ToolSpec("unfreeze_card", "Unfreeze a frozen card.", _obj(card, ["card_id"]),
                     self.unfreeze_card),
            ToolSpec("report_card_lost", "Permanently close a lost or stolen card and "
                     "optionally ship a replacement to the address on file.",
                     _obj({**card, "reason": {"type": "string", "enum": list(LOST_REASONS)},
                           "ship_replacement": {"type": "boolean"}},
                          ["card_id", "reason", "ship_replacement"]),
                     self.report_card_lost),
            ToolSpec("file_dispute", "Open a dispute on a card transaction.",
                     _obj({**txn, "reason": {"type": "string", "enum": list(DISPUTE_REASONS)}},
                          ["transaction_id", "reason"]), self.file_dispute),
            ToolSpec("issue_provisional_credit", "Credit the disputed amount to the account "
                     "while the dispute is investigated.",
                     _obj({"dispute_id": {**_S, "description": "e.g. D-T1042"}}, ["dispute_id"]),
                     self.issue_provisional_credit),
            ToolSpec("waive_fee", "Refund a fee transaction to the customer's account.",
                     _obj(txn, ["transaction_id"]), self.waive_fee),
            ToolSpec("update_address", "Replace the customer's mailing address.",
                     _obj({**cid, "street": _S, "city": _S, "state": {**_S, "description":
                           "2-letter code"}, "zip": _S},
                          ["customer_id", "street", "city", "state", "zip"]),
                     self.update_address),
            ToolSpec("transfer_to_human", "Hand the conversation to a human agent. Ends the chat.",
                     _obj({"reason": {"type": "string", "enum": list(TRANSFER_REASONS)},
                           "summary": {**_S, "description": "One-paragraph case summary"}},
                          ["reason", "summary"]), self.transfer_to_human),
        ]

    def tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def execute(self, name: str, args: dict) -> Any:
        """Run a tool. Raises ToolError for invalid calls."""
        if name not in self.tools:
            raise ToolError(f"unknown tool '{name}'")
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        try:
            return self.tools[name].fn(**args)
        except TypeError as e:  # wrong/missing argument names
            raise ToolError(f"bad arguments for {name}: {e}") from None


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def normalize_db(db: dict) -> dict:
    """Comparable view of the database: free-text handoff summaries are dropped."""
    out = copy.deepcopy(db)
    out["handoffs"] = [h["reason"] != "" for h in out["handoffs"]]  # count only
    return out


def hash_db(db: dict) -> str:
    blob = json.dumps(normalize_db(db), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
