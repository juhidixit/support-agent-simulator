"""Domain constants shared by the tools, the guardrail and the task generator."""

from __future__ import annotations

from datetime import date

# The simulation clock is frozen so every run sees the same "today".
TODAY = date(2026, 11, 1)

DISPUTE_WINDOW_DAYS = 60
FEE_WAIVER_WINDOW_DAYS = 90
STANDARD_WAIVERS_PER_YEAR = 1
PROVISIONAL_CREDIT_MAX = 500.0
MAX_VERIFY_ATTEMPTS = 3

DISPUTE_REASONS = ("unauthorized", "duplicate", "not_received", "incorrect_amount")
LOST_REASONS = ("lost", "stolen")
TRANSFER_REASONS = ("out_of_scope", "customer_request", "other")
