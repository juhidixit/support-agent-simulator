"""Lightweight tracing with OpenTelemetry-style spans and PII redaction.

Spans are written as JSON lines using OpenTelemetry field names (trace_id, span_id,
parent_span_id, start/end in unix nanoseconds) and GenAI semantic-convention
attribute names (``gen_ai.request.model``, ``gen_ai.usage.input_tokens``,
``gen_ai.tool.name``), so they can be loaded into an OTel backend or viewed in the
bundled web UI. Customer identifiers are redacted before anything is written.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PII_KEYS = {"dob", "zip", "email", "phone", "street"}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
_DOB = re.compile(r"\b(19|20)\d\d-\d\d-\d\d\b")


def redact(obj: Any) -> Any:
    """Mask emails, phone numbers and identity fields in nested data.

    Dates are only masked under PII keys or in free text that also mentions a
    birth date, so transaction dates stay readable in traces."""
    if isinstance(obj, dict):
        return {k: ("[REDACTED]" if k in PII_KEYS and v else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        s = _PHONE.sub("[PHONE]", _EMAIL.sub("[EMAIL]", obj))
        if re.search(r"birth|dob|born", s, re.I):
            s = _DOB.sub("[DATE]", s)
        return s
    return obj


class Span:
    def __init__(self, tracer: Tracer, name: str, trace_id: str, parent: str | None,
                 attributes: dict) -> None:
        self.tracer = tracer
        self.name = name
        self.trace_id = trace_id
        self.span_id = secrets.token_hex(8)
        self.parent = parent
        self.attributes = dict(attributes)
        self.start = time.time_ns()
        self.end: int | None = None
        self.status = "OK"

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "span_id": self.span_id,
                "parent_span_id": self.parent, "name": self.name,
                "start_time_unix_nano": self.start, "end_time_unix_nano": self.end,
                "status": self.status, "attributes": redact(self.attributes)}


class Tracer:
    """Collects spans for one or more conversations; optionally appends to a file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.spans: list[dict] = []
        self._stack = threading.local()
        self._lock = threading.Lock()

    def _current(self) -> list[Span]:
        if not hasattr(self._stack, "s"):
            self._stack.s = []
        return self._stack.s

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        stack = self._current()
        parent = stack[-1] if stack else None
        trace_id = parent.trace_id if parent else secrets.token_hex(16)
        sp = Span(self, name, trace_id, parent.span_id if parent else None, attributes)
        stack.append(sp)
        try:
            yield sp
        except Exception as e:
            sp.status = "ERROR"
            sp.set(error=f"{type(e).__name__}: {e}")
            raise
        finally:
            sp.end = time.time_ns()
            stack.pop()
            rec = sp.to_dict()
            with self._lock:
                self.spans.append(rec)
                if self.path:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")

    def trace(self, trace_id: str) -> list[dict]:
        return sorted((s for s in self.spans if s["trace_id"] == trace_id),
                      key=lambda s: s["start_time_unix_nano"])


NULL_TRACER = Tracer()
