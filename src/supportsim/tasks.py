"""Task (scenario) definitions and loading.

A task gives the simulated customer a private instruction (who they are, what they
want, what they know) and gives the grader the *expected actions*: the tool calls a
policy-compliant agent would make. Replaying those actions on a fresh database
produces the expected end state; an agent passes when its own conversation leaves
the database in exactly that state and it told the customer every required fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib import resources


@dataclass
class Action:
    name: str
    args: dict


@dataclass
class Task:
    id: str
    category: str
    instruction: str
    actions: list[Action]
    outputs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(id=d["id"], category=d["category"], instruction=d["instruction"],
                   actions=[Action(**a) for a in d["actions"]], outputs=d.get("outputs", []),
                   tags=d.get("tags", []))

    def to_dict(self) -> dict:
        return asdict(self)


def load_tasks(path: str | None = None) -> list[Task]:
    if path:
        with open(path) as f:
            raw = json.load(f)
    else:
        raw = json.loads(resources.files("supportsim.data").joinpath("tasks.json").read_text())
    return [Task.from_dict(d) for d in raw]
