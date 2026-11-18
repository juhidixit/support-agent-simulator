from supportsim.cli import main
from supportsim.domain.env import BankEnv
from supportsim.evaluate import grade
from supportsim.metrics import pass_hat_k, summarize
from supportsim.taskgen import build
from supportsim.tasks import load_tasks


def test_task_suite_is_reproducible():
    assert [t.to_dict() for t in build(11)] == [t.to_dict() for t in load_tasks()]


def test_suite_size_and_unique_ids():
    tasks = load_tasks()
    assert len(tasks) >= 60
    assert len({t.id for t in tasks}) == len(tasks)


def test_validate_command_passes():
    assert main(["validate"]) == 0


def test_oracle_actions_pass_and_doing_nothing_fails_write_tasks():
    for t in load_tasks():
        env = BankEnv()
        for a in t.actions:
            env.execute(a.name, dict(a.args))
        texts = list(t.outputs)  # an oracle reply that states every required fact
        assert grade(t, env, texts).passed, t.id
        idle = grade(t, BankEnv(), [])
        if "write" in t.tags or "handoff" in t.tags or t.outputs:
            assert not idle.passed, t.id
        else:
            assert idle.passed, t.id


def test_failure_types():
    t = next(t for t in load_tasks() if t.category == "card_freeze")
    assert grade(t, BankEnv(), []).failure_type == "missing_write"
    env = BankEnv()
    for a in t.actions:
        env.execute(a.name, dict(a.args))
    env.transfer_to_human("other", "x")
    assert grade(t, env, []).failure_type == "unexpected_write"


def _r(task, trial, passed):
    return {"task_id": task, "trial": trial, "passed": passed, "category": "c",
            "violations": [], "turns": 3, "tool_calls": 2,
            "agent_usage": {"prompt_tokens": 10, "completion_tokens": 5, "llm_seconds": 1.0}}


def test_pass_hat_k_matches_definition():
    rs = [_r("a", 0, True), _r("a", 1, True), _r("a", 2, False),
          _r("b", 0, True), _r("b", 1, True), _r("b", 2, True)]
    assert abs(pass_hat_k(rs, 1) - (2 / 3 + 1) / 2) < 1e-9
    assert abs(pass_hat_k(rs, 2) - (1 / 3 + 1) / 2) < 1e-9
    assert pass_hat_k(rs, 3) == 0.5
    s = summarize(rs)
    assert s["trials_per_task"] == 3 and s["pass_hat_k"][3] == 0.5
