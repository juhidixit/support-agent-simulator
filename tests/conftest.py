import pytest

from supportsim.domain.env import BankEnv


@pytest.fixture
def env() -> BankEnv:
    return BankEnv()


@pytest.fixture
def verified(env: BankEnv):
    """An env with customer C1001 verified, plus that customer's record."""
    c = env.db["customers"]["C1001"]
    assert env.verify_identity("C1001", c["dob"], c["address"]["zip"])["verified"]
    return env, c
