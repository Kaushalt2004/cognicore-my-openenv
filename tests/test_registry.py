"""Tests for the CogniCore environment registry."""

import pytest

import cognicore
from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult
from cognicore.envs.registry import register, make, list_envs, _REGISTRY


class TestRegistry:
    """Registry make / register / list."""

    def test_list_envs_has_safety(self):
        envs = cognicore.list_envs()
        ids = [e["id"] for e in envs]
        assert "SafetyClassification-v1" in ids
        assert "SafetyClassification-Easy-v1" in ids
        assert "SafetyClassification-Medium-v1" in ids
        assert "SafetyClassification-Hard-v1" in ids

    def test_make_valid_env(self):
        env = cognicore.make("SafetyClassification-v1")
        assert isinstance(env, CogniCoreEnv)

    def test_make_invalid_env(self):
        with pytest.raises(KeyError, match="not found"):
            cognicore.make("NonexistentEnv-v99")

    def test_register_custom_env(self):
        """Register and instantiate a custom env at runtime."""

        class DummyEnv(CogniCoreEnv):
            def _setup(self, **kw):
                pass

            def _generate_tasks(self):
                return [{"q": "test", "category": "dummy"}]

            def _evaluate(self, action):
                return EvalResult(base_score=1.0, correct=True, category="dummy")

            def _get_obs(self):
                return {"q": "test"}

        cognicore.register(
            "DummyTest-v1",
            entry_point=DummyEnv,
            description="Test env",
        )

        env = cognicore.make("DummyTest-v1")
        assert isinstance(env, DummyEnv)

        obs = env.reset()
        assert obs["q"] == "test"

        obs, reward, done, truncated, info = env.step({"answer": "yes"})
        assert done is True
        assert reward.base_score == 1.0

        # Cleanup
        del _REGISTRY["DummyTest-v1"]

    def test_make_with_kwargs_override(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="hard")
        assert env.difficulty == "hard"
