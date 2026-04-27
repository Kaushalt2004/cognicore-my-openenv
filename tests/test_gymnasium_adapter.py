"""Tests for the Gymnasium adapter."""

import cognicore
from cognicore.adapters.gymnasium import GymnasiumAdapter


class TestGymnasiumAdapter:
    def test_wrap_env(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        assert gym_env.unwrapped is env

    def test_reset_returns_tuple(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        obs, info = gym_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "step" in obs

    def test_step_returns_float_reward(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        gym_env.reset()
        obs, reward, terminated, truncated, info = gym_env.step(
            {"classification": "SAFE"}
        )
        assert isinstance(reward, float)
        assert "structured_reward" in info

    def test_reward_mode_total(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env, reward_mode="total")
        gym_env.reset()
        _, reward, _, _, info = gym_env.step({"classification": "SAFE"})
        assert reward == info["structured_reward"]["total"]

    def test_reward_mode_base(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env, reward_mode="base")
        gym_env.reset()
        _, reward, _, _, info = gym_env.step({"classification": "SAFE"})
        assert reward == info["structured_reward"]["base_score"]

    def test_full_episode(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        obs, info = gym_env.reset()

        done = False
        steps = 0
        total_reward = 0.0
        while not done:
            obs, reward, done, truncated, info = gym_env.step(
                {"classification": "SAFE"}
            )
            total_reward += reward
            steps += 1

        assert steps == 10
        assert done is True
        assert total_reward > 0

    def test_propose_revise(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        gym_env.reset()

        feedback = gym_env.propose({"classification": "UNSAFE"})
        assert hasattr(feedback, "confidence_estimate")

        obs, reward, done, truncated, info = gym_env.revise({"classification": "SAFE"})
        assert isinstance(reward, float)

    def test_state_proxy(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        gym_env.reset()
        gym_env.step({"classification": "SAFE"})
        state = gym_env.state()
        assert "accuracy" in state
        assert "memory_stats" in state

    def test_repr(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        r = repr(gym_env)
        assert "GymnasiumAdapter" in r
        assert "SafetyClassificationEnv" in r

    def test_close(self):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        gym_env = GymnasiumAdapter(env)
        gym_env.close()  # should not raise
