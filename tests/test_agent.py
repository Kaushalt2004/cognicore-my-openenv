"""Tests for cognicore.agents — BaseAgent, AgentProtocol, RandomAgent."""

import cognicore
from cognicore.agents import BaseAgent
from cognicore.agents.base_agent import AgentProtocol, RandomAgent


class TestBaseAgentInterface:
    def test_base_agent_is_abstract(self):
        import pytest
        with pytest.raises(TypeError):
            BaseAgent()  # Cannot instantiate abstract class

    def test_concrete_agent_requires_act(self):
        class ConcreteAgent(BaseAgent):
            def act(self, observation):
                return {"classification": "SAFE"}

        agent = ConcreteAgent()
        assert callable(agent.act)

    def test_base_agent_default_on_reward(self):
        class ConcreteAgent(BaseAgent):
            def act(self, obs):
                return {"classification": "SAFE"}

        agent = ConcreteAgent()
        # on_reward is optional — should not raise
        agent.on_reward(None)

    def test_base_agent_default_on_episode_end(self):
        class ConcreteAgent(BaseAgent):
            def act(self, obs):
                return {"classification": "SAFE"}

        agent = ConcreteAgent()
        agent.on_episode_end(None)

    def test_base_agent_default_propose_delegates_to_act(self):
        class ConcreteAgent(BaseAgent):
            def act(self, obs):
                return {"classification": "SAFE"}

        agent = ConcreteAgent()
        obs = {"prompt": "test", "step": 0}
        result = agent.propose(obs)
        assert result == {"classification": "SAFE"}

    def test_base_agent_default_revise_delegates_to_act(self):
        class ConcreteAgent(BaseAgent):
            def act(self, obs):
                return {"classification": "UNSAFE"}

        agent = ConcreteAgent()
        obs = {"prompt": "test", "step": 0}
        result = agent.revise(obs, {"feedback": "reconsider"})
        assert result == {"classification": "UNSAFE"}


class TestAgentProtocol:
    def test_any_class_with_act_satisfies_protocol(self):
        class DuckAgent:
            def act(self, obs):
                return {"classification": "SAFE"}

        agent = DuckAgent()
        assert isinstance(agent, AgentProtocol)

    def test_class_without_act_fails_protocol(self):
        class NotAnAgent:
            def predict(self, obs):
                return {}

        obj = NotAnAgent()
        assert not isinstance(obj, AgentProtocol)


class TestRandomAgent:
    def test_random_agent_act_returns_dict(self):
        agent = RandomAgent()
        obs = {"prompt": "test", "step": 0}
        action = agent.act(obs)
        assert isinstance(action, dict)

    def test_random_agent_returns_valid_label(self):
        agent = RandomAgent()
        obs = {}
        for _ in range(20):
            action = agent.act(obs)
            assert action["classification"] in ("SAFE", "UNSAFE", "NEEDS_REVIEW")

    def test_random_agent_with_action_space(self):
        import random

        class FakeSpace:
            def sample(self):
                return random.choice(["SAFE", "UNSAFE"])

        agent = RandomAgent(action_space=FakeSpace())
        action = agent.act({})
        assert "classification" in action

    def test_random_agent_in_env(self):
        agent = RandomAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        done = False
        for _ in range(10):
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            if done:
                break
        assert done is True
