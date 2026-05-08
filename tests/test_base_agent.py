"""Tests for cognicore.agents.base_agent — BaseAgent, AgentProtocol, RandomAgent."""

import pytest
import cognicore
from cognicore.agents.base_agent import BaseAgent, AgentProtocol, RandomAgent


class _MinimalAgent(BaseAgent):
    """Minimal concrete implementation of BaseAgent for testing."""

    def act(self, observation):
        return {"classification": "SAFE"}


class _TrackingAgent(BaseAgent):
    """Agent that records all lifecycle callbacks."""

    def __init__(self):
        self.rewards = []
        self.episode_ends = []
        self.hints = []

    def act(self, observation):
        return {"classification": "SAFE"}

    def on_reward(self, reward):
        self.rewards.append(reward)

    def on_episode_end(self, stats):
        self.episode_ends.append(stats)

    def on_reflection_hint(self, hint):
        self.hints.append(hint)


class TestBaseAgentAbstract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseAgent()

    def test_concrete_subclass_instantiates(self):
        agent = _MinimalAgent()
        assert agent is not None

    def test_act_returns_dict(self):
        agent = _MinimalAgent()
        result = agent.act({"step": 0, "prompt": "test"})
        assert isinstance(result, dict)


class TestBaseAgentLifecycle:
    def test_on_reward_default_no_raise(self):
        agent = _MinimalAgent()
        agent.on_reward(None)  # should not raise

    def test_on_episode_end_default_no_raise(self):
        agent = _MinimalAgent()
        agent.on_episode_end(None)  # should not raise

    def test_on_reflection_hint_default_no_raise(self):
        agent = _MinimalAgent()
        agent.on_reflection_hint("think harder")  # should not raise

    def test_tracking_agent_receives_rewards(self):
        agent = _TrackingAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        for _ in range(10):
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            agent.on_reward(reward)
            if done:
                agent.on_episode_end(env.episode_stats())
                break
        assert len(agent.rewards) == 10
        assert len(agent.episode_ends) == 1

    def test_tracking_agent_receives_hints(self):
        agent = _TrackingAgent()
        agent.on_reflection_hint("consider the context")
        assert len(agent.hints) == 1
        assert agent.hints[0] == "consider the context"


class TestBaseAgentProposeRevise:
    def test_propose_delegates_to_act(self):
        agent = _MinimalAgent()
        obs = {"prompt": "test", "step": 0}
        result = agent.propose(obs)
        assert result == agent.act(obs)

    def test_revise_delegates_to_act(self):
        agent = _MinimalAgent()
        obs = {"prompt": "test", "step": 0}
        result = agent.revise(obs, {"feedback": "reconsider"})
        assert result == agent.act(obs)

    def test_propose_revise_can_be_overridden(self):
        class SmartAgent(BaseAgent):
            def act(self, obs):
                return {"classification": "SAFE"}

            def propose(self, obs):
                return {"classification": "UNSAFE", "tentative": True}

            def revise(self, obs, feedback):
                return {"classification": "SAFE", "revised": True}

        agent = SmartAgent()
        proposal = agent.propose({"step": 0})
        assert proposal["tentative"] is True
        revision = agent.revise({"step": 0}, {})
        assert revision["revised"] is True


class TestAgentProtocol:
    def test_object_with_act_satisfies_protocol(self):
        class Duck:
            def act(self, obs):
                return {}

        assert isinstance(Duck(), AgentProtocol)

    def test_object_without_act_fails_protocol(self):
        class NoDuck:
            def predict(self, obs):
                return {}

        assert not isinstance(NoDuck(), AgentProtocol)

    def test_base_agent_subclass_satisfies_protocol(self):
        assert isinstance(_MinimalAgent(), AgentProtocol)


class TestRandomAgent:
    def test_act_returns_dict(self):
        agent = RandomAgent()
        result = agent.act({})
        assert isinstance(result, dict)
        assert "classification" in result

    def test_act_valid_labels(self):
        agent = RandomAgent()
        valid = {"SAFE", "UNSAFE", "NEEDS_REVIEW"}
        for _ in range(30):
            result = agent.act({})
            assert result["classification"] in valid

    def test_random_agent_is_base_agent(self):
        agent = RandomAgent()
        assert isinstance(agent, BaseAgent)

    def test_random_agent_is_protocol(self):
        agent = RandomAgent()
        assert isinstance(agent, AgentProtocol)

    def test_random_agent_full_episode(self):
        agent = RandomAgent()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        steps = 0
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            steps += 1
        assert steps == 10
