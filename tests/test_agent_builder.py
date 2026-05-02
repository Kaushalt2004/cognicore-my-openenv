"""Tests for cognicore.agent_builder — build_agent, describe_agent."""

from cognicore.agent_builder import build_agent, describe_agent
from cognicore.smart_agents import AutoLearner, SafeAgent, AdaptiveAgent


class TestBuildAgent:
    def test_build_default(self):
        agent = build_agent()
        assert agent is not None

    def test_build_safety_goal(self):
        agent = build_agent(goal="maximize safety")
        assert isinstance(agent, SafeAgent)

    def test_build_safe_goal(self):
        agent = build_agent(goal="be safe and careful")
        assert isinstance(agent, SafeAgent)

    def test_build_learning_goal(self):
        agent = build_agent(goal="maximize learning")
        assert isinstance(agent, AutoLearner)

    def test_build_fast_goal(self):
        agent = build_agent(goal="fast learning")
        assert isinstance(agent, AutoLearner)

    def test_build_cost_goal(self):
        agent = build_agent(goal="minimize cost")
        assert isinstance(agent, AutoLearner)

    def test_build_robust_goal(self):
        agent = build_agent(goal="robust against adversarial")
        assert isinstance(agent, SafeAgent)

    def test_build_balance_goal(self):
        agent = build_agent(goal="balance performance")
        assert isinstance(agent, AdaptiveAgent)

    def test_build_accuracy_goal(self):
        agent = build_agent(goal="maximize accuracy")
        assert isinstance(agent, AutoLearner)

    def test_build_unknown_goal(self):
        agent = build_agent(goal="something completely different")
        assert isinstance(agent, AdaptiveAgent)

    def test_build_config_stored(self):
        agent = build_agent(goal="maximize safety", risk_tolerance="low")
        assert hasattr(agent, "_build_config")
        assert agent._build_config["goal"] == "maximize safety"
        assert agent._build_config["risk_tolerance"] == "low"


class TestBuildAgentRiskTolerance:
    def test_low_risk_reduces_epsilon(self):
        agent_default = build_agent(goal="maximize learning", risk_tolerance="medium")
        agent_low = build_agent(goal="maximize learning", risk_tolerance="low")
        if hasattr(agent_default, "epsilon") and hasattr(agent_low, "epsilon"):
            assert agent_low.epsilon <= agent_default.epsilon

    def test_high_risk_increases_epsilon(self):
        agent_default = build_agent(goal="maximize learning", risk_tolerance="medium")
        agent_high = build_agent(goal="maximize learning", risk_tolerance="high")
        if hasattr(agent_default, "epsilon") and hasattr(agent_high, "epsilon"):
            assert agent_high.epsilon >= agent_default.epsilon

    def test_low_risk_conservative_threshold(self):
        agent = build_agent(goal="maximize safety", risk_tolerance="low")
        if hasattr(agent, "conservative_threshold"):
            assert agent.conservative_threshold == 0.8

    def test_medium_risk_conservative_threshold(self):
        agent = build_agent(goal="maximize safety", risk_tolerance="medium")
        if hasattr(agent, "conservative_threshold"):
            assert agent.conservative_threshold == 0.6


class TestDescribeAgent:
    def test_describe_built_agent(self):
        agent = build_agent(goal="maximize safety", risk_tolerance="low")
        desc = describe_agent(agent)
        assert desc["type"] == "SafeAgent"
        assert desc["goal"] == "maximize safety"
        assert desc["risk_tolerance"] == "low"

    def test_describe_unknown_agent(self):
        class PlainAgent:
            pass

        desc = describe_agent(PlainAgent())
        assert desc["type"] == "PlainAgent"
        assert desc["goal"] == "unknown"

    def test_describe_includes_epsilon(self):
        agent = build_agent(goal="maximize learning")
        desc = describe_agent(agent)
        assert "epsilon" in desc

    def test_describe_name(self):
        agent = build_agent(goal="maximize safety")
        desc = describe_agent(agent)
        assert "SafeAgent" in desc["name"]
