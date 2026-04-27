"""
CogniCore Agent Builder — Autonomous agent creation.

Automatically selects architecture, configures memory, and tunes
strategy based on a high-level goal.

Usage::

    from cognicore.agent_builder import build_agent

    agent = build_agent(goal="maximize safety classification accuracy")
    agent = build_agent(goal="minimize cost", model="gemini-flash")
"""

from __future__ import annotations

from typing import Any, Dict

from cognicore.smart_agents import AutoLearner, SafeAgent, AdaptiveAgent


def build_agent(
    goal: str = "maximize accuracy",
    model: str = "",
    risk_tolerance: str = "medium",
    memory_size: int = 1000,
    **kwargs,
) -> Any:
    """Automatically build an agent based on a high-level goal.

    Parameters
    ----------
    goal : str
        High-level objective. Supported goals:
        - "maximize accuracy" / "maximize safety" → SafeAgent
        - "maximize learning" / "fast learning" → AutoLearner
        - "balance" / "adaptive" → AdaptiveAgent
        - "minimize cost" → AutoLearner with conservative settings
        - "robust" / "adversarial" → SafeAgent with high threshold
    model : str
        Optional model hint (for LLM-backed agents).
    risk_tolerance : str
        "low", "medium", "high"
    """
    goal_lower = goal.lower()

    # Determine best agent type
    if any(w in goal_lower for w in ["safety", "safe", "conservative", "careful"]):
        agent = SafeAgent(
            conservative_threshold=0.8 if risk_tolerance == "low" else 0.6
        )
        agent.name = "SafeAgent (built for safety)"

    elif any(w in goal_lower for w in ["learn", "fast", "speed", "quick"]):
        agent = AutoLearner()
        agent.epsilon = 0.3 if risk_tolerance == "high" else 0.15
        agent.name = "AutoLearner (built for fast learning)"

    elif any(w in goal_lower for w in ["robust", "adversarial", "attack", "defend"]):
        agent = SafeAgent(conservative_threshold=0.9)
        agent.name = "SafeAgent (built for robustness)"

    elif any(w in goal_lower for w in ["cost", "cheap", "budget", "efficient"]):
        agent = AutoLearner()
        agent.epsilon = 0.05  # minimal exploration
        agent.name = "AutoLearner (built for cost efficiency)"

    elif any(w in goal_lower for w in ["balance", "adaptive", "general", "all"]):
        agent = AdaptiveAgent()
        agent.name = "AdaptiveAgent (built for balance)"

    elif any(w in goal_lower for w in ["accuracy", "correct", "precise", "best"]):
        agent = AutoLearner()
        agent.epsilon = 0.1
        agent.name = "AutoLearner (built for max accuracy)"

    else:
        agent = AdaptiveAgent()
        agent.name = f"AdaptiveAgent (goal: {goal[:30]})"

    # Apply risk tolerance
    if risk_tolerance == "low":
        if hasattr(agent, "epsilon"):
            agent.epsilon = max(0.05, agent.epsilon * 0.5)
    elif risk_tolerance == "high":
        if hasattr(agent, "epsilon"):
            agent.epsilon = min(0.5, agent.epsilon * 2)

    agent._build_config = {
        "goal": goal,
        "model": model,
        "risk_tolerance": risk_tolerance,
        "agent_type": type(agent).__name__,
    }

    return agent


def describe_agent(agent) -> Dict[str, Any]:
    """Describe a built agent's configuration."""
    config = getattr(agent, "_build_config", {})
    return {
        "name": getattr(agent, "name", type(agent).__name__),
        "type": type(agent).__name__,
        "goal": config.get("goal", "unknown"),
        "risk_tolerance": config.get("risk_tolerance", "unknown"),
        "epsilon": getattr(agent, "epsilon", None),
        "threshold": getattr(agent, "threshold", None),
        "knowledge_categories": len(getattr(agent, "knowledge", {})),
    }
