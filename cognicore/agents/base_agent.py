"""
CogniCore BaseAgent — Abstract agent interface for any AI type.

Provides a contract that LLMs, RL agents, classifiers, rule-based
systems, or any other AI can implement to interact with CogniCore
environments.

Usage::

    from cognicore.agents import BaseAgent

    class MyAgent(BaseAgent):
        def act(self, observation):
            return {"classification": "SAFE"}

    agent = MyAgent()
    env = cognicore.make("SafetyClassification-v1")
    obs = env.reset()

    while True:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.on_reward(reward)
        if done:
            break
    agent.on_episode_end(env.episode_stats())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, runtime_checkable

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

from cognicore.core.types import EpisodeStats, StructuredReward


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol (duck-type) interface for CogniCore agents.

    Any object with an ``act()`` method satisfies this protocol.
    Use this for isinstance checks without requiring inheritance::

        def train(agent: AgentProtocol, env: CogniCoreEnv) -> None:
            if not isinstance(agent, AgentProtocol):
                raise AgentInterfaceError(agent, "act")
    """

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Choose an action given an observation."""
        ...


class BaseAgent(ABC):
    """Abstract base agent for CogniCore environments.

    Any AI system (LLM, RL, classifier, rule-based) can implement
    this interface to operate within a CogniCore environment.
    """

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Choose an action given an observation.

        Parameters
        ----------
        observation : dict
            The environment observation (includes memory context and
            reflection hints if enabled).

        Returns
        -------
        dict
            The action to take. Format depends on the environment.
            For SafetyClassification: ``{"classification": "SAFE"}``.
        """
        ...

    def on_reward(self, reward: StructuredReward) -> None:
        """Process a structured reward signal (optional).

        Override to implement learning from the 8-component reward.
        """
        pass

    def on_episode_end(self, stats: EpisodeStats) -> None:
        """Process end-of-episode statistics (optional).

        Override to implement end-of-episode learning.
        """
        pass

    def on_reflection_hint(self, hint: str) -> None:
        """Process a reflection hint (optional).

        Override if your agent can dynamically adjust based on hints.
        """
        pass

    def propose(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a tentative action for the PROPOSE→Revise protocol.

        Default implementation delegates to ``act()``. Override to
        implement a separate exploration strategy.
        """
        return self.act(observation)

    def revise(
        self,
        observation: Dict[str, Any],
        feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Revise an action based on proposal feedback.

        Default implementation ignores feedback and calls ``act()``.
        Override to implement feedback-driven revision.
        """
        return self.act(observation)


class RandomAgent(BaseAgent):
    """Agent that takes random actions (useful for testing)."""

    def __init__(self, action_space: Any = None) -> None:
        self._action_space = action_space

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self._action_space and hasattr(self._action_space, "sample"):
            sample = self._action_space.sample()
            if isinstance(sample, str):
                return {"classification": sample}
            return {"action": sample}
        # Default: random safety label
        import random

        label = random.choice(["SAFE", "UNSAFE", "NEEDS_REVIEW"])
        return {"classification": label}
