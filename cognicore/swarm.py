"""
CogniCore Swarm Intelligence — Local multi-agent swarm simulation.

Multiple agents share a global memory pool and collaborate to solve
tasks better than any individual agent.

Usage::

    from cognicore.swarm import Swarm

    swarm = Swarm(size=5)
    result = swarm.solve("SafetyClassification-v1", episodes=3)
    result.print_report()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import cognicore
from cognicore.smart_agents import AutoLearner, SafeAgent, AdaptiveAgent


class SharedMemory:
    """Global shared memory pool for swarm agents."""

    def __init__(self):
        self.knowledge: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.contributions: Dict[str, int] = defaultdict(int)

    def contribute(self, agent_name: str, category: str, action: str, correct: bool):
        """Agent shares what it learned with the swarm."""
        if correct:
            self.knowledge[category][action] += 1.0
        else:
            self.knowledge[category][action] -= 0.3
        self.contributions[agent_name] += 1

    def consult(self, category: str) -> Optional[str]:
        """Ask the swarm for the best action for a category."""
        if category not in self.knowledge:
            return None
        actions = self.knowledge[category]
        if not actions:
            return None
        return max(actions, key=actions.get)

    def stats(self) -> Dict[str, Any]:
        return {
            "categories": len(self.knowledge),
            "total_contributions": sum(self.contributions.values()),
            "top_contributors": dict(
                sorted(self.contributions.items(), key=lambda x: -x[1])[:5]
            ),
        }


class SwarmAgent:
    """An agent that participates in a swarm."""

    def __init__(self, name: str, inner, shared: SharedMemory):
        self.name = name
        self.inner = inner
        self.shared = shared
        self.correct = 0
        self.total = 0

    def act(self, obs: Dict) -> Dict:
        """Act using both personal knowledge and swarm knowledge."""
        category = obs.get("category", "")

        # Check swarm consensus first
        swarm_suggestion = self.shared.consult(category)
        if swarm_suggestion and hasattr(self.inner, "epsilon"):
            # Trust swarm with some probability
            import random
            if random.random() > self.inner.epsilon:
                return {"classification": swarm_suggestion}

        return self.inner.act(obs)

    def learn(self, reward, info, obs):
        """Learn and share with the swarm."""
        if hasattr(self.inner, "learn"):
            self.inner.learn(reward, info)

        er = info.get("eval_result", {})
        correct = er.get("correct", False)
        category = er.get("category", "")
        action = str(er.get("predicted", ""))

        self.total += 1
        if correct:
            self.correct += 1

        # Share knowledge with the swarm
        self.shared.contribute(self.name, category, action, correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0


class SwarmResult:
    """Results from a swarm session."""

    def __init__(self, agents: List[SwarmAgent], shared: SharedMemory):
        self.agents = agents
        self.shared = shared

    @property
    def best_agent(self) -> SwarmAgent:
        return max(self.agents, key=lambda a: a.accuracy)

    @property
    def avg_accuracy(self) -> float:
        return sum(a.accuracy for a in self.agents) / len(self.agents) if self.agents else 0

    def print_report(self):
        print(f"\n{'=' * 60}")
        print(f"  Swarm Intelligence Report")
        print(f"  {len(self.agents)} agents | avg accuracy: {self.avg_accuracy:.0%}")
        print(f"{'=' * 60}")

        for a in sorted(self.agents, key=lambda x: -x.accuracy):
            bar_len = int(a.accuracy * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {a.name:15s} [{bar}] {a.accuracy:.0%} ({a.correct}/{a.total})")

        mem = self.shared.stats()
        print(f"\n  Shared Memory: {mem['categories']} categories, {mem['total_contributions']} contributions")
        if mem["top_contributors"]:
            print(f"  Top contributors: {', '.join(f'{k}({v})' for k, v in mem['top_contributors'].items())}")
        print(f"{'=' * 60}\n")


class Swarm:
    """Multi-agent swarm with shared global memory.

    Parameters
    ----------
    size : int
        Number of agents in the swarm.
    diversity : bool
        If True, use different agent types for diversity.
    """

    def __init__(self, size: int = 5, diversity: bool = True):
        self.size = size
        self.diversity = diversity
        self.shared = SharedMemory()
        self.agents: List[SwarmAgent] = []

        # Create diverse swarm
        agent_types = [AutoLearner, SafeAgent, AdaptiveAgent]
        for i in range(size):
            if diversity:
                cls = agent_types[i % len(agent_types)]
            else:
                cls = AutoLearner

            inner = cls()
            name = f"agent_{i+1}"
            self.agents.append(SwarmAgent(name, inner, self.shared))

    def solve(
        self,
        env_id: str = "SafetyClassification-v1",
        difficulty: str = "easy",
        episodes: int = 1,
        verbose: bool = True,
    ) -> SwarmResult:
        """Run the swarm on an environment.

        Each agent takes turns solving episodes, sharing knowledge
        through shared memory.
        """
        if verbose:
            print(f"\n  Swarm: {self.size} agents on {env_id}")
            print(f"  Episodes per agent: {episodes}")
            print(f"  {'─' * 50}")

        for agent in self.agents:
            for ep in range(episodes):
                env = cognicore.make(env_id, difficulty=difficulty)
                obs = env.reset()

                while True:
                    action = agent.act(obs)
                    obs, reward, done, _, info = env.step(action)
                    agent.learn(reward, info, obs)
                    if done:
                        break

                if verbose:
                    stats = env.episode_stats()
                    print(f"  {agent.name} ep{ep+1}: accuracy={stats.accuracy:.0%}")

        return SwarmResult(self.agents, self.shared)
