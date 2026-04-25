"""
CogniCore Swarm Intelligence — Local multi-agent swarm simulation.

Multiple agents share a global memory pool and collaborate to solve
tasks better than any individual agent.

Usage::

    from cognicore.swarm import Swarm, SwarmEnv

    # Sequential swarm (one agent at a time)
    swarm = Swarm(size=5)
    result = swarm.solve("SafetyClassification-v1", episodes=3)
    result.print_report()

    # Parallel swarm (all agents act simultaneously in a shared environment)
    class MySwarmEnv(SwarmEnv):
        def _setup(self, **kw): ...
        def _generate_tasks(self): ...
        def _evaluate_multi(self, actions): ...
        def _get_obs_for_agent(self, agent_id): ...

    swarm = Swarm(size=3)
    result = swarm.solve_parallel(MySwarmEnv(num_agents=3), episodes=2)
    result.print_report()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import cognicore
from cognicore.smart_agents import AutoLearner, SafeAgent, AdaptiveAgent
from cognicore.multi_agent import MultiAgentEnv
from cognicore.core.types import CogniCoreConfig


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


class SwarmEnv(MultiAgentEnv):
    """Multi-agent environment enhanced with swarm intelligence.

    Extends :class:`~cognicore.multi_agent.MultiAgentEnv` so all agents
    share a global :class:`SharedMemory` pool.  After every resolved step
    each agent's result is automatically contributed to that pool, and the
    swarm's collective best-known action for the current task category is
    injected into every agent's observation as a ``swarm_suggestion`` key.

    Subclasses must implement the same four abstract methods as
    :class:`~cognicore.multi_agent.MultiAgentEnv`:

    - ``_setup(**kwargs)``
    - ``_generate_tasks()``
    - ``_evaluate_multi(actions)``
    - ``_get_obs_for_agent(agent_id)``

    Parameters
    ----------
    num_agents : int
        Number of agents that will act in the environment.
    agent_ids : list of str, optional
        Custom agent identifiers.  Defaults to ``["agent_0", "agent_1", …]``.
    config : CogniCoreConfig, optional
        Middleware configuration passed to the base environment.
    **kwargs
        Forwarded to ``_setup()``.

    Example
    -------
    ::

        from cognicore.swarm import Swarm, SwarmEnv

        class MyEnv(SwarmEnv):
            def _setup(self, **kw): ...
            def _generate_tasks(self): return [...]
            def _evaluate_multi(self, actions): ...
            def _get_obs_for_agent(self, agent_id): ...

        swarm = Swarm(size=3)
        result = swarm.solve_parallel(MyEnv(num_agents=3), episodes=2)
        result.print_report()
    """

    def __init__(
        self,
        num_agents: int = 3,
        agent_ids: Optional[List[str]] = None,
        config: Optional[CogniCoreConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.shared_memory = SharedMemory()
        super().__init__(num_agents=num_agents, agent_ids=agent_ids, config=config, **kwargs)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """Reset environment and inject swarm context into initial observations."""
        multi_obs = super().reset(**kwargs)
        for aid in self.agent_ids:
            if aid in multi_obs:
                multi_obs[aid]["swarm_stats"] = self.shared_memory.stats()
        return multi_obs

    def _resolve_step(self) -> Dict[str, Any]:
        """Evaluate all agents, update shared memory, and return results."""
        results = super()._resolve_step()

        # Contribute each agent's outcome to the shared memory pool
        for aid in self.agent_ids:
            if aid not in results:
                continue
            er = results[aid].get("eval_result", {})
            category = er.get("category", "")
            predicted = str(er.get("predicted", ""))
            correct = er.get("correct", False)
            if category:
                self.shared_memory.contribute(aid, category, predicted, correct)

        return results

    # ------------------------------------------------------------------
    # New public helpers
    # ------------------------------------------------------------------

    def get_swarm_obs(self, agent_id: str) -> Dict[str, Any]:
        """Return per-agent observation enriched with swarm knowledge.

        Calls :meth:`_get_obs_for_agent` and then appends:

        - ``swarm_suggestion`` — the action the swarm collectively found
          most successful for the current task category (if any).
        - ``swarm_stats`` — a snapshot of shared-memory statistics.

        Parameters
        ----------
        agent_id : str
            The ID of the agent requesting the observation.

        Returns
        -------
        dict
            Enriched observation dictionary.
        """
        obs = self._get_obs_for_agent(agent_id)
        category = obs.get("category", "")
        if category:
            suggestion = self.shared_memory.consult(category)
            if suggestion is not None:
                obs["swarm_suggestion"] = suggestion
        obs["swarm_stats"] = self.shared_memory.stats()
        return obs

    def swarm_stats(self) -> Dict[str, Any]:
        """Return swarm-level statistics from the shared memory pool.

        Returns
        -------
        dict
            Dictionary with ``categories``, ``total_contributions``, and
            ``top_contributors``.
        """
        return self.shared_memory.stats()


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

    def solve_parallel(
        self,
        swarm_env: SwarmEnv,
        episodes: int = 1,
        verbose: bool = True,
    ) -> SwarmResult:
        """Run the swarm in a :class:`SwarmEnv` (parallel mode).

        All agents act *simultaneously* each step — one agent per slot in
        the environment — sharing knowledge through the environment's
        :class:`SharedMemory` pool.  After every episode the pool is merged
        into this :class:`Swarm`'s own :attr:`shared` memory so results
        remain available via :class:`SwarmResult`.

        Parameters
        ----------
        swarm_env : SwarmEnv
            A :class:`SwarmEnv` instance whose ``num_agents`` matches
            ``self.size``.
        episodes : int
            Number of episodes to run.
        verbose : bool
            Print per-episode progress.

        Returns
        -------
        SwarmResult
            Aggregated results across all agents and episodes.

        Raises
        ------
        TypeError
            If *swarm_env* is not a :class:`SwarmEnv` instance.
        ValueError
            If the number of agent slots in *swarm_env* does not match
            ``self.size``.
        """
        if not isinstance(swarm_env, SwarmEnv):
            raise TypeError(
                f"swarm_env must be a SwarmEnv instance, got {type(swarm_env).__name__}"
            )
        if len(swarm_env.agent_ids) != self.size:
            raise ValueError(
                f"SwarmEnv has {len(swarm_env.agent_ids)} agent slots but "
                f"Swarm has {self.size} agents"
            )

        if verbose:
            print(f"\n  Swarm (parallel): {self.size} agents")
            print(f"  Environment: {type(swarm_env).__name__}")
            print(f"  Episodes: {episodes}")
            print(f"  {'─' * 50}")

        for ep in range(episodes):
            obs_map = swarm_env.reset()
            done = False

            while not done:
                last_results: Optional[Dict[str, Any]] = None

                # All agents submit their actions in turn; evaluation happens
                # once the last agent has acted (inside step_agent).
                for agent, aid in zip(self.agents, swarm_env.agent_ids):
                    obs = obs_map.get(aid, {})
                    # Enrich obs with swarm suggestion before acting
                    category = obs.get("category", "")
                    if category:
                        suggestion = swarm_env.shared_memory.consult(category)
                        if suggestion is not None:
                            obs["swarm_suggestion"] = suggestion

                    action = agent.act(obs)
                    result = swarm_env.step_agent(aid, action)

                    # step_agent returns the full results dict once the last
                    # agent has acted (contains "_step" and "_done").
                    if "_step" in result:
                        last_results = result

                if last_results is None:
                    # Should not happen if env is consistent, but guard anyway
                    break

                done = last_results.get("_done", False)

                # Credit agents and build the next observation map
                obs_map = {}
                for agent, aid in zip(self.agents, swarm_env.agent_ids):
                    agent_result = last_results.get(aid, {})
                    er = agent_result.get("eval_result", {})
                    if er.get("correct", False):
                        agent.correct += 1
                    agent.total += 1
                    if not done:
                        obs_map[aid] = swarm_env.get_swarm_obs(aid)

            if verbose:
                stats = swarm_env.episode_stats()
                print(f"  ep{ep + 1}: accuracy={stats.accuracy:.0%}")

        # Merge the environment's shared memory into the Swarm's own pool
        for cat, actions in swarm_env.shared_memory.knowledge.items():
            for action, score in actions.items():
                self.shared.knowledge[cat][action] += score
        for agent_name, count in swarm_env.shared_memory.contributions.items():
            self.shared.contributions[agent_name] += count

        return SwarmResult(self.agents, self.shared)
