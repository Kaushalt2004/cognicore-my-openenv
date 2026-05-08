"""
Real RL Agents — Agents that ACTUALLY learn from experience.

These are not LLM wrappers. These are proper reinforcement learning
agents with Q-tables, epsilon-greedy exploration, and learning rates.

CogniCore's cognitive middleware (Memory + Reflection) is designed to
accelerate ANY of these agents, not just LLMs.
"""

from __future__ import annotations

import random
import math
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from cognicore.agents.base_agent import BaseAgent
from cognicore.core.types import EpisodeStats, StructuredReward

logger = logging.getLogger("cognicore.agents.rl")


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning agent.

    Learns an action-value function Q(s, a) from experience.
    Supports epsilon-greedy exploration with decay.

    Works with ANY CogniCore environment that has discrete actions.

    Example::

        agent = QLearningAgent(
            actions=["UP", "DOWN", "LEFT", "RIGHT"],
            learning_rate=0.1,
            discount=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
        )
        results = cc.train(agent=agent, env_id="GridWorld-v1", episodes=500)
    """

    def __init__(
        self,
        actions: List[str],
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: state_key -> {action: value}
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )

        self._last_state: Optional[str] = None
        self._last_action: Optional[str] = None
        self._pending_reward: Optional[float] = None
        self._episode_rewards: List[float] = []
        self._current_episode_reward: float = 0.0
        self._total_episodes: int = 0

    def _state_key(self, observation: Dict[str, Any]) -> str:
        """Convert observation to a hashable state key."""
        if "agent_pos" in observation:
            return str(tuple(observation["agent_pos"]))
        # Generic: hash the sorted items
        key_parts = []
        for k in sorted(observation.keys()):
            v = observation[k]
            if isinstance(v, (list, dict, set)):
                continue  # Skip non-hashable
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Epsilon-greedy action selection with deferred Q-update."""
        state = self._state_key(observation)

        # Apply pending Q-update from previous step (now we know next state)
        if self._last_state is not None and self._pending_reward is not None:
            old_q = self.q_table[self._last_state][self._last_action]
            max_next_q = max(self.q_table[state].values())
            td_target = self._pending_reward + self.gamma * max_next_q
            new_q = old_q + self.lr * (td_target - old_q)
            self.q_table[self._last_state][self._last_action] = new_q
            self._pending_reward = None

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)

        self._last_state = state
        self._last_action = action

        return {"action": action}

    def on_reward(self, reward: StructuredReward) -> None:
        """Store reward for deferred TD update (applied in next act())."""
        if self._last_state is None or self._last_action is None:
            return

        r = reward.total if hasattr(reward, "total") else float(reward)
        self._current_episode_reward += r
        self._pending_reward = r

    def on_episode_end(self, stats: EpisodeStats) -> None:
        """Decay epsilon after each episode."""
        self._total_episodes += 1
        self._episode_rewards.append(self._current_episode_reward)
        self._current_episode_reward = 0.0

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return learning statistics."""
        return {
            "episodes": self._total_episodes,
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "avg_reward_last_10": (
                round(sum(self._episode_rewards[-10:]) / max(1, min(10, len(self._episode_rewards))), 2)
                if self._episode_rewards else 0.0
            ),
        }


class SARSAAgent(BaseAgent):
    """SARSA (on-policy TD) agent.

    Unlike Q-Learning (off-policy), SARSA updates based on the action
    actually taken, making it more conservative and safer.

    Useful for environments where safety matters — the agent learns
    to avoid risky actions even during exploration.
    """

    def __init__(
        self,
        actions: List[str],
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )

        self._last_state: Optional[str] = None
        self._last_action: Optional[str] = None
        self._total_episodes: int = 0

    def _state_key(self, obs: Dict[str, Any]) -> str:
        if "agent_pos" in obs:
            return str(tuple(obs["agent_pos"]))
        return str(sorted((k, v) for k, v in obs.items() if isinstance(v, (int, float, str, bool))))

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        state = self._state_key(observation)

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best)

        # SARSA: update from PREVIOUS (s, a) -> current (s', a')
        if self._last_state is not None:
            old_q = self.q_table[self._last_state][self._last_action]
            next_q = self.q_table[state][action]
            # Note: reward is applied in on_reward
            self.q_table[self._last_state][self._last_action] = old_q

        self._last_state = state
        self._last_action = action
        return {"action": action}

    def on_reward(self, reward: StructuredReward) -> None:
        if self._last_state is None:
            return
        r = reward.total if hasattr(reward, "total") else float(reward)
        old_q = self.q_table[self._last_state][self._last_action]
        self.q_table[self._last_state][self._last_action] = old_q + self.lr * (r - old_q)

    def on_episode_end(self, stats: EpisodeStats) -> None:
        self._total_episodes += 1
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        self._last_state = None
        self._last_action = None


class GeneticAgent(BaseAgent):
    """Evolutionary / Genetic Algorithm agent.

    Maintains a population of action strategies and evolves them
    based on fitness (cumulative reward). NOT gradient-based —
    this is a completely different learning paradigm.

    Shows CogniCore works with non-gradient, non-LLM methods.
    """

    def __init__(
        self,
        actions: List[str],
        population_size: int = 20,
        mutation_rate: float = 0.1,
        strategy_length: int = 50,
    ) -> None:
        self.actions = actions
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.strategy_length = strategy_length

        # Population: list of action sequences
        self.population: List[List[str]] = [
            [random.choice(actions) for _ in range(strategy_length)]
            for _ in range(population_size)
        ]
        self.fitness: List[float] = [0.0] * population_size

        self._current_individual: int = 0
        self._step_in_strategy: int = 0
        self._current_fitness: float = 0.0
        self._generation: int = 0
        self._best_fitness: float = float("-inf")

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Use current individual's strategy."""
        strategy = self.population[self._current_individual]
        idx = self._step_in_strategy % len(strategy)
        action = strategy[idx]
        self._step_in_strategy += 1
        return {"action": action}

    def on_reward(self, reward: StructuredReward) -> None:
        r = reward.total if hasattr(reward, "total") else float(reward)
        self._current_fitness += r

    def on_episode_end(self, stats: EpisodeStats) -> None:
        """Evaluate individual and move to next, or evolve."""
        self.fitness[self._current_individual] = self._current_fitness
        self._current_fitness = 0.0
        self._step_in_strategy = 0
        self._current_individual += 1

        if self._current_individual >= self.pop_size:
            self._evolve()
            self._current_individual = 0
            self._generation += 1

    def _evolve(self) -> None:
        """Select, crossover, and mutate to create next generation."""
        self._best_fitness = max(self.fitness)

        # Tournament selection + elitism
        sorted_pop = sorted(
            zip(self.fitness, self.population),
            key=lambda x: x[0],
            reverse=True,
        )

        # Keep top 20% (elitism)
        elite_count = max(2, self.pop_size // 5)
        new_pop = [ind for _, ind in sorted_pop[:elite_count]]

        # Fill rest with crossover + mutation
        while len(new_pop) < self.pop_size:
            # Tournament selection
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            # Single-point crossover
            point = random.randint(1, self.strategy_length - 1)
            child = p1[:point] + p2[point:]

            # Mutation
            child = [
                random.choice(self.actions) if random.random() < self.mutation_rate else a
                for a in child
            ]
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        self.fitness = [0.0] * self.pop_size

    def _tournament_select(self, k: int = 3) -> List[str]:
        """Select individual via tournament."""
        contestants = random.sample(
            list(zip(self.fitness, self.population)),
            min(k, len(self.population)),
        )
        return max(contestants, key=lambda x: x[0])[1]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "best_fitness": round(self._best_fitness, 2),
            "population_size": self.pop_size,
            "mutation_rate": self.mutation_rate,
        }


class BanditAgent(BaseAgent):
    """Multi-armed bandit agent using Upper Confidence Bound (UCB1).

    For environments where the optimal action depends on the context
    but there's no sequential state transition. Uses UCB1 for
    exploration-exploitation tradeoff.
    """

    def __init__(self, actions: List[str]) -> None:
        self.actions = actions
        self.counts: Dict[str, int] = {a: 0 for a in actions}
        self.values: Dict[str, float] = {a: 0.0 for a in actions}
        self.total_steps: int = 0
        self._last_action: Optional[str] = None

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self.total_steps += 1

        # Try each action at least once
        untried = [a for a in self.actions if self.counts[a] == 0]
        if untried:
            action = random.choice(untried)
        else:
            # UCB1
            ucb_values = {}
            for a in self.actions:
                exploitation = self.values[a]
                exploration = math.sqrt(2 * math.log(self.total_steps) / self.counts[a])
                ucb_values[a] = exploitation + exploration
            action = max(ucb_values, key=ucb_values.get)

        self._last_action = action
        self.counts[action] += 1
        return {"action": action}

    def on_reward(self, reward: StructuredReward) -> None:
        if self._last_action is None:
            return
        r = reward.total if hasattr(reward, "total") else float(reward)
        a = self._last_action
        n = self.counts[a]
        # Incremental mean update
        self.values[a] = self.values[a] + (r - self.values[a]) / n
