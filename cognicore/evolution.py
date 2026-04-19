"""
CogniCore Evolutionary Training — Natural selection for AI agents.

Runs populations of agents, selects the best, mutates, and repeats
— evolving optimal strategies through generations.

Usage::

    from cognicore.evolution import EvolutionEngine

    engine = EvolutionEngine("SafetyClassification-v1", population_size=20)
    best = engine.evolve(generations=10)
    print(best.knowledge)
"""

from __future__ import annotations

import random
import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cognicore


class EvolvableAgent:
    """Agent with a mutable genome (strategy parameters).

    The genome encodes:
    - keyword_weights: how much unsafe keywords matter
    - memory_reliance: how much to trust memory
    - exploration_rate: how often to try random actions
    - category_biases: default assumption per category
    """

    def __init__(self, genome: Optional[Dict] = None):
        self.genome = genome or self._random_genome()
        self.fitness = 0.0
        self.generation = 0
        self.knowledge: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    @staticmethod
    def _random_genome() -> Dict[str, Any]:
        return {
            "keyword_weight": random.uniform(0.1, 1.0),
            "memory_reliance": random.uniform(0.0, 1.0),
            "exploration_rate": random.uniform(0.05, 0.5),
            "default_action": random.choice(["SAFE", "UNSAFE", "NEEDS_REVIEW"]),
            "hint_trust": random.uniform(0.0, 1.0),
            "confidence_threshold": random.uniform(0.3, 0.9),
        }

    def act(self, obs: Dict) -> Dict[str, Any]:
        """Choose action based on genome."""
        prompt = obs.get("prompt", "").lower()
        category = obs.get("category", "unknown")

        # Exploration
        if random.random() < self.genome["exploration_rate"]:
            return {"classification": random.choice(["SAFE", "UNSAFE", "NEEDS_REVIEW"])}

        # Memory-based decision
        if category in self.knowledge and random.random() < self.genome["memory_reliance"]:
            best = max(self.knowledge[category], key=self.knowledge[category].get)
            return {"classification": best}

        # Hint-based
        hints = str(obs.get("reflection_hints", "")).lower()
        if hints and random.random() < self.genome["hint_trust"]:
            if "unsafe" in hints:
                return {"classification": "UNSAFE"}

        # Keyword-based
        unsafe_words = ["malware", "hack", "weapon", "bomb", "exploit", "phish",
                        "attack", "drug", "kill", "password"]
        score = sum(self.genome["keyword_weight"] for w in unsafe_words if w in prompt)
        if score > self.genome["confidence_threshold"]:
            return {"classification": "UNSAFE"}

        return {"classification": self.genome["default_action"]}

    def learn(self, reward, info):
        er = info.get("eval_result", {})
        cat = er.get("category", "?")
        pred = str(er.get("predicted", ""))
        if er.get("correct"):
            self.knowledge[cat][pred] += reward.total + 0.5
        else:
            self.knowledge[cat][pred] -= 0.3

    def mutate(self, rate: float = 0.2) -> "EvolvableAgent":
        """Create a mutated copy."""
        new_genome = copy.deepcopy(self.genome)

        for key in new_genome:
            if random.random() < rate:
                if isinstance(new_genome[key], float):
                    new_genome[key] += random.gauss(0, 0.1)
                    new_genome[key] = max(0.01, min(1.0, new_genome[key]))
                elif isinstance(new_genome[key], str):
                    new_genome[key] = random.choice(["SAFE", "UNSAFE", "NEEDS_REVIEW"])

        child = EvolvableAgent(new_genome)
        child.knowledge = copy.deepcopy(self.knowledge)
        return child

    @staticmethod
    def crossover(parent_a: "EvolvableAgent", parent_b: "EvolvableAgent") -> "EvolvableAgent":
        """Create child from two parents."""
        child_genome = {}
        for key in parent_a.genome:
            if random.random() < 0.5:
                child_genome[key] = parent_a.genome[key]
            else:
                child_genome[key] = parent_b.genome[key]

        child = EvolvableAgent(child_genome)
        # Inherit knowledge from better parent
        if parent_a.fitness >= parent_b.fitness:
            child.knowledge = copy.deepcopy(parent_a.knowledge)
        else:
            child.knowledge = copy.deepcopy(parent_b.knowledge)
        return child


class EvolutionEngine:
    """Evolutionary training engine — natural selection for AI agents.

    Parameters
    ----------
    env_id : str
        Environment to train on.
    population_size : int
        Number of agents per generation.
    elite_count : int
        Top N agents that survive each generation.
    mutation_rate : float
        Probability of mutating each gene.
    """

    def __init__(
        self,
        env_id: str = "SafetyClassification-v1",
        difficulty: str = "easy",
        population_size: int = 20,
        elite_count: int = 4,
        mutation_rate: float = 0.2,
    ):
        self.env_id = env_id
        self.difficulty = difficulty
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.population: List[EvolvableAgent] = [
            EvolvableAgent() for _ in range(population_size)
        ]
        self.history: List[Dict] = []

    def _evaluate(self, agent: EvolvableAgent) -> float:
        """Evaluate agent fitness on the environment."""
        env = cognicore.make(self.env_id, difficulty=self.difficulty)
        obs = env.reset()
        total_reward = 0

        while True:
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            agent.learn(reward, info)
            total_reward += reward.total
            if done:
                break

        stats = env.episode_stats()
        # Fitness = weighted combination of accuracy and reward
        fitness = stats.accuracy * 50 + env.get_score() * 50
        return fitness

    def evolve(
        self,
        generations: int = 10,
        verbose: bool = True,
    ) -> EvolvableAgent:
        """Run evolutionary training.

        Returns the best agent from the final generation.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Evolutionary Training")
            print(f"  Env: {self.env_id} ({self.difficulty})")
            print(f"  Population: {self.population_size} | Generations: {generations}")
            print(f"{'=' * 60}")

        for gen in range(1, generations + 1):
            # Evaluate all agents
            for agent in self.population:
                agent.fitness = self._evaluate(agent)
                agent.generation = gen

            # Sort by fitness
            self.population.sort(key=lambda a: -a.fitness)

            best = self.population[0]
            avg = sum(a.fitness for a in self.population) / len(self.population)

            self.history.append({
                "generation": gen,
                "best_fitness": best.fitness,
                "avg_fitness": avg,
                "best_genome": copy.deepcopy(best.genome),
            })

            if verbose:
                print(
                    f"  Gen {gen:3d}: best={best.fitness:.1f} "
                    f"avg={avg:.1f} "
                    f"explore={best.genome['exploration_rate']:.2f} "
                    f"default={best.genome['default_action']}"
                )

            if gen == generations:
                break

            # Selection: keep elite
            elite = self.population[:self.elite_count]

            # Create next generation
            new_pop = list(elite)  # elite survive

            while len(new_pop) < self.population_size:
                if random.random() < 0.7:
                    # Crossover
                    p1 = random.choice(elite)
                    p2 = random.choice(elite)
                    child = EvolvableAgent.crossover(p1, p2)
                else:
                    # Mutation of random elite
                    parent = random.choice(elite)
                    child = parent.mutate(self.mutation_rate)

                new_pop.append(child)

            self.population = new_pop

        best = self.population[0]
        if verbose:
            print(f"\n  Best agent genome:")
            for k, v in best.genome.items():
                print(f"    {k:25s} = {v}")
            print(f"  Fitness: {best.fitness:.1f}")
            print(f"{'=' * 60}\n")

        return best
