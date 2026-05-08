"""
ResourceGathering — Multi-objective RL environment.

An agent must gather resources (food, wood, stone) to build structures,
while managing energy and avoiding hazards. This is a proper multi-objective
optimization problem — NOT a classification task.

Features:
- Multi-objective reward (food, energy, building progress)
- Resource management (limited energy per episode)
- Strategic planning (gather vs build vs explore)
- Memory benefit: remembers resource locations across episodes
- Reflection benefit: learns which strategies work vs fail
"""

from __future__ import annotations

import random
import logging
from typing import Any, Dict, List

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace

logger = logging.getLogger("cognicore.envs.resource_gathering")


class ResourceGatheringEnv(CogniCoreEnv):
    """Multi-objective resource gathering environment.

    The agent must:
    1. Explore to find resource nodes
    2. Gather resources (food, wood, stone)
    3. Build structures for points
    4. Manage energy (each action costs energy)
    5. Survive until episode ends

    Actions: GATHER_FOOD, GATHER_WOOD, GATHER_STONE, BUILD, REST, EXPLORE
    """

    ACTIONS = ["GATHER_FOOD", "GATHER_WOOD", "GATHER_STONE", "BUILD", "REST", "EXPLORE"]

    def _setup(self, **kwargs: Any) -> None:
        self.difficulty: str = kwargs.get("difficulty", "easy")
        self.num_tasks: int = kwargs.get("num_tasks", 30)

        configs = {
            "easy":   {"max_energy": 100, "build_cost": 5,  "resource_richness": 0.8},
            "medium": {"max_energy": 80,  "build_cost": 10, "resource_richness": 0.5},
            "hard":   {"max_energy": 60,  "build_cost": 15, "resource_richness": 0.3},
        }
        cfg = configs.get(self.difficulty, configs["easy"])

        self.max_energy: int = cfg["max_energy"]
        self.build_cost: int = cfg["build_cost"]
        self.resource_richness: float = cfg["resource_richness"]

        self.action_space = DiscreteSpace(n=6, labels=self.ACTIONS)
        self.observation_space = DictSpace(fields={
            "food": "Current food supply",
            "wood": "Current wood supply",
            "stone": "Current stone supply",
            "energy": "Remaining energy",
            "buildings": "Number of structures built",
            "explored_areas": "Number of areas explored",
            "turn": "Current turn number",
        })

        # State
        self.food: int = 0
        self.wood: int = 0
        self.stone: int = 0
        self.energy: int = self.max_energy
        self.buildings: int = 0
        self.explored: int = 0
        self.known_resources: Dict[str, int] = {"food": 2, "wood": 2, "stone": 2}

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        self.food = 0
        self.wood = 0
        self.stone = 0
        self.energy = self.max_energy
        self.buildings = 0
        self.explored = 0
        self.known_resources = {"food": 2, "wood": 2, "stone": 2}
        return [{"turn": i} for i in range(self.num_tasks)]

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "food": self.food,
            "wood": self.wood,
            "stone": self.stone,
            "energy": self.energy,
            "buildings": self.buildings,
            "explored_areas": self.explored,
            "known_resources": dict(self.known_resources),
            "turn": self._current_step,
            "max_turns": self.num_tasks,
            "can_build": self.food >= self.build_cost
                and self.wood >= self.build_cost
                and self.stone >= self.build_cost,
        }

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        move = action.get("action", "REST")
        if isinstance(move, int):
            move = self.ACTIONS[move % len(self.ACTIONS)]
        move = str(move).upper()

        energy_cost = 5 if move != "REST" else 0
        score = 0.0
        event = "neutral"

        if self.energy <= 0:
            # Out of energy — episode ends
            self._current_step = len(self._tasks) - 1
            return EvalResult(
                base_score=0.0, correct=False, category="survival",
                ground_truth="MANAGE_ENERGY", predicted=move,
                metadata={"event": "exhausted", "buildings": self.buildings},
            )

        self.energy -= energy_cost

        if move == "GATHER_FOOD":
            if self.known_resources["food"] > 0:
                amount = random.randint(1, 3) if random.random() < self.resource_richness else 0
                self.food += amount
                score = 0.4 if amount > 0 else 0.1
                event = f"gathered {amount} food"
            else:
                score = 0.0
                event = "no food sources known"

        elif move == "GATHER_WOOD":
            if self.known_resources["wood"] > 0:
                amount = random.randint(1, 3) if random.random() < self.resource_richness else 0
                self.wood += amount
                score = 0.4 if amount > 0 else 0.1
                event = f"gathered {amount} wood"
            else:
                score = 0.0
                event = "no wood sources known"

        elif move == "GATHER_STONE":
            if self.known_resources["stone"] > 0:
                amount = random.randint(1, 2) if random.random() < self.resource_richness else 0
                self.stone += amount
                score = 0.4 if amount > 0 else 0.1
                event = f"gathered {amount} stone"
            else:
                score = 0.0
                event = "no stone sources known"

        elif move == "BUILD":
            if (self.food >= self.build_cost
                    and self.wood >= self.build_cost
                    and self.stone >= self.build_cost):
                self.food -= self.build_cost
                self.wood -= self.build_cost
                self.stone -= self.build_cost
                self.buildings += 1
                score = 1.0
                event = f"built structure #{self.buildings}"
            else:
                score = 0.0
                event = "not enough resources to build"

        elif move == "REST":
            recovered = min(20, self.max_energy - self.energy)
            self.energy += recovered
            score = 0.2
            event = f"rested, recovered {recovered} energy"

        elif move == "EXPLORE":
            self.explored += 1
            # Exploration reveals new resource nodes
            if random.random() < 0.6:
                resource = random.choice(["food", "wood", "stone"])
                self.known_resources[resource] += 1
                score = 0.5
                event = f"discovered new {resource} source"
            else:
                score = 0.2
                event = "explored but found nothing"

        correct = score >= 0.4
        return EvalResult(
            base_score=score,
            correct=correct,
            category="resource_management",
            ground_truth="OPTIMAL_STRATEGY",
            predicted=move,
            metadata={
                "event": event,
                "buildings": self.buildings,
                "total_resources": self.food + self.wood + self.stone,
                "energy_remaining": self.energy,
            },
        )
