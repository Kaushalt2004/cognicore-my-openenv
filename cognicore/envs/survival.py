"""
SurvivalEnv — Long-horizon survival with threats, crafting, exploration.

The agent must survive as long as possible by managing:
  - Health (decreases over time, restored by food)
  - Hunger (must eat regularly)
  - Shelter (protects from weather events)
  - Tools (crafted from gathered materials)
  - Threats (predators, storms, disease)

Memory helps: "Storm comes every ~20 steps, I need shelter before then"
Long-horizon planning is REQUIRED — pure greedy fails.
"""

from __future__ import annotations
import random
import logging
from typing import Any, Dict, List

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace

logger = logging.getLogger("cognicore.envs.survival")


class SurvivalEnv(CogniCoreEnv):
    """Survival environment requiring long-horizon planning.

    Actions: FORAGE, HUNT, BUILD_SHELTER, CRAFT_TOOL, REST, EXPLORE, DEFEND
    """

    ACTIONS = ["FORAGE", "HUNT", "BUILD_SHELTER", "CRAFT_TOOL", "REST", "EXPLORE", "DEFEND"]

    def _setup(self, **kwargs: Any) -> None:
        self.difficulty = kwargs.get("difficulty", "easy")
        configs = {
            "easy":   {"max_steps": 100, "threat_rate": 0.05, "decay_rate": 1},
            "medium": {"max_steps": 200, "threat_rate": 0.10, "decay_rate": 2},
            "hard":   {"max_steps": 300, "threat_rate": 0.15, "decay_rate": 3},
        }
        cfg = configs.get(self.difficulty, configs["easy"])
        self.max_steps_val = cfg["max_steps"]
        self.threat_rate = cfg["threat_rate"]
        self.decay_rate = cfg["decay_rate"]
        self.num_tasks = kwargs.get("num_tasks", self.max_steps_val)

        self.action_space = DiscreteSpace(n=7, labels=self.ACTIONS)
        self.observation_space = DictSpace(fields={
            "health": "Current health (0-100)",
            "hunger": "Current hunger (0-100, 100=starving)",
            "shelter_level": "Shelter quality (0-5)",
            "tools": "Number of tools",
            "food_stored": "Food in inventory",
            "materials": "Building materials",
            "threat_level": "Current danger level",
            "weather": "Current weather",
            "day": "Current day",
        })

        self._reset_state()

    def _reset_state(self):
        self.health = 100.0
        self.hunger = 0.0
        self.shelter = 0
        self.tools = 0
        self.food = 5
        self.materials = 0
        self.day = 0
        self.weather = "clear"
        self.threat_level = 0
        self.survived_days = 0
        self.alive = True

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        self._reset_state()
        return [{"step": i} for i in range(self.max_steps_val)]

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "health": round(self.health, 1),
            "hunger": round(self.hunger, 1),
            "shelter_level": self.shelter,
            "tools": self.tools,
            "food_stored": self.food,
            "materials": self.materials,
            "threat_level": self.threat_level,
            "weather": self.weather,
            "day": self.day,
            "alive": self.alive,
            "steps_survived": self.survived_days,
        }

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        if not self.alive:
            self._current_step = len(self._tasks) - 1
            return EvalResult(
                base_score=0.0, correct=False, category="survival",
                ground_truth="SURVIVE", predicted="DEAD",
                metadata={"event": "dead", "survived": self.survived_days},
            )

        move = action.get("action", "REST")
        if isinstance(move, int):
            move = self.ACTIONS[move % len(self.ACTIONS)]
        move = str(move).upper()

        event = "step"
        score = 0.2  # Surviving = baseline good

        # Natural decay
        self.hunger = min(100, self.hunger + self.decay_rate)
        if self.hunger > 80:
            self.health -= 2  # Starving damages health

        # Execute action
        if move == "FORAGE":
            found = random.randint(1, 3) if self.tools > 0 else random.randint(0, 2)
            self.food += found
            self.materials += random.randint(0, 1)
            score = 0.3 if found > 0 else 0.1
            event = f"foraged {found} food"

        elif move == "HUNT":
            if self.tools > 0:
                success = random.random() < 0.6
                if success:
                    self.food += random.randint(3, 6)
                    score = 0.5
                    event = "hunt success"
                else:
                    self.health -= 5  # Failed hunt = injury
                    score = 0.0
                    event = "hunt failed, injured"
            else:
                score = 0.0
                event = "no tools to hunt"

        elif move == "BUILD_SHELTER":
            if self.materials >= 3:
                self.materials -= 3
                self.shelter = min(5, self.shelter + 1)
                score = 0.5
                event = f"shelter level {self.shelter}"
            else:
                score = 0.05
                event = "not enough materials"

        elif move == "CRAFT_TOOL":
            if self.materials >= 2:
                self.materials -= 2
                self.tools += 1
                score = 0.4
                event = f"crafted tool ({self.tools} total)"
            else:
                score = 0.05
                event = "not enough materials"

        elif move == "REST":
            heal = min(15, 100 - self.health)
            self.health += heal
            score = 0.2 if self.health < 80 else 0.1
            event = f"rested, healed {heal:.0f}"

        elif move == "EXPLORE":
            self.materials += random.randint(1, 4)
            if random.random() < 0.3:
                self.food += random.randint(1, 2)
            score = 0.3
            event = f"found {self.materials} materials"

        elif move == "DEFEND":
            if self.threat_level > 0:
                defense_power = 1 + self.tools * 0.5 + self.shelter * 0.3
                if defense_power > self.threat_level:
                    self.threat_level = 0
                    score = 0.6
                    event = "threat neutralized"
                else:
                    damage = (self.threat_level - defense_power) * 10
                    self.health -= damage
                    self.threat_level = max(0, self.threat_level - 1)
                    score = 0.1
                    event = f"partially defended, took {damage:.0f} damage"
            else:
                score = 0.05
                event = "no threat to defend against"

        # Eat if hungry and have food
        if self.hunger > 40 and self.food > 0:
            eat = min(self.food, 2)
            self.food -= eat
            self.hunger = max(0, self.hunger - eat * 25)

        # Weather events
        if random.random() < 0.1:
            self.weather = random.choice(["clear", "rain", "storm", "heat"])
        if self.weather == "storm" and self.shelter < 2:
            self.health -= 10
            event += " | STORM DAMAGE"
        elif self.weather == "heat" and random.random() < 0.3:
            self.hunger += 5

        # Random threats
        if random.random() < self.threat_rate:
            self.threat_level = random.randint(1, 3)

        if self.threat_level > 0 and move != "DEFEND":
            # Undefended threat damages health
            self.health -= self.threat_level * 5

        # Check death
        if self.health <= 0:
            self.alive = False
            self._current_step = len(self._tasks) - 1
            score = 0.0
            event = "DIED"

        self.survived_days += 1
        self.day = self.survived_days

        # Bonus for long survival
        survival_bonus = min(0.3, self.survived_days / self.max_steps_val)
        score = min(1.0, score + survival_bonus)

        return EvalResult(
            base_score=score, correct=self.alive, category="survival",
            ground_truth="SURVIVE", predicted=move,
            metadata={"event": event, "health": round(self.health, 1),
                      "hunger": round(self.hunger, 1),
                      "survived": self.survived_days},
        )
