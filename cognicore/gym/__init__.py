"""
CogniCore Gymnasium-Native Environments.

These are proper gymnasium.Env subclasses that:
  - Use numpy observation/action spaces
  - Return float rewards (structured reward in info)
  - Work with SB3, RLlib, CleanRL out of the box
  - Auto-register with gymnasium

Usage:
    import gymnasium as gym
    import cognicore.gym  # registers all envs

    env = gym.make("cognicore/MazeRunner-v0")
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import logging
from typing import Any, Dict, Optional, Tuple, Set, List

logger = logging.getLogger("cognicore.gym")


# ═══════════════════════════════════════════════════════════════
#  MazeRunner — Gymnasium Native
# ═══════════════════════════════════════════════════════════════

class MazeRunnerGym(gym.Env):
    """Procedurally generated maze — proper Gymnasium env.

    Observation (Box):
      [agent_row, agent_col, goal_row, goal_col, distance_to_goal,
       visited_count, steps_remaining, wall_N, wall_S, wall_E, wall_W]

    Action (Discrete(4)): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

    Reward (float):
      +10.0  for reaching goal
      +0.1   for moving closer
      -0.1   for moving farther
      -0.3   for hitting wall
      -0.05  per revisit
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 10}
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(
        self,
        size: int = 8,
        max_steps: int = 0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.size = size
        self.max_episode_steps = max_steps or size * size * 3
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([size, size, size, size, size*2, 100, size*size*3,
                          1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # State
        self.walls: Set[Tuple[int, int]] = set()
        self.agent_pos = (1, 1)
        self.goal_pos = (size - 2, size - 2)
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self.steps_taken = 0

        # Cognitive memory (CogniCore's value-add)
        self.trap_memory: Set[Tuple[int, int]] = set()  # walls agent has bumped into
        self.successful_paths: List[List[Tuple[int, int]]] = []
        self.current_path: List[Tuple[int, int]] = []
        self._np_random = None

    def _generate_maze(self) -> None:
        """DFS maze generation — deterministic given RNG state."""
        self.walls = set()
        for r in range(self.size):
            for c in range(self.size):
                self.walls.add((r, c))

        stack = [(1, 1)]
        self.walls.discard((1, 1))

        while stack:
            current = stack[-1]
            r, c = current

            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.size - 1 and 1 <= nc < self.size - 1:
                    if (nr, nc) in self.walls:
                        neighbors.append((nr, nc, r + dr // 2, c + dc // 2))

            if neighbors:
                nr, nc, wr, wc = neighbors[self._np_random.integers(len(neighbors))]
                self.walls.discard((nr, nc))
                self.walls.discard((wr, wc))
                stack.append((nr, nc))
            else:
                stack.pop()

        self.walls.discard(self.goal_pos)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = self.goal_pos[0] + dr, self.goal_pos[1] + dc
            if 1 <= nr < self.size - 1 and 1 <= nc < self.size - 1:
                self.walls.discard((nr, nc))

    def _get_obs(self) -> np.ndarray:
        r, c = self.agent_pos
        gr, gc = self.goal_pos
        dist = abs(r - gr) + abs(c - gc)
        visits = self.visit_counts.get(self.agent_pos, 0)
        remaining = self.max_episode_steps - self.steps_taken

        # Wall detection (N, S, E, W)
        wall_n = 1.0 if (r - 1, c) in self.walls else 0.0
        wall_s = 1.0 if (r + 1, c) in self.walls else 0.0
        wall_e = 1.0 if (r, c + 1) in self.walls else 0.0
        wall_w = 1.0 if (r, c - 1) in self.walls else 0.0

        return np.array([r, c, gr, gc, dist, visits, remaining,
                        wall_n, wall_s, wall_e, wall_w], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "distance": abs(self.agent_pos[0] - self.goal_pos[0])
                + abs(self.agent_pos[1] - self.goal_pos[1]),
            "visits": self.visit_counts.get(self.agent_pos, 0),
            "walls_known": len(self.trap_memory),
            "steps": self.steps_taken,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._np_random = self.np_random

        self._generate_maze()
        self.agent_pos = (1, 1)
        self.visit_counts = {(1, 1): 1}
        self.steps_taken = 0
        self.current_path = [(1, 1)]

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        dr, dc = self.ACTIONS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
        new_pos = (new_r, new_c)

        self.steps_taken += 1
        terminated = False
        truncated = self.steps_taken >= self.max_episode_steps

        # Hit wall
        if new_pos in self.walls or not (0 <= new_r < self.size and 0 <= new_c < self.size):
            self.trap_memory.add(new_pos)  # Remember this wall
            reward = -0.3
            info = self._get_info()
            info["event"] = "wall"
            return self._get_obs(), reward, terminated, truncated, info

        # Move
        old_dist = abs(self.goal_pos[0] - self.agent_pos[0]) + abs(self.goal_pos[1] - self.agent_pos[1])
        self.agent_pos = new_pos
        self.visit_counts[new_pos] = self.visit_counts.get(new_pos, 0) + 1
        self.current_path.append(new_pos)
        new_dist = abs(self.goal_pos[0] - new_pos[0]) + abs(self.goal_pos[1] - new_pos[1])

        # Reached goal
        if new_pos == self.goal_pos:
            efficiency = 1.0 - (self.steps_taken / self.max_episode_steps)
            reward = 10.0 + 5.0 * efficiency
            terminated = True
            self.successful_paths.append(list(self.current_path))
            info = self._get_info()
            info["event"] = "goal"
            info["efficiency"] = efficiency
            return self._get_obs(), reward, terminated, truncated, info

        # Direction reward
        reward = 0.1 if new_dist < old_dist else -0.1

        # Revisit penalty
        if self.visit_counts[new_pos] > 1:
            reward -= 0.05 * (self.visit_counts[new_pos] - 1)

        info = self._get_info()
        info["event"] = "step"
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi" or self.render_mode == "human":
            lines = []
            for r in range(self.size):
                row = ""
                for c in range(self.size):
                    pos = (r, c)
                    if pos == self.agent_pos:
                        row += "A"
                    elif pos == self.goal_pos:
                        row += "G"
                    elif pos in self.walls:
                        row += "#"
                    elif pos in self.visit_counts:
                        row += "."
                    else:
                        row += " "
                lines.append(row)
            result = "\n".join(lines)
            if self.render_mode == "human":
                print(result)
            return result
        return None


# ═══════════════════════════════════════════════════════════════
#  GridWorld — Gymnasium Native
# ═══════════════════════════════════════════════════════════════

class GridWorldGym(gym.Env):
    """Grid navigation with traps — proper Gymnasium env.

    Obs: [row, col, goal_row, goal_col, dist, *trap_dists]
    Action: Discrete(4) — UP/DOWN/LEFT/RIGHT
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 5, num_traps: int = 3,
                 max_steps: int = 50, render_mode: Optional[str] = None):
        super().__init__()
        self.size = size
        self.num_traps = num_traps
        self.max_episode_steps = max_steps
        self.render_mode = render_mode

        obs_dim = 5 + num_traps * 2  # pos, goal, dist + trap positions
        self.observation_space = spaces.Box(
            low=0, high=max(size, max_steps),
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self.agent_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.traps: List[Tuple[int, int]] = []
        self.steps_taken = 0

    def _get_obs(self) -> np.ndarray:
        features = [
            self.agent_pos[0], self.agent_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            abs(self.agent_pos[0] - self.goal_pos[0])
            + abs(self.agent_pos[1] - self.goal_pos[1]),
        ]
        for trap in self.traps:
            features.extend([trap[0], trap[1]])
        return np.array(features, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = (0, 0)
        self.steps_taken = 0

        self.traps = []
        while len(self.traps) < self.num_traps:
            pos = (self.np_random.integers(self.size),
                   self.np_random.integers(self.size))
            if pos != (0, 0) and pos != self.goal_pos and pos not in self.traps:
                self.traps.append(pos)

        return self._get_obs(), {"traps": self.traps}

    def step(self, action: int):
        dr, dc = self.ACTIONS[action]
        new_r = max(0, min(self.size - 1, self.agent_pos[0] + dr))
        new_c = max(0, min(self.size - 1, self.agent_pos[1] + dc))
        self.agent_pos = (new_r, new_c)
        self.steps_taken += 1

        truncated = self.steps_taken >= self.max_episode_steps

        if self.agent_pos in self.traps:
            return self._get_obs(), -5.0, True, truncated, {"event": "trap"}
        if self.agent_pos == self.goal_pos:
            bonus = max(0, 1.0 - self.steps_taken / self.max_episode_steps)
            return self._get_obs(), 10.0 + 5.0 * bonus, True, truncated, {"event": "goal"}

        # Step penalty
        return self._get_obs(), -0.1, False, truncated, {"event": "step"}


# ═══════════════════════════════════════════════════════════════
#  Trading — Gymnasium Native
# ═══════════════════════════════════════════════════════════════

class TradingGym(gym.Env):
    """Trading environment — continuous obs, discrete actions.

    Obs: [price, return_5d, volatility, portfolio_value, cash,
          position, drawdown, regime_encoded]
    Action: Discrete(3) — 0=HOLD, 1=BUY, 2=SELL
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_steps: int = 200, render_mode: Optional[str] = None):
        super().__init__()
        self.max_episode_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.price = 100.0
        self.cash = 10000.0
        self.position = 0
        self.peak_value = 10000.0
        self.returns_history: List[float] = []
        self.steps_taken = 0
        self.regime = 0  # 0=low vol, 1=medium, 2=high

    def _portfolio_value(self):
        return self.cash + self.position * self.price

    def _get_obs(self):
        val = self._portfolio_value()
        recent = self.returns_history[-5:] if self.returns_history else [0.0]
        avg_ret = sum(recent) / len(recent)
        vol = np.std(recent) if len(recent) > 1 else 0.0
        dd = 1.0 - val / self.peak_value if self.peak_value > 0 else 0.0

        return np.array([
            self.price, avg_ret, vol, val, self.cash,
            self.position, dd, float(self.regime),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.price = 100.0
        self.cash = 10000.0
        self.position = 0
        self.peak_value = 10000.0
        self.returns_history = []
        self.steps_taken = 0
        self.regime = 0
        return self._get_obs(), {}

    def step(self, action: int):
        old_value = self._portfolio_value()
        self.steps_taken += 1

        # Execute trade
        if action == 1 and self.cash >= self.price:  # BUY
            qty = int(self.cash * 0.3 / self.price)
            if qty > 0:
                self.position += qty
                self.cash -= qty * self.price
        elif action == 2 and self.position > 0:  # SELL
            qty = max(1, self.position // 2)
            self.cash += qty * self.price
            self.position -= qty

        # Market move
        vol = [0.01, 0.025, 0.05][self.regime]
        ret = self.np_random.normal(0.0005, vol)
        if self.np_random.random() < 0.02:
            ret += self.np_random.normal(0, vol * 5)
        self.price *= (1 + ret)
        self.price = max(1.0, self.price)
        self.returns_history.append(ret)

        if self.np_random.random() < 0.05:
            self.regime = int(self.np_random.integers(3))

        new_value = self._portfolio_value()
        self.peak_value = max(self.peak_value, new_value)
        pnl = new_value - old_value

        truncated = self.steps_taken >= self.max_episode_steps
        reward = pnl / 100.0  # Normalize

        return self._get_obs(), reward, False, truncated, {
            "pnl": pnl, "portfolio": new_value, "regime": self.regime,
        }


# ═══════════════════════════════════════════════════════════════
#  Survival — Gymnasium Native
# ═══════════════════════════════════════════════════════════════

class SurvivalGym(gym.Env):
    """Survival environment — long-horizon planning.

    Obs: [health, hunger, shelter, tools, food, materials,
          threat_level, weather_encoded, day]
    Action: Discrete(7) — FORAGE/HUNT/BUILD/CRAFT/REST/EXPLORE/DEFEND
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}
    ACTION_NAMES = ["FORAGE", "HUNT", "BUILD", "CRAFT", "REST", "EXPLORE", "DEFEND"]

    def __init__(self, max_steps: int = 200, render_mode: Optional[str] = None):
        super().__init__()
        self.max_episode_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=300, shape=(9,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(7)
        self._reset_state()

    def _reset_state(self):
        self.health = 100.0
        self.hunger = 0.0
        self.shelter = 0
        self.tools = 0
        self.food = 5
        self.materials = 0
        self.threat = 0
        self.weather = 0  # 0=clear, 1=rain, 2=storm, 3=heat
        self.day = 0
        self.alive = True

    def _get_obs(self):
        return np.array([
            self.health, self.hunger, self.shelter, self.tools,
            self.food, self.materials, self.threat,
            float(self.weather), float(self.day),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {"alive": True}

    def step(self, action: int):
        if not self.alive:
            return self._get_obs(), 0.0, True, False, {"event": "dead", "day": self.day}

        reward = 0.1  # Alive bonus
        event = "step"

        # Natural decay
        self.hunger = min(100, self.hunger + 2)
        if self.hunger > 80:
            self.health -= 3

        # Actions
        if action == 0:  # FORAGE
            found = self.np_random.integers(0, 3 if self.tools > 0 else 2)
            self.food += found
            self.materials += int(self.np_random.integers(0, 2))
            reward += 0.2 * found
            event = f"foraged {found}"

        elif action == 1:  # HUNT
            if self.tools > 0 and self.np_random.random() < 0.6:
                gained = int(self.np_random.integers(3, 7))
                self.food += gained
                reward += 0.5
                event = "hunt success"
            else:
                self.health -= 5
                reward -= 0.3
                event = "hunt failed"

        elif action == 2:  # BUILD
            if self.materials >= 3:
                self.materials -= 3
                self.shelter = min(5, self.shelter + 1)
                reward += 0.5
                event = f"built shelter ({self.shelter})"
            else:
                reward -= 0.1

        elif action == 3:  # CRAFT
            if self.materials >= 2:
                self.materials -= 2
                self.tools += 1
                reward += 0.4
                event = f"crafted tool ({self.tools})"
            else:
                reward -= 0.1

        elif action == 4:  # REST
            heal = min(15, 100 - self.health)
            self.health += heal
            reward += 0.1
            event = f"rested +{heal:.0f}hp"

        elif action == 5:  # EXPLORE
            self.materials += int(self.np_random.integers(1, 5))
            if self.np_random.random() < 0.3:
                self.food += int(self.np_random.integers(1, 3))
            reward += 0.3
            event = "explored"

        elif action == 6:  # DEFEND
            if self.threat > 0:
                power = 1 + self.tools * 0.5 + self.shelter * 0.3
                if power > self.threat:
                    self.threat = 0
                    reward += 1.0
                    event = "defended"
                else:
                    dmg = (self.threat - power) * 10
                    self.health -= dmg
                    reward -= 0.3
                    event = f"partial defense, -{dmg:.0f}hp"
            else:
                reward -= 0.05

        # Auto-eat
        if self.hunger > 40 and self.food > 0:
            eat = min(self.food, 2)
            self.food -= eat
            self.hunger = max(0, self.hunger - eat * 25)

        # Weather
        if self.np_random.random() < 0.1:
            self.weather = int(self.np_random.integers(4))
        if self.weather == 2 and self.shelter < 2:
            self.health -= 10
            event += " +STORM"

        # Threats
        if self.np_random.random() < 0.08:
            self.threat = int(self.np_random.integers(1, 4))
        if self.threat > 0 and action != 6:
            self.health -= self.threat * 3

        # Death check
        if self.health <= 0:
            self.alive = False
            reward = -5.0
            event = "DIED"

        self.day += 1
        terminated = not self.alive
        truncated = self.day >= self.max_episode_steps

        # Survival bonus scales with time
        if self.alive:
            reward += 0.01 * self.day

        info = {"event": event, "day": self.day, "health": self.health,
                "alive": self.alive}
        return self._get_obs(), reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════

def register_gymnasium_envs():
    """Register all CogniCore envs with Gymnasium."""
    _envs = {
        "cognicore/MazeRunner-v0":       (MazeRunnerGym, {"size": 8}),
        "cognicore/MazeRunner-Medium-v0": (MazeRunnerGym, {"size": 12}),
        "cognicore/MazeRunner-Hard-v0":  (MazeRunnerGym, {"size": 16}),
        "cognicore/GridWorld-v0":        (GridWorldGym, {"size": 5, "num_traps": 3}),
        "cognicore/GridWorld-Hard-v0":   (GridWorldGym, {"size": 10, "num_traps": 15}),
        "cognicore/Trading-v0":          (TradingGym, {"max_steps": 200}),
        "cognicore/Survival-v0":         (SurvivalGym, {"max_steps": 200}),
    }

    for env_id, (cls, kwargs) in _envs.items():
        try:
            gym.register(
                id=env_id,
                entry_point=cls,
                kwargs=kwargs,
                max_episode_steps=kwargs.get("max_steps", kwargs.get("max_episode_steps", 500)),
            )
        except gym.error.NameAlreadyRegistered:
            pass  # Already registered


# Auto-register on import
register_gymnasium_envs()
