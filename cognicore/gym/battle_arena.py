"""
BattleArena — Multi-agent competitive environment.

Two agents on a grid compete for territory control.
Each agent can move, attack, or defend.
First agent to lose all health loses.

Gymnasium-compatible with proper spaces.
"""
from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple


class BattleArenaGym(gym.Env):
    """2-player battle on a grid.

    Obs: [my_row, my_col, my_health, opp_row, opp_col, opp_health,
          distance, my_attack, my_defense]
    Action: Discrete(6) — UP/DOWN/LEFT/RIGHT/ATTACK/DEFEND
    
    Player 1 is the learning agent. Player 2 uses a built-in heuristic.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}
    ACTIONS = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1), 4: "attack", 5: "defend"}

    def __init__(self, size: int = 7, max_steps: int = 100,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.size = size
        self.max_episode_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=max(size, 100, max_steps),
            shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.p1 = {"pos": [1, 1], "hp": 100, "atk": 10, "def": 0}
        self.p2 = {"pos": [size-2, size-2], "hp": 100, "atk": 10, "def": 0}
        self.steps = 0

    def _dist(self):
        return abs(self.p1["pos"][0]-self.p2["pos"][0]) + abs(self.p1["pos"][1]-self.p2["pos"][1])

    def _get_obs(self):
        return np.array([
            self.p1["pos"][0], self.p1["pos"][1], self.p1["hp"],
            self.p2["pos"][0], self.p2["pos"][1], self.p2["hp"],
            self._dist(), self.p1["atk"], self.p1["def"],
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p1 = {"pos": [1, 1], "hp": 100, "atk": 10, "def": 0}
        self.p2 = {"pos": [self.size-2, self.size-2], "hp": 100, "atk": 10, "def": 0}
        self.steps = 0
        return self._get_obs(), {}

    def _move(self, player, dr, dc):
        nr = max(0, min(self.size-1, player["pos"][0] + dr))
        nc = max(0, min(self.size-1, player["pos"][1] + dc))
        player["pos"] = [nr, nc]

    def _do_attack(self, attacker, defender):
        dmg = max(0, attacker["atk"] - defender["def"])
        defender["hp"] -= dmg
        defender["def"] = 0
        return dmg

    def _opponent_act(self):
        """Simple heuristic opponent."""
        dist = self._dist()
        if dist <= 1:
            if self.np_random.random() < 0.6:
                self._do_attack(self.p2, self.p1)
            else:
                self.p2["def"] = 5
        else:
            dr = np.sign(self.p1["pos"][0] - self.p2["pos"][0])
            dc = np.sign(self.p1["pos"][1] - self.p2["pos"][1])
            if self.np_random.random() < 0.5:
                self._move(self.p2, int(dr), 0)
            else:
                self._move(self.p2, 0, int(dc))

    def step(self, action: int):
        self.steps += 1
        self.p1["def"] = 0
        reward = -0.01  # Step cost
        event = "step"

        if action < 4:
            dr, dc = self.ACTIONS[action]
            self._move(self.p1, dr, dc)
            # Closer to opponent = small reward
            if self._dist() < 3:
                reward += 0.05
        elif action == 4:  # Attack
            if self._dist() <= 1:
                dmg = self._do_attack(self.p1, self.p2)
                reward += dmg * 0.1
                event = f"attack {dmg}dmg"
            else:
                reward -= 0.1
                event = "attack missed"
        elif action == 5:  # Defend
            self.p1["def"] = 5
            reward += 0.02
            event = "defend"

        self._opponent_act()

        terminated = False
        if self.p2["hp"] <= 0:
            reward += 10.0
            terminated = True
            event = "WIN"
        elif self.p1["hp"] <= 0:
            reward -= 10.0
            terminated = True
            event = "LOSE"

        truncated = self.steps >= self.max_episode_steps
        if truncated and not terminated:
            reward += (self.p1["hp"] - self.p2["hp"]) * 0.01

        info = {"event": event, "p1_hp": self.p1["hp"], "p2_hp": self.p2["hp"],
                "distance": self._dist(), "steps": self.steps}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode in ("ansi", "human"):
            lines = []
            for r in range(self.size):
                row = ""
                for c in range(self.size):
                    if [r,c] == self.p1["pos"]:
                        row += "1"
                    elif [r,c] == self.p2["pos"]:
                        row += "2"
                    else:
                        row += "."
                lines.append(row)
            result = "\n".join(lines)
            result += f"\nP1: {self.p1['hp']}hp | P2: {self.p2['hp']}hp"
            if self.render_mode == "human":
                print(result)
            return result
        return None
