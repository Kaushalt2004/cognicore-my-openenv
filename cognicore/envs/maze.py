"""
CogniCore v0.7.0 — World-Class Environments

Environments that test REAL intelligence, not text classification.
Every env provides Memory + Reflection as infrastructure.
"""

# ═══════════════════════════════════════════════════════════════
#  MazeRunner — Procedurally generated mazes where MEMORY MATTERS
#  The maze is FIXED per episode reset — so remembering dead ends
#  from step 5 helps at step 20. THIS is where memory shines.
# ═══════════════════════════════════════════════════════════════

from __future__ import annotations
import random
import logging
from typing import Any, Dict, List, Tuple, Set

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace

logger = logging.getLogger("cognicore.envs.maze")


class MazeRunnerEnv(CogniCoreEnv):
    """Procedurally generated maze with FIXED walls.

    Unlike GridWorld (random traps each episode), the maze structure
    is CONSISTENT within an episode. An agent that remembers
    "wall at (3,2)" from step 2 won't walk into it at step 15.

    THIS is where CogniCore's memory middleware proves its value.

    Difficulty:
      easy:   8x8 maze, wide corridors
      medium: 12x12 maze, narrow corridors
      hard:   16x16 maze, minimal corridors, dead ends
    """

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
    DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

    def _setup(self, **kwargs: Any) -> None:
        self.difficulty = kwargs.get("difficulty", "easy")
        sizes = {"easy": 8, "medium": 12, "hard": 16}
        self.size = sizes.get(self.difficulty, 8)
        self.max_steps = self.size * self.size * 3
        self.num_tasks = kwargs.get("num_tasks", self.max_steps)

        self.action_space = DiscreteSpace(n=4, labels=self.ACTIONS)
        self.observation_space = DictSpace(fields={
            "agent_pos": "Current position",
            "goal_pos": "Goal position",
            "walls_nearby": "Wall positions within vision",
            "visited_count": "Times current cell was visited",
            "steps_remaining": "Steps left",
        })

        self.walls: Set[Tuple[int, int]] = set()
        self.agent_pos = (1, 1)
        self.goal_pos = (self.size - 2, self.size - 2)
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self.steps_taken = 0

    def _generate_maze(self) -> None:
        """Generate maze using recursive backtracker (DFS)."""
        self.walls = set()
        # Start with all walls
        for r in range(self.size):
            for c in range(self.size):
                self.walls.add((r, c))

        # Carve paths using DFS
        stack = [(1, 1)]
        self.walls.discard((1, 1))

        while stack:
            current = stack[-1]
            r, c = current

            # Find unvisited neighbors (2 cells away)
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.size - 1 and 1 <= nc < self.size - 1:
                    if (nr, nc) in self.walls:
                        neighbors.append((nr, nc, r + dr // 2, c + dc // 2))

            if neighbors:
                nr, nc, wr, wc = random.choice(neighbors)
                self.walls.discard((nr, nc))
                self.walls.discard((wr, wc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # Ensure goal is reachable
        self.walls.discard(self.goal_pos)
        # Clear around goal
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = self.goal_pos[0] + dr, self.goal_pos[1] + dc
            if 1 <= nr < self.size - 1 and 1 <= nc < self.size - 1:
                self.walls.discard((nr, nc))

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        self._generate_maze()
        self.agent_pos = (1, 1)
        self.visit_counts = {(1, 1): 1}
        self.steps_taken = 0
        return [{"step": i} for i in range(self.max_steps)]

    def _get_obs(self) -> Dict[str, Any]:
        vision = 3
        nearby_walls = []
        for dr in range(-vision, vision + 1):
            for dc in range(-vision, vision + 1):
                pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
                if pos in self.walls:
                    nearby_walls.append(list(pos))

        return {
            "agent_pos": list(self.agent_pos),
            "goal_pos": list(self.goal_pos),
            "walls_nearby": nearby_walls,
            "visited_count": self.visit_counts.get(self.agent_pos, 0),
            "steps_remaining": self.max_steps - self.steps_taken,
            "distance_to_goal": abs(self.goal_pos[0] - self.agent_pos[0])
                + abs(self.goal_pos[1] - self.agent_pos[1]),
            "maze_size": self.size,
        }

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        move = action.get("action", "UP")
        if isinstance(move, int):
            move = self.ACTIONS[move % 4]
        move = str(move).upper()

        dr, dc = self.DELTAS.get(move, (0, 0))
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
        new_pos = (new_r, new_c)

        self.steps_taken += 1

        # Hit a wall — stay in place, waste a step
        if new_pos in self.walls or not (0 <= new_r < self.size and 0 <= new_c < self.size):
            return EvalResult(
                base_score=0.0, correct=False, category="navigation",
                ground_truth="AVOID_WALL", predicted=move,
                metadata={"event": "wall", "wasted_step": True},
            )

        old_dist = abs(self.goal_pos[0] - self.agent_pos[0]) + abs(self.goal_pos[1] - self.agent_pos[1])
        self.agent_pos = new_pos
        self.visit_counts[new_pos] = self.visit_counts.get(new_pos, 0) + 1
        new_dist = abs(self.goal_pos[0] - new_pos[0]) + abs(self.goal_pos[1] - new_pos[1])

        # Reached goal
        if new_pos == self.goal_pos:
            efficiency = 1.0 - (self.steps_taken / self.max_steps)
            score = 0.5 + 0.5 * max(0, efficiency)
            self._current_step = len(self._tasks) - 1
            return EvalResult(
                base_score=score, correct=True, category="navigation",
                ground_truth="GOAL", predicted=move,
                metadata={"event": "goal", "steps": self.steps_taken,
                          "efficiency": round(efficiency, 3)},
            )

        # Revisiting a cell = bad (agent is going in circles)
        revisit_penalty = -0.05 * (self.visit_counts[new_pos] - 1)
        moving_closer = new_dist < old_dist
        direction_bonus = 0.1 if moving_closer else -0.02

        score = max(0, 0.2 + direction_bonus + revisit_penalty)

        if self.steps_taken >= self.max_steps:
            self._current_step = len(self._tasks) - 1

        return EvalResult(
            base_score=score, correct=moving_closer, category="navigation",
            ground_truth="NAVIGATE", predicted=move,
            metadata={"event": "step", "distance": new_dist,
                      "revisits": self.visit_counts[new_pos],
                      "moving_closer": moving_closer},
        )

    def render(self) -> str:
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row.append("A")
                elif pos == self.goal_pos:
                    row.append("G")
                elif pos in self.walls:
                    row.append("#")
                elif pos in self.visit_counts:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)
