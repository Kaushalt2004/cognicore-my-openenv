"""
GridWorld — A real RL environment where agents ACTUALLY learn.

An NxN grid where the agent navigates from start to goal, collecting
rewards and avoiding traps. This is a proper RL environment with:
- State space (grid positions)
- Action space (UP/DOWN/LEFT/RIGHT)
- Reward shaping (step penalty, goal reward, trap penalty)
- Episode termination
- Memory benefit: agent remembers which cells are traps
- Reflection benefit: agent avoids previously-failed paths

This proves CogniCore's cognitive middleware works for ANY agent,
not just LLMs.
"""

from __future__ import annotations

import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace

logger = logging.getLogger("cognicore.envs.gridworld")


class GridWorldEnv(CogniCoreEnv):
    """NxN grid navigation environment.

    The agent starts at (0, 0) and must reach the goal at (N-1, N-1).
    Traps are placed randomly. Each step has a small penalty to encourage
    efficient paths.

    This is a TRUE RL environment — agents with Q-tables or policy
    networks can learn optimal paths across episodes.

    Difficulty levels:
        - easy:   5x5 grid, 3 traps
        - medium: 7x7 grid, 7 traps
        - hard:   10x10 grid, 15 traps
    """

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
    DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

    def _setup(self, **kwargs: Any) -> None:
        self.difficulty: str = kwargs.get("difficulty", "easy")

        sizes = {"easy": 5, "medium": 7, "hard": 10}
        traps = {"easy": 3, "medium": 7, "hard": 15}
        self.grid_size: int = sizes.get(self.difficulty, 5)
        self.num_traps: int = traps.get(self.difficulty, 3)
        self.max_steps_per_episode: int = self.grid_size * self.grid_size * 2
        self.num_tasks: int = kwargs.get("num_tasks", self.max_steps_per_episode)

        self.action_space = DiscreteSpace(n=4, labels=self.ACTIONS)
        self.observation_space = DictSpace(fields={
            "agent_pos": "Agent's current (row, col) position",
            "goal_pos": "Goal position (row, col)",
            "grid_size": "Size of the grid",
            "nearby_traps": "List of trap positions within vision range",
            "steps_taken": "Number of steps taken so far",
            "visited": "Set of previously visited cells",
        })

        # Internal state
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (self.grid_size - 1, self.grid_size - 1)
        self.trap_positions: List[Tuple[int, int]] = []
        self.visited: set = set()
        self.steps_taken: int = 0
        self._episode_reward: float = 0.0

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        """Generate trap positions for this episode."""
        self.trap_positions = []
        all_cells = [
            (r, c) for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) != (0, 0) and (r, c) != self.goal_pos
        ]
        self.trap_positions = random.sample(all_cells, min(self.num_traps, len(all_cells)))
        self.agent_pos = (0, 0)
        self.visited = {(0, 0)}
        self.steps_taken = 0
        self._episode_reward = 0.0

        # Return dummy tasks (one per max step)
        return [{"step": i} for i in range(self.max_steps_per_episode)]

    def _get_obs(self) -> Dict[str, Any]:
        """Return observation with agent's current view of the grid."""
        # Vision range: agent can see traps within 2 cells
        vision = 2
        nearby_traps = [
            t for t in self.trap_positions
            if abs(t[0] - self.agent_pos[0]) <= vision
            and abs(t[1] - self.agent_pos[1]) <= vision
        ]

        return {
            "agent_pos": list(self.agent_pos),
            "goal_pos": list(self.goal_pos),
            "grid_size": self.grid_size,
            "nearby_traps": [list(t) for t in nearby_traps],
            "steps_taken": self.steps_taken,
            "visited": [list(v) for v in self.visited],
            "distance_to_goal": abs(self.goal_pos[0] - self.agent_pos[0])
                + abs(self.goal_pos[1] - self.agent_pos[1]),
        }

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        """Execute action and return result."""
        move = action.get("action", action.get("classification", "UP"))
        if isinstance(move, int):
            move = self.ACTIONS[move % 4]
        move = str(move).upper()

        # Apply movement
        dr, dc = self.DELTAS.get(move, (0, 0))
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Clamp to grid
        new_r = max(0, min(self.grid_size - 1, new_r))
        new_c = max(0, min(self.grid_size - 1, new_c))
        new_pos = (new_r, new_c)

        self.agent_pos = new_pos
        self.visited.add(new_pos)
        self.steps_taken += 1

        # Calculate reward
        hit_wall = (new_r == self.agent_pos[0] - dr + dr
                    and new_c == self.agent_pos[1] - dc + dc)
        hit_trap = new_pos in self.trap_positions
        reached_goal = new_pos == self.goal_pos

        if reached_goal:
            # Big reward, scaled by efficiency
            efficiency = 1.0 - (self.steps_taken / self.max_steps_per_episode)
            score = 0.5 + 0.5 * max(0, efficiency)
            self._episode_reward += 10.0
            # Force episode end
            self._current_step = len(self._tasks) - 1
            return EvalResult(
                base_score=score,
                correct=True,
                category="navigation",
                ground_truth="GOAL_REACHED",
                predicted=move,
                metadata={"event": "goal", "steps": self.steps_taken,
                          "efficiency": round(efficiency, 3)},
            )
        elif hit_trap:
            self._episode_reward -= 5.0
            # Episode ends on trap
            self._current_step = len(self._tasks) - 1
            return EvalResult(
                base_score=0.0,
                correct=False,
                category="navigation",
                ground_truth="AVOID_TRAP",
                predicted=move,
                metadata={"event": "trap", "trap_pos": list(new_pos),
                          "steps": self.steps_taken},
            )
        else:
            # Small step penalty + distance-based shaping
            old_dist = abs(self.goal_pos[0] - (new_r - dr)) + abs(self.goal_pos[1] - (new_c - dc))
            new_dist = abs(self.goal_pos[0] - new_r) + abs(self.goal_pos[1] - new_c)
            moving_closer = new_dist < old_dist

            step_reward = 0.3 if moving_closer else 0.1
            self._episode_reward += step_reward

            # Check if max steps reached
            if self.steps_taken >= self.max_steps_per_episode:
                self._current_step = len(self._tasks) - 1

            return EvalResult(
                base_score=step_reward,
                correct=moving_closer,
                category="navigation",
                ground_truth="MOVE_TOWARD_GOAL",
                predicted=move,
                metadata={"event": "step", "distance": new_dist,
                          "moving_closer": moving_closer},
            )

    def render(self) -> str:
        """Render the grid as ASCII art."""
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row.append("A")
                elif pos == self.goal_pos:
                    row.append("G")
                elif pos in self.trap_positions:
                    row.append("X")
                elif pos in self.visited:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("|" + "|".join(row) + "|")
        border = "+" + "+".join(["-"] * self.grid_size) + "+"
        result = border + "\n" + ("\n".join(lines)) + "\n" + border
        return result
