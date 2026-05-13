"""
CogniCore Multi-Agent Coordination Environment.

Agents coordinate to gather resources, avoid conflicts, and complete
objectives. Memory tracks successful strategies; reflection analyzes
team coordination failures.

Usage::

    env = cognicore.make("MultiAgent-v1", difficulty="medium")
    obs, info = env.reset()
    action = {"agent_0": "gather", "agent_1": "scout", "agent_2": "build"}
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult


# ── Task data ──────────────────────────────────────────────────────

SCENARIOS = {
    "easy": [
        {
            "id": "resource_collect",
            "desc": "Three drones must collect 5 scattered resources and return to base.",
            "agents": ["Drone-A", "Drone-B", "Drone-C"],
            "resources": 5,
            "obstacles": 1,
            "optimal_strategy": "split",
            "actions": ["gather", "scout", "return", "wait"],
        },
        {
            "id": "convoy_escort",
            "desc": "Escort a supply convoy through a corridor. One agent leads, others flank.",
            "agents": ["Lead", "Flank-L", "Flank-R"],
            "resources": 0,
            "obstacles": 2,
            "optimal_strategy": "formation",
            "actions": ["advance", "hold", "flank", "retreat"],
        },
        {
            "id": "area_sweep",
            "desc": "Sweep a grid area for anomalies. Agents must cover all sectors.",
            "agents": ["Scanner-1", "Scanner-2", "Scanner-3"],
            "resources": 0,
            "obstacles": 0,
            "optimal_strategy": "distribute",
            "actions": ["scan_north", "scan_south", "scan_east", "scan_west"],
        },
        {
            "id": "relay_chain",
            "desc": "Form a communication relay chain between two distant points.",
            "agents": ["Node-A", "Node-B", "Node-C"],
            "resources": 0,
            "obstacles": 1,
            "optimal_strategy": "chain",
            "actions": ["move_forward", "hold_position", "boost_signal", "retreat"],
        },
        {
            "id": "cargo_sort",
            "desc": "Sort 5 cargo containers into 3 depots by type. Agents coordinate to avoid collisions.",
            "agents": ["Lifter-1", "Lifter-2", "Lifter-3"],
            "resources": 5,
            "obstacles": 0,
            "optimal_strategy": "assign",
            "actions": ["pick_A", "pick_B", "pick_C", "deliver"],
        },
    ],
    "medium": [
        {
            "id": "search_rescue",
            "desc": "Search-and-rescue in a disaster zone. Locate 3 survivors and extract them.",
            "agents": ["Recon", "Medic", "Extractor", "Relay"],
            "resources": 3,
            "obstacles": 4,
            "optimal_strategy": "role_based",
            "actions": ["search", "mark", "extract", "relay", "wait"],
        },
        {
            "id": "warehouse_ops",
            "desc": "Warehouse robots fulfill 8 orders simultaneously without path conflicts.",
            "agents": ["Bot-1", "Bot-2", "Bot-3", "Bot-4"],
            "resources": 8,
            "obstacles": 3,
            "optimal_strategy": "zone_assign",
            "actions": ["pick", "place", "move_aisle", "yield", "charge"],
        },
        {
            "id": "traffic_control",
            "desc": "Manage traffic at a 4-way intersection. Minimize wait times, prevent collisions.",
            "agents": ["Signal-N", "Signal-S", "Signal-E", "Signal-W"],
            "resources": 0,
            "obstacles": 0,
            "optimal_strategy": "phase_rotate",
            "actions": ["green", "yellow", "red", "emergency_stop"],
        },
    ],
    "hard": [
        {
            "id": "swarm_defense",
            "desc": "Defend a perimeter against 10 incoming threats with 5 agents. Resources limited.",
            "agents": ["Guard-1", "Guard-2", "Guard-3", "Guard-4", "Guard-5"],
            "resources": 10,
            "obstacles": 6,
            "optimal_strategy": "dynamic_reposition",
            "actions": ["intercept", "hold", "reinforce", "retreat", "resupply", "signal"],
        },
        {
            "id": "supply_chain",
            "desc": "Optimize a supply chain with 5 nodes. Balance inventory, minimize delays, handle disruptions.",
            "agents": ["Supplier", "Factory", "Warehouse", "Distributor", "Retailer"],
            "resources": 15,
            "obstacles": 5,
            "optimal_strategy": "pipeline_balance",
            "actions": ["produce", "ship", "store", "distribute", "reorder", "hold"],
        },
    ],
}


class MultiAgentEnv(CogniCoreEnv):
    """Multi-agent coordination environment with cognitive middleware."""

    def _setup(self, difficulty: str = "easy", **kw):
        self.difficulty = difficulty
        self.scenarios = SCENARIOS.get(difficulty, SCENARIOS["easy"])
        self._task_idx = 0
        self.action_space = {"type": "dict", "keys": ["assignments"]}
        self.observation_space = {"type": "dict", "keys": [
            "scenario", "agents", "resources_remaining", "step",
            "conflicts", "completed_tasks"
        ]}

    def _generate_tasks(self):
        tasks = []
        for sc in self.scenarios:
            tasks.append({
                "scenario": sc["desc"],
                "agents": sc["agents"],
                "actions": sc["actions"],
                "optimal": sc["optimal_strategy"],
            })
        self._current = self.scenarios[0]
        self._step_in_task = 0
        self._completed = 0
        self._conflicts = 0
        self._agent_positions = {a: random.randint(0, 5) for a in self._current["agents"]}
        return tasks

    def _get_obs(self) -> dict:
        sc = self._current
        return {
            "scenario": sc["desc"],
            "agents": sc["agents"],
            "resources_remaining": sc["resources"] - self._completed,
            "step": self._step_in_task,
            "conflicts": self._conflicts,
            "completed_tasks": self._completed,
        }

    def _evaluate(self, action: Any) -> EvalResult:
        sc = self._current
        self._step_in_task += 1

        # Parse action — expect dict of agent assignments or a string
        if isinstance(action, dict):
            assignments = action.get("assignments", {})
        elif isinstance(action, str):
            assignments = {sc["agents"][0]: action}
        else:
            assignments = {sc["agents"][0]: str(action)}

        # Check for conflicts (agents doing same thing at same position)
        actions_taken = list(assignments.values())
        has_conflict = len(actions_taken) != len(set(actions_taken))
        if has_conflict:
            self._conflicts += 1

        # Check strategy alignment
        strategy_bonus = 0
        if sc["optimal_strategy"] == "split" and len(set(actions_taken)) >= 2:
            strategy_bonus = 0.3
        elif sc["optimal_strategy"] == "formation" and "advance" in actions_taken:
            strategy_bonus = 0.2
        elif sc["optimal_strategy"] == "role_based" and len(assignments) >= 3:
            strategy_bonus = 0.3

        # Resource completion
        if "gather" in actions_taken or "pick" in actions_taken or "extract" in actions_taken:
            if sc["resources"] > 0:
                self._completed = min(self._completed + 1, sc["resources"])

        score = max(0, 1.0 - self._conflicts * 0.2 + strategy_bonus)
        correct = score >= 0.7

        return EvalResult(
            base_score=min(1.0, score),
            correct=correct,
            ground_truth=sc["optimal_strategy"],
            predicted=str(assignments),
            category="multi_agent",
            metadata={"feedback": f"{'Coordinated' if correct else 'Conflict'}: {self._completed}/{sc['resources']} resources, {self._conflicts} conflicts"},
        )
