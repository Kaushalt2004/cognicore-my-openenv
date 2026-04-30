"""
CogniCore Difficulty Estimator — Auto-rate how hard each test case is.

Analyzes test cases and estimates their difficulty based on
ambiguity, complexity, and agent performance.

Usage::

    from cognicore.difficulty import DifficultyEstimator

    estimator = DifficultyEstimator()
    estimator.calibrate("SafetyClassification-v1", episodes=5)
    scores = estimator.get_difficulty_map()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import cognicore
from cognicore.agents.base_agent import RandomAgent
import logging

logger = logging.getLogger("cognicore.difficulty")


class DifficultyEstimator:
    """Estimate difficulty of individual test cases.

    Uses multi-agent calibration: runs several agents and measures
    how often each case is answered correctly.
    """

    def __init__(self):
        self._case_stats: Dict[str, Dict] = defaultdict(
            lambda: {"correct": 0, "total": 0, "rewards": [], "categories": set()}
        )

    def calibrate(
        self,
        env_id: str = "SafetyClassification-v1",
        difficulty: str = "easy",
        episodes: int = 5,
        verbose: bool = True,
    ):
        """Calibrate difficulty by running agents."""
        if verbose:
            logger.info(f"\nCalibrating difficulty: {env_id} ({difficulty})")
            logger.info(f"  Episodes: {episodes}")

        for ep in range(episodes):
            env = cognicore.make(env_id, difficulty=difficulty)
            agent = RandomAgent(env.action_space)
            obs = env.reset()

            while True:
                action = agent.act(obs)
                obs, reward, done, _, info = env.step(action)
                er = info.get("eval_result", {})

                prompt = str(obs.get("prompt", ""))[:80]
                key = f"{er.get('category', '?')}:{prompt[:40]}"

                self._case_stats[key]["total"] += 1
                self._case_stats[key]["rewards"].append(reward.total)
                self._case_stats[key]["categories"].add(er.get("category", "?"))
                if er.get("correct"):
                    self._case_stats[key]["correct"] += 1

                if done:
                    break

        if verbose:
            logger.info(f"  Cases tracked: {len(self._case_stats)}")

    def get_difficulty_map(self) -> List[Dict[str, Any]]:
        """Get difficulty scores for all tracked cases.

        Returns list sorted by difficulty (hardest first).
        """
        results = []
        for key, data in self._case_stats.items():
            total = data["total"]
            correct = data["correct"]
            success_rate = correct / total if total else 0

            # Difficulty = 1 - success_rate (hardest = highest)
            difficulty = 1.0 - success_rate
            avg_reward = (
                sum(data["rewards"]) / len(data["rewards"]) if data["rewards"] else 0
            )

            if difficulty >= 0.8:
                level = "VERY_HARD"
            elif difficulty >= 0.6:
                level = "HARD"
            elif difficulty >= 0.4:
                level = "MEDIUM"
            elif difficulty >= 0.2:
                level = "EASY"
            else:
                level = "TRIVIAL"

            results.append(
                {
                    "case": key,
                    "difficulty": round(difficulty, 3),
                    "level": level,
                    "success_rate": round(success_rate, 3),
                    "observations": total,
                    "avg_reward": round(avg_reward, 3),
                }
            )

        return sorted(results, key=lambda x: -x["difficulty"])

    def print_map(self, top_n: int = 20):
        """Print difficulty map."""
        results = self.get_difficulty_map()
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  Difficulty Map ({len(results)} cases)")
        logger.info(f"{'=' * 70}")
        for r in results[:top_n]:
            bar_len = int(r["difficulty"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            logger.info(f"  [{r['level']:10s}] [{bar}] {r['case'][:40]}")
        logger.info(f"{'=' * 70}\n")
