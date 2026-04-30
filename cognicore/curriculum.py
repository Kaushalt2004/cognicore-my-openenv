"""
CogniCore Curriculum — Adaptive difficulty progression.

Automatically adjusts environment difficulty based on agent performance.
Agents progress from easy -> medium -> hard as they demonstrate mastery.

Usage::

    from cognicore.curriculum import CurriculumRunner

    runner = CurriculumRunner(
        env_id="SafetyClassification-v1",
        promotion_threshold=0.8,
        demotion_threshold=0.3,
    )
    runner.run(agent, max_episodes=30)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import cognicore
import logging

logger = logging.getLogger("cognicore.curriculum")


class CurriculumRunner:
    """Adaptive curriculum that progresses difficulty automatically.

    Promotes to harder difficulty when accuracy exceeds
    ``promotion_threshold`` for ``window`` consecutive episodes.
    Demotes when accuracy drops below ``demotion_threshold``.
    """

    LEVELS = ["easy", "medium", "hard"]

    def __init__(
        self,
        env_id: str,
        promotion_threshold: float = 0.8,
        demotion_threshold: float = 0.3,
        window: int = 3,
        **env_kwargs,
    ):
        self.env_id = env_id
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.window = window
        self.env_kwargs = env_kwargs
        self.current_level = 0
        self.history: List[Dict[str, Any]] = []
        self._callbacks: List[Callable] = []

    @property
    def difficulty(self) -> str:
        return self.LEVELS[self.current_level]

    def on_level_change(self, callback: Callable):
        """Register a callback for difficulty changes."""
        self._callbacks.append(callback)

    def run(
        self,
        agent=None,
        max_episodes: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run curriculum through episodes, adapting difficulty.

        Parameters
        ----------
        agent : BaseAgent or None
            Agent to run. None uses RandomAgent.
        max_episodes : int
            Maximum total episodes across all levels.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Summary with history, final level, and stats.
        """
        if agent is None:
            from cognicore.agents.base_agent import RandomAgent

        for ep in range(1, max_episodes + 1):
            env = cognicore.make(
                self.env_id,
                difficulty=self.difficulty,
                **self.env_kwargs,
            )

            if agent is None:
                from cognicore.agents.base_agent import RandomAgent

                _agent = RandomAgent(env.action_space)
            else:
                _agent = agent

            obs = env.reset()
            while True:
                action = _agent.act(obs)
                obs, reward, done, _, info = env.step(action)
                if done:
                    break

            stats = env.episode_stats()
            record = {
                "episode": ep,
                "difficulty": self.difficulty,
                "level": self.current_level,
                "accuracy": stats.accuracy,
                "score": env.get_score(),
                "correct": stats.correct_count,
                "total": stats.steps,
            }
            self.history.append(record)

            if verbose:
                print(
                    f"  Ep {ep:3d} [{self.difficulty:6s}] "
                    f"accuracy={stats.accuracy:.0%} "
                    f"score={env.get_score():.4f} "
                    f"correct={stats.correct_count}/{stats.steps}"
                )

            # Check promotion / demotion
            recent = self.history[-self.window :]
            recent_acc = [
                r["accuracy"] for r in recent if r["level"] == self.current_level
            ]

            if len(recent_acc) >= self.window:
                avg = sum(recent_acc) / len(recent_acc)

                if (
                    avg >= self.promotion_threshold
                    and self.current_level < len(self.LEVELS) - 1
                ):
                    old = self.difficulty
                    self.current_level += 1
                    if verbose:
                        logger.info(f"  >> PROMOTED: {old} -> {self.difficulty}")
                    for cb in self._callbacks:
                        cb("promote", old, self.difficulty)

                elif avg <= self.demotion_threshold and self.current_level > 0:
                    old = self.difficulty
                    self.current_level -= 1
                    if verbose:
                        logger.info(f"  >> DEMOTED: {old} -> {self.difficulty}")
                    for cb in self._callbacks:
                        cb("demote", old, self.difficulty)

        return {
            "episodes": len(self.history),
            "final_difficulty": self.difficulty,
            "final_level": self.current_level,
            "history": self.history,
            "avg_accuracy": sum(h["accuracy"] for h in self.history)
            / len(self.history),
            "avg_score": sum(h["score"] for h in self.history) / len(self.history),
        }
