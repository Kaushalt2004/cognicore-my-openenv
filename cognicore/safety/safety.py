"""
CogniCore Safety — Built-in safety layer for the environment.

In the Colab notebook, Safety blocks actions in unsafe FrozenLake states.
In the AI Safety Monitor, the safety layer tracks wrong-streak penalties
and flags when the agent is consistently misclassifying cases.
"""

from typing import Dict, Any, Optional


class Safety:
    """Safety layer that monitors agent performance and applies penalties.

    Tracks consecutive wrong answers (wrong streaks) and applies
    penalties when the agent makes 3+ wrong classifications in a row,
    incentivizing the agent to self-correct.
    """

    def __init__(self, streak_threshold: int = 3, streak_penalty: float = -0.1):
        self.streak_threshold = streak_threshold
        self.streak_penalty = streak_penalty
        self.wrong_streak = 0
        self._blocked_count = 0
        self._allowed_count = 0

    def check(self, correct: bool) -> float:
        """Check whether to apply a safety penalty.

        Args:
            correct: Whether the agent's last classification was correct.

        Returns:
            Penalty value (0.0 or negative).
        """
        if correct:
            self.wrong_streak = 0
            self._allowed_count += 1
            return 0.0
        else:
            self.wrong_streak += 1
            if self.wrong_streak >= self.streak_threshold:
                self._blocked_count += 1
                return self.streak_penalty
            self._allowed_count += 1
            return 0.0

    def get_wrong_streak(self) -> int:
        """Return current wrong streak count."""
        return self.wrong_streak

    def reset(self) -> None:
        """Reset the wrong streak counter (called on episode reset)."""
        self.wrong_streak = 0

    def block_rate(self) -> float:
        """Return the fraction of steps that received a penalty."""
        total = self._blocked_count + self._allowed_count
        if total == 0:
            return 0.0
        return self._blocked_count / total

    def stats(self) -> Dict[str, Any]:
        """Return safety statistics."""
        return {
            "blocked_count": self._blocked_count,
            "allowed_count": self._allowed_count,
            "block_rate": self.block_rate(),
            "current_wrong_streak": self.wrong_streak,
            "streak_threshold": self.streak_threshold,
        }
