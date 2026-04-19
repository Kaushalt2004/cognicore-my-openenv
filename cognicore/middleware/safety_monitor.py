"""
CogniCore Safety Monitor — Streak detection and performance degradation.

Evolved from the AI-Safety-specific ``Safety`` class into a
general-purpose agent performance monitor.

Features:
  - Wrong-streak penalty (n consecutive errors)
  - Rolling-window accuracy degradation detection
  - Health status: healthy / warning / critical
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict


class SafetyMonitor:
    """Monitors agent performance and applies corrective signals.

    Parameters
    ----------
    streak_threshold : int
        Number of consecutive errors before a penalty is applied.
    streak_penalty : float
        Penalty amount (should be negative).
    degradation_window : int
        Rolling window size for degradation detection.
    degradation_threshold : float
        If rolling accuracy drops below this, status becomes "warning".
    """

    def __init__(
        self,
        streak_threshold: int = 3,
        streak_penalty: float = -0.1,
        degradation_window: int = 10,
        degradation_threshold: float = 0.3,
    ) -> None:
        self.streak_threshold = streak_threshold
        self.streak_penalty = streak_penalty
        self.degradation_window = degradation_window
        self.degradation_threshold = degradation_threshold

        self.wrong_streak = 0
        self._blocked_count = 0
        self._allowed_count = 0
        self._history: deque = deque(maxlen=degradation_window)

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(self, correct: bool) -> float:
        """Record a step result and return any streak penalty.

        Parameters
        ----------
        correct : bool
            Whether the agent's action was correct.

        Returns
        -------
        float
            ``0.0`` normally, or ``streak_penalty`` (negative) when
            the wrong streak reaches the threshold.
        """
        self._history.append(correct)

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

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Return the agent's health status.

        Returns
        -------
        str
            ``"healthy"``, ``"warning"``, or ``"critical"``.
        """
        if self.wrong_streak >= self.streak_threshold:
            return "critical"

        if len(self._history) >= self.degradation_window:
            rolling_acc = sum(self._history) / len(self._history)
            if rolling_acc < self.degradation_threshold:
                return "warning"

        return "healthy"

    def get_wrong_streak(self) -> int:
        """Return the current wrong-streak count."""
        return self.wrong_streak

    def rolling_accuracy(self) -> float:
        """Return rolling-window accuracy (or 0.0 if no history)."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    # ------------------------------------------------------------------
    # Reset / stats
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset streak counter (e.g. on episode reset)."""
        self.wrong_streak = 0

    def full_reset(self) -> None:
        """Reset everything including history and counters."""
        self.wrong_streak = 0
        self._blocked_count = 0
        self._allowed_count = 0
        self._history.clear()

    def block_rate(self) -> float:
        """Fraction of steps that received a streak penalty."""
        total = self._blocked_count + self._allowed_count
        return self._blocked_count / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "blocked_count": self._blocked_count,
            "allowed_count": self._allowed_count,
            "block_rate": self.block_rate(),
            "current_wrong_streak": self.wrong_streak,
            "streak_threshold": self.streak_threshold,
            "rolling_accuracy": self.rolling_accuracy(),
            "status": self.status(),
        }
