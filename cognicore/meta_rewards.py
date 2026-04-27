"""
CogniCore Meta Rewards — Self-evolving reward system.

The system learns how to reward better by tracking which reward
components actually improve agent performance, then auto-tunes weights.

Usage::

    from cognicore.meta_rewards import MetaRewardOptimizer

    meta = MetaRewardOptimizer()
    meta.observe(reward, accuracy_improved=True)
    new_weights = meta.optimize()
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


class MetaRewardOptimizer:
    """Self-evolving reward system — learns how to reward better.

    Tracks which reward component weights correlate with improvement,
    then adjusts weights to maximize learning speed.

    Parameters
    ----------
    components : list of str
        Reward component names to optimize.
    learning_rate : float
        How fast weights change per optimization step.
    """

    def __init__(
        self,
        components: List[str] = None,
        learning_rate: float = 0.01,
    ):
        self.components = components or [
            "memory_bonus",
            "reflection_bonus",
            "streak_penalty",
            "novelty_bonus",
            "confidence_cal",
            "propose_bonus",
        ]
        self.learning_rate = learning_rate
        self.weights: Dict[str, float] = {c: 1.0 for c in self.components}
        self._history: List[Dict] = []
        self._generation = 0

    def observe(
        self,
        reward_components: Dict[str, float],
        accuracy_improved: bool,
        accuracy: float = 0.0,
    ):
        """Record a reward observation and whether it led to improvement."""
        self._history.append(
            {
                "components": reward_components,
                "improved": accuracy_improved,
                "accuracy": accuracy,
                "weights": copy.deepcopy(self.weights),
            }
        )

    def optimize(self, verbose: bool = False) -> Dict[str, float]:
        """Optimize reward weights based on observed correlations.

        For each component, checks: when this component was high,
        did the agent improve? If yes, increase weight. If no, decrease.
        """
        if len(self._history) < 4:
            return self.weights

        self._generation += 1

        # Split history into improving vs not-improving
        improving = [h for h in self._history if h["improved"]]
        declining = [h for h in self._history if not h["improved"]]

        if not improving or not declining:
            return self.weights

        for comp in self.components:
            # Average component value when improving vs declining
            imp_avg = sum(abs(h["components"].get(comp, 0)) for h in improving) / len(
                improving
            )
            dec_avg = sum(abs(h["components"].get(comp, 0)) for h in declining) / len(
                declining
            )

            # If component is higher when improving → increase weight
            if imp_avg > dec_avg:
                self.weights[comp] += self.learning_rate
            elif dec_avg > imp_avg:
                self.weights[comp] = max(0.1, self.weights[comp] - self.learning_rate)

            # Clamp
            self.weights[comp] = max(0.1, min(3.0, self.weights[comp]))

        if verbose:
            print(f"\n  Meta-Reward Optimization (gen {self._generation}):")
            for comp, w in sorted(self.weights.items(), key=lambda x: -x[1]):
                bar = "█" * int(w * 10) + "░" * (30 - int(w * 10))
                print(f"    {comp:20s} [{bar}] {w:.2f}")

        return self.weights

    def apply_weights(self, reward) -> float:
        """Apply learned weights to a StructuredReward."""
        total = getattr(reward, "base_score", 0)
        for comp in self.components:
            val = getattr(reward, comp, 0)
            total += val * self.weights.get(comp, 1.0)
        return total

    def stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "observations": len(self._history),
            "weights": dict(self.weights),
            "improving_rate": (
                sum(1 for h in self._history if h["improved"]) / len(self._history)
                if self._history
                else 0
            ),
        }
