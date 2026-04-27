"""
CogniCore Predictive Failure — Predict failure BEFORE it happens.

Uses trailing performance windows, category risk profiles, and
streak momentum to forecast whether the agent will fail on the
next step — and what the risk level is.

Usage::

    from cognicore.predictive import FailurePredictor

    predictor = FailurePredictor()
    # feed it step outcomes
    predictor.observe(category="security", correct=False, confidence=0.9)
    predictor.observe(category="security", correct=False, confidence=0.8)
    risk = predictor.predict_risk(next_category="security")
    # -> {"risk": 0.87, "level": "CRITICAL", "reasons": [...]}
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional


class FailurePredictor:
    """Predicts agent failure before it happens.

    Tracks three signals:
    1. **Category risk** — per-category failure rate
    2. **Streak momentum** — consecutive failures increase risk
    3. **Confidence calibration** — overconfident+wrong = danger sign

    Parameters
    ----------
    window : int
        Number of recent steps to consider.
    streak_weight : float
        How much streak contributes to risk (0-1).
    calibration_weight : float
        How much confidence miscalibration contributes (0-1).
    """

    def __init__(
        self,
        window: int = 20,
        streak_weight: float = 0.3,
        calibration_weight: float = 0.2,
    ):
        self.window = window
        self.streak_weight = streak_weight
        self.calibration_weight = calibration_weight

        # Tracking
        self._history: deque = deque(maxlen=window * 5)
        self._category_stats: Dict[str, Dict] = defaultdict(
            lambda: {"correct": 0, "wrong": 0}
        )
        self._streak = 0  # negative = failure streak
        self._confidences: List[Dict] = []
        self._alerts: List[Dict] = []

    def observe(
        self,
        category: str,
        correct: bool,
        confidence: float = 0.5,
        step: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Record an observation and check for early warnings.

        Returns an alert dict if risk is high, None otherwise.
        """
        self._history.append({
            "category": category,
            "correct": correct,
            "confidence": confidence,
            "step": step,
        })

        # Update category stats
        self._category_stats[category]["correct" if correct else "wrong"] += 1

        # Update streak
        if correct:
            self._streak = max(self._streak + 1, 1)
        else:
            self._streak = min(self._streak - 1, -1)

        # Track confidence calibration
        self._confidences.append({
            "confidence": confidence,
            "correct": correct,
            "overconfident": confidence > 0.7 and not correct,
        })

        # Check for early warning
        risk = self.predict_risk(category)
        if risk["risk"] >= 0.7:
            alert = {
                "type": "EARLY_WARNING",
                "step": step,
                "category": category,
                "risk": risk["risk"],
                "level": risk["level"],
                "reasons": risk["reasons"],
            }
            self._alerts.append(alert)
            return alert

        return None

    def predict_risk(self, next_category: str = "") -> Dict[str, Any]:
        """Predict failure risk for the next step.

        Returns
        -------
        dict with:
          - risk: float 0-1 (probability of failure)
          - level: "LOW", "MEDIUM", "HIGH", "CRITICAL"
          - reasons: list of contributing factors
          - trajectory: "improving", "declining", "stable"
        """
        reasons = []
        risk_factors = []

        # 1. Category-based risk
        cat_risk = self._category_risk(next_category)
        risk_factors.append(cat_risk)
        if cat_risk > 0.5:
            stats = self._category_stats.get(next_category, {})
            total = stats.get("correct", 0) + stats.get("wrong", 0)
            reasons.append(
                f"Category '{next_category}' has {cat_risk:.0%} failure rate "
                f"({stats.get('wrong', 0)}/{total})"
            )

        # 2. Streak momentum
        streak_risk = self._streak_risk()
        risk_factors.append(streak_risk * self.streak_weight)
        if self._streak <= -2:
            reasons.append(f"Failure streak of {abs(self._streak)} — momentum trending down")

        # 3. Confidence calibration
        cal_risk = self._calibration_risk()
        risk_factors.append(cal_risk * self.calibration_weight)
        if cal_risk > 0.5:
            reasons.append("Agent is overconfident — high confidence but frequent errors")

        # 4. Recent window trend
        trend_risk = self._trend_risk()
        risk_factors.append(trend_risk * 0.2)
        if trend_risk > 0.6:
            reasons.append("Performance declining in recent window")

        # Combine
        combined = min(1.0, sum(risk_factors) / max(len(risk_factors), 1))

        # Overall trajectory
        trajectory = self._get_trajectory()

        if combined >= 0.8:
            level = "CRITICAL"
        elif combined >= 0.6:
            level = "HIGH"
        elif combined >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        if not reasons:
            reasons.append("No significant risk factors detected")

        return {
            "risk": round(combined, 3),
            "level": level,
            "reasons": reasons,
            "trajectory": trajectory,
            "streak": self._streak,
            "category_risk": round(cat_risk, 3),
        }

    def _category_risk(self, category: str) -> float:
        """Failure probability for this category."""
        stats = self._category_stats.get(category)
        if not stats:
            return 0.5  # unknown category = moderate risk
        total = stats["correct"] + stats["wrong"]
        if total == 0:
            return 0.5
        return stats["wrong"] / total

    def _streak_risk(self) -> float:
        """Risk from consecutive failures."""
        if self._streak >= 0:
            return 0.0
        # Scale: -1=0.2, -2=0.4, -3=0.6, -4+=0.8
        return min(1.0, abs(self._streak) * 0.2)

    def _calibration_risk(self) -> float:
        """Risk from overconfidence."""
        recent = self._confidences[-self.window:]
        if not recent:
            return 0.0
        overconfident = sum(1 for c in recent if c["overconfident"])
        return overconfident / len(recent)

    def _trend_risk(self) -> float:
        """Risk from declining performance trend."""
        recent = list(self._history)[-self.window:]
        if len(recent) < 4:
            return 0.3  # not enough data

        half = len(recent) // 2
        first_half = sum(1 for s in recent[:half] if s["correct"]) / half
        second_half = sum(1 for s in recent[half:] if s["correct"]) / max(len(recent) - half, 1)

        if second_half < first_half:
            return min(1.0, (first_half - second_half) * 2)
        return 0.0

    def _get_trajectory(self) -> str:
        """Determine overall performance trajectory."""
        recent = list(self._history)[-self.window:]
        if len(recent) < 6:
            return "insufficient_data"

        third = len(recent) // 3
        early = sum(1 for s in recent[:third] if s["correct"]) / max(third, 1)
        late = sum(1 for s in recent[-third:] if s["correct"]) / max(third, 1)

        if late > early + 0.1:
            return "improving"
        elif late < early - 0.1:
            return "declining"
        return "stable"

    def get_alerts(self, last_n: int = 10) -> List[Dict]:
        """Get recent early warning alerts."""
        return self._alerts[-last_n:]

    def risk_report(self) -> Dict[str, Any]:
        """Generate a comprehensive risk report."""
        categories = {}
        for cat, stats in self._category_stats.items():
            total = stats["correct"] + stats["wrong"]
            categories[cat] = {
                "failure_rate": stats["wrong"] / total if total else 0,
                "total": total,
                "correct": stats["correct"],
                "wrong": stats["wrong"],
            }

        # Sort by risk
        risky = sorted(categories.items(), key=lambda x: -x[1]["failure_rate"])

        return {
            "total_observations": len(self._history),
            "current_streak": self._streak,
            "trajectory": self._get_trajectory(),
            "alerts_triggered": len(self._alerts),
            "highest_risk_categories": [
                {"category": cat, **data}
                for cat, data in risky[:5]
            ],
            "overall_accuracy": (
                sum(1 for s in self._history if s["correct"]) /
                len(self._history) if self._history else 0
            ),
        }

    def print_report(self):
        """Print formatted risk report."""
        r = self.risk_report()
        print(f"\n{'=' * 55}")
        print("  Predictive Failure Report")
        print(f"{'=' * 55}")
        print(f"  Observations: {r['total_observations']}")
        print(f"  Accuracy: {r['overall_accuracy']:.0%}")
        print(f"  Trajectory: {r['trajectory']}")
        print(f"  Streak: {r['current_streak']}")
        print(f"  Alerts: {r['alerts_triggered']}")

        if r["highest_risk_categories"]:
            print("\n  Riskiest categories:")
            for c in r["highest_risk_categories"][:5]:
                bar_len = int(c["failure_rate"] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    {c['category']:20s} [{bar}] {c['failure_rate']:.0%} fail")

        print(f"{'=' * 55}\n")
