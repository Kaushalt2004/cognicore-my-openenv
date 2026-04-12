"""
CogniCore Reflection — Analyzes past patterns to improve future decisions.

Tracks which categories the agent struggles with and provides
hints when it makes mistakes. Triggers on FIRST mistake (not 2nd).
Also detects cross-category patterns.
"""

from typing import Dict, Any, Optional, Tuple
from cognicore.memory.vector_memory import VectorMemory


class Reflection:
    """Reflection engine that learns from past classification mistakes.

    Tracks failure patterns by category and provides actionable hints
    when the agent misclassifies cases. Now with:
    1. First-mistake triggering (no waiting for 2+ errors)
    2. Cross-category pattern detection
    3. Accuracy alerts
    """

    def __init__(self, memory: VectorMemory):
        self.memory = memory
        self._suggestion_count = 0
        self._override_count = 0

    def analyze(self, category: str) -> Dict[str, Any]:
        """Analyze past performance in a given category."""
        entries = self.memory.retrieve(category, top_k=50)

        if not entries:
            return {
                "n_similar": 0,
                "good_predictions": {},
                "bad_predictions": {},
                "recommendation": None,
            }

        good = {}
        bad = {}

        for entry in entries:
            predicted = entry["predicted"]
            if entry["correct"]:
                good[predicted] = good.get(predicted, 0) + 1
            else:
                bad[predicted] = bad.get(predicted, 0) + 1

        recommendation = None
        if good:
            recommendation = max(good, key=good.get)

        return {
            "n_similar": len(entries),
            "good_predictions": good,
            "bad_predictions": bad,
            "recommendation": recommendation,
        }

    def suggest_action(
        self, category: str, model_prediction: str
    ) -> Tuple[str, str]:
        """Suggest whether to keep or override the model's prediction."""
        self._suggestion_count += 1

        analysis = self.analyze(category)
        bad = analysis["bad_predictions"]
        recommendation = analysis["recommendation"]

        # Override if model's prediction has been wrong even once
        if model_prediction in bad and bad[model_prediction] >= 1:
            if recommendation and recommendation != model_prediction:
                self._override_count += 1
                return recommendation, "reflection_override"

        return model_prediction, "model_action"

    def get_reflection_hint(self, category: str) -> Optional[str]:
        """Generate a natural-language reflection hint for the agent.

        Now triggers on FIRST mistake and includes cross-category patterns.
        """
        hints = []

        # ── Signal 1: Same-category mistakes (triggers on 1st mistake) ──
        analysis = self.analyze(category)
        bad = analysis["bad_predictions"]

        if bad:
            worst_prediction = max(bad, key=bad.get)
            fail_count = bad[worst_prediction]
            hints.append(
                f"REFLECTION: In '{category}' cases, predicting "
                f"'{worst_prediction}' was wrong {fail_count} time(s)."
            )
            if analysis["recommendation"]:
                hints.append(
                    f"Consider '{analysis['recommendation']}' instead."
                )

        # ── Signal 2: Cross-category patterns ──
        all_entries = self.memory.entries[-30:]  # last 30 across all categories
        if len(all_entries) >= 3:
            wrong_all = [e for e in all_entries if not e["correct"]]
            if wrong_all:
                wrong_preds = {}
                for e in wrong_all:
                    wrong_preds[e["predicted"]] = wrong_preds.get(e["predicted"], 0) + 1
                worst_global = max(wrong_preds, key=wrong_preds.get)
                if wrong_preds[worst_global] >= 2:
                    hints.append(
                        f"PATTERN: You incorrectly predicted "
                        f"'{worst_global}' {wrong_preds[worst_global]} times across categories."
                    )

                # Accuracy alert
                total = len(all_entries)
                wrong = len(wrong_all)
                acc = (total - wrong) / total
                if acc < 0.7:
                    hints.append(
                        f"ACCURACY ALERT: {acc:.0%} overall ({wrong}/{total} wrong)."
                    )

        return " | ".join(hints) if hints else None

    def override_rate(self) -> float:
        """Return the fraction of suggestions that were overrides."""
        if self._suggestion_count == 0:
            return 0.0
        return self._override_count / self._suggestion_count

    def stats(self) -> Dict[str, Any]:
        """Return reflection statistics."""
        return {
            "total_suggestions": self._suggestion_count,
            "overrides": self._override_count,
            "override_rate": self.override_rate(),
        }
