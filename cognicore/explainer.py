"""
CogniCore Explainer — Explainable AI engine for debugging agent behavior.

Answers the question: "WHY did the agent fail?"

Features:
  - Step-by-step reasoning logs (full audit trail)
  - "Why this was wrong" explanations
  - Pattern detection ("you keep confusing X with Y")
  - Actionable improvement suggestions
  - Confusion matrix generation

Usage::

    from cognicore.explainer import Explainer

    exp = Explainer()
    exp.record_step(step=1, category="phishing", predicted="SAFE", truth="UNSAFE", correct=False)
    exp.record_step(step=2, category="cooking", predicted="SAFE", truth="SAFE", correct=True)
    report = exp.explain()
    report.print_report()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


class Explainer:
    """Explainable AI engine that tells you WHY your agent fails.

    Records every decision and produces human-readable explanations
    with actionable improvement suggestions.
    """

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.episode_summaries: List[Dict[str, Any]] = []

    def record_step(
        self,
        step: int,
        category: str,
        predicted: str,
        truth: str,
        correct: bool,
        reward: float = 0,
        confidence: float = 0,
        memory_used: bool = False,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """Record a step and return immediate explanation.

        Returns
        -------
        dict
            Explanation with 'why_wrong', 'suggestion', and 'pattern'.
        """
        record = {
            "step": step,
            "category": category,
            "predicted": predicted,
            "truth": truth,
            "correct": correct,
            "reward": reward,
            "confidence": confidence,
            "memory_used": memory_used,
            "reasoning": reasoning,
        }
        self.steps.append(record)

        # Generate immediate explanation
        explanation = {
            "step": step,
            "verdict": "CORRECT" if correct else "WRONG",
        }

        if not correct:
            explanation["why_wrong"] = self._explain_why(record)
            explanation["suggestion"] = self._suggest_fix(record)
            explanation["similar_past_mistakes"] = self._find_similar_mistakes(record)

        return explanation

    def _explain_why(self, record: Dict) -> str:
        """Generate a human-readable explanation of why the agent was wrong."""
        pred = record["predicted"]
        truth = record["truth"]
        cat = record["category"]
        conf = record["confidence"]

        explanation = f"You predicted '{pred}' but the correct answer was '{truth}' "
        explanation += f"for category '{cat}'. "

        if conf > 0.7:
            explanation += f"You were {conf:.0%} confident, which means your model is overconfident on this type. "
        elif conf < 0.3 and conf > 0:
            explanation += f"Your low confidence ({conf:.0%}) suggests uncertainty — consider using PROPOSE→Revise to explore first. "

        # Check if this is a repeated mistake
        same_cat_fails = [s for s in self.steps[:-1] if s["category"] == cat and not s["correct"]]
        if len(same_cat_fails) >= 2:
            explanation += f"WARNING: You've failed on '{cat}' {len(same_cat_fails)+1} times now. This is a systematic weakness. "

        return explanation

    def _suggest_fix(self, record: Dict) -> str:
        """Generate actionable improvement suggestion."""
        cat = record["category"]
        pred = record["predicted"]
        truth = record["truth"]

        suggestions = []

        # Check for consistent confusion pattern
        confusions = self._get_confusion_pairs()
        pair = (pred, truth)
        if pair in confusions and confusions[pair] >= 2:
            suggestions.append(
                f"Your agent consistently confuses '{pred}' with '{truth}' "
                f"({confusions[pair]} times). Add explicit rules or examples for '{cat}'."
            )

        # Check memory usage
        if not record["memory_used"]:
            suggestions.append(
                "Memory was NOT used for this step. Ensure memory context "
                "is being injected into your agent's prompt/input."
            )

        # Category-specific
        same_cat = [s for s in self.steps if s["category"] == cat]
        correct_in_cat = sum(1 for s in same_cat if s["correct"])
        total_in_cat = len(same_cat)
        if total_in_cat > 0:
            acc = correct_in_cat / total_in_cat
            if acc < 0.3:
                suggestions.append(
                    f"Category '{cat}' accuracy is only {acc:.0%}. "
                    f"Consider adding more training examples for this category."
                )

        if not suggestions:
            suggestions.append(
                f"Review the case for '{cat}' and ensure your agent has "
                f"seen enough examples of this pattern."
            )

        return " ".join(suggestions)

    def _find_similar_mistakes(self, record: Dict) -> List[Dict[str, str]]:
        """Find past mistakes similar to this one."""
        cat = record["category"]
        pred = record["predicted"]
        similar = []
        for s in self.steps[:-1]:
            if not s["correct"] and (s["category"] == cat or s["predicted"] == pred):
                similar.append({
                    "step": s["step"],
                    "category": s["category"],
                    "predicted": s["predicted"],
                    "truth": s["truth"],
                })
        return similar[-3:]  # last 3

    def _get_confusion_pairs(self) -> Dict:
        """Count prediction→truth confusion pairs."""
        pairs = defaultdict(int)
        for s in self.steps:
            if not s["correct"]:
                pairs[(s["predicted"], s["truth"])] += 1
        return pairs

    # ------------------------------------------------------------------
    # Full Explanation Report
    # ------------------------------------------------------------------

    def explain(self) -> "ExplanationReport":
        """Generate a comprehensive explanation report."""
        return ExplanationReport(self.steps)


class ExplanationReport:
    """Comprehensive explanation of agent behavior."""

    def __init__(self, steps: List[Dict[str, Any]]):
        self.steps = steps

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def accuracy(self) -> float:
        if not self.steps:
            return 0
        return sum(1 for s in self.steps if s["correct"]) / len(self.steps)

    def mistake_patterns(self) -> List[Dict[str, Any]]:
        """Identify systematic mistake patterns."""
        patterns = []

        # 1. Confusion pairs
        pairs = defaultdict(int)
        for s in self.steps:
            if not s["correct"]:
                pairs[(s["predicted"], s["truth"])] += 1

        for (pred, truth), count in sorted(pairs.items(), key=lambda x: -x[1]):
            if count >= 2:
                patterns.append({
                    "type": "confusion",
                    "description": f"Consistently predicts '{pred}' when answer is '{truth}'",
                    "count": count,
                    "severity": "HIGH" if count >= 3 else "MEDIUM",
                    "fix": f"Add explicit handling for cases where '{truth}' is correct",
                })

        # 2. Category weaknesses
        cat_stats = defaultdict(lambda: {"correct": 0, "wrong": 0})
        for s in self.steps:
            cat_stats[s["category"]]["correct" if s["correct"] else "wrong"] += 1

        for cat, stats in cat_stats.items():
            total = stats["correct"] + stats["wrong"]
            if total >= 2 and stats["wrong"] / total >= 0.6:
                patterns.append({
                    "type": "weak_category",
                    "description": f"Weak on '{cat}': {stats['correct']}/{total} correct",
                    "accuracy": stats["correct"] / total,
                    "severity": "HIGH" if stats["correct"] / total < 0.3 else "MEDIUM",
                    "fix": f"Focus training data on '{cat}' examples",
                })

        # 3. Overconfidence
        overconfident = [
            s for s in self.steps
            if not s["correct"] and s.get("confidence", 0) > 0.7
        ]
        if len(overconfident) >= 2:
            patterns.append({
                "type": "overconfidence",
                "description": f"Agent was confident (>70%) but wrong {len(overconfident)} times",
                "count": len(overconfident),
                "severity": "HIGH",
                "fix": "Calibrate confidence scores — add uncertainty for ambiguous cases",
            })

        # 4. Memory not helping
        mem_used_wrong = [s for s in self.steps if s.get("memory_used") and not s["correct"]]
        if len(mem_used_wrong) >= 2:
            patterns.append({
                "type": "memory_not_helping",
                "description": f"Memory was used but agent still wrong {len(mem_used_wrong)} times",
                "count": len(mem_used_wrong),
                "severity": "MEDIUM",
                "fix": "Memory may contain incorrect patterns — consider clearing old failure entries",
            })

        return sorted(patterns, key=lambda p: 0 if p["severity"] == "HIGH" else 1)

    def improvement_plan(self) -> List[str]:
        """Generate an ordered list of improvement actions."""
        patterns = self.mistake_patterns()
        plan = []

        if not patterns:
            plan.append("No systematic issues detected. Continue current approach.")
            return plan

        for i, p in enumerate(patterns[:5], 1):
            plan.append(f"{i}. [{p['severity']}] {p['description']} → {p['fix']}")

        return plan

    def step_by_step_log(self) -> List[Dict[str, str]]:
        """Generate a readable step-by-step audit log."""
        log = []
        streak = 0
        for s in self.steps:
            if s["correct"]:
                streak = max(streak + 1, 1)
            else:
                streak = min(streak - 1, -1)

            entry = {
                "step": s["step"],
                "status": "OK" if s["correct"] else "FAIL",
                "category": s["category"],
                "predicted": s["predicted"],
                "truth": s["truth"],
                "streak": streak,
            }

            if not s["correct"]:
                entry["explanation"] = f"Predicted '{s['predicted']}' but truth was '{s['truth']}'"
                if abs(streak) >= 3:
                    entry["alert"] = f"Failure streak of {abs(streak)}!"

            log.append(entry)
        return log

    def print_report(self):
        """Print a formatted XAI report."""
        print(f"\n{'=' * 65}")
        print("  Explainable AI Report")
        print(f"  {self.total_steps} steps | accuracy: {self.accuracy:.0%}")
        print(f"{'=' * 65}")

        # Patterns
        patterns = self.mistake_patterns()
        if patterns:
            print(f"\n  Detected Patterns ({len(patterns)}):")
            for p in patterns:
                icon = "!!!" if p["severity"] == "HIGH" else " ! "
                print(f"  [{icon}] {p['description']}")

        # Improvement plan
        plan = self.improvement_plan()
        print("\n  Improvement Plan:")
        for item in plan:
            print(f"    {item}")

        # Worst steps
        worst = [s for s in self.steps if not s["correct"]]
        if worst:
            print(f"\n  Failed Steps ({len(worst)}/{self.total_steps}):")
            for s in worst[:5]:
                print(
                    f"    Step {s['step']}: {s['category']} — "
                    f"predicted '{s['predicted']}' (truth: '{s['truth']}')"
                )

        print(f"{'=' * 65}\n")
