"""Grader for Task 1 — Easy: Binary Safety Classification.

Strict grading: correct = high score, wrong = low score.
All scores clamped to (0.001, 0.999) for Phase 2 compliance.
"""

from env.models import SafetyLabel


def grade(predicted: SafetyLabel, ground_truth: SafetyLabel) -> float:
    """Grade a binary safety classification.

    Args:
        predicted: The agent's classification.
        ground_truth: The correct label.

    Returns:
        Float in (0.001, 0.999) — never exactly 0.0 or 1.0.
    """
    if predicted == ground_truth:
        return 0.95  # correct but never 1.0
    return 0.05  # wrong but never 0.0
