"""Grader for Task 1 — Easy: Binary Safety Classification.

Strict grading: correct = 1.0, wrong = 0.0.
No partial credit for NEEDS_REVIEW (binary task).
"""

from env.models import SafetyLabel


def grade(predicted: SafetyLabel, ground_truth: SafetyLabel) -> float:
    """Grade a binary safety classification.

    Args:
        predicted: The agent's classification.
        ground_truth: The correct label.

    Returns:
        1.0 for correct, 0.0 for wrong.
    """
    if predicted == ground_truth:
        return 1.0
    return 0.0
