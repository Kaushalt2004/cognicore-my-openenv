"""
baseline.py — Deterministic verification script.

Proves the grading system works by submitting perfect ground-truth
answers for all 54 cases and verifying expected scores.

Usage:
    python baseline.py

Expected output:
    Each task scores ~0.90+ with perfect answers.
    All rewards strictly in (0.01, 0.99).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from server.environment import SafetyMonitorEnvironment
from models import SafetyAction
from dataset import EASY_CASES, MEDIUM_CASES, HARD_CASES


TASKS = [
    {"name": "binary_safety_classification", "difficulty": "easy", "cases": EASY_CASES},
    {"name": "nuanced_safety_detection", "difficulty": "medium", "cases": MEDIUM_CASES},
    {"name": "adversarial_safety_monitoring", "difficulty": "hard", "cases": HARD_CASES},
]


def run_baseline() -> None:
    env = SafetyMonitorEnvironment()
    summary = {}

    for task in TASKS:
        task_name = task["name"]
        difficulty = task["difficulty"]
        cases = task["cases"]

        # Reset
        env.reset(task=task_name, difficulty=difficulty)

        rewards = []
        for case in cases:
            # Submit perfect ground-truth answer
            action = SafetyAction(
                classification=case.ground_truth.value,
                confidence=0.90,
                severity=case.severity.value if case.severity else "medium",
                reasoning=case.explanation[:100] if case.explanation else "baseline",
            )
            obs = env.step(action)
            reward_obj = env.last_reward
            step_info = env.last_step_info

            reward_val = reward_obj.value if reward_obj else obs.reward
            rewards.append(reward_val)

            # Verify reward is in valid range
            assert 0.01 <= reward_val <= 0.99, \
                f"Reward out of range for {case.id}: {reward_val}"

            # Verify correctness
            assert step_info.correct, \
                f"Ground truth answer marked incorrect for {case.id}: predicted={step_info.predicted}, truth={step_info.ground_truth}"

        avg_reward = sum(rewards) / len(rewards)
        score = env.get_score()
        state = env.state

        summary[difficulty] = {
            "task": task_name,
            "cases": len(cases),
            "correct": state.correct_count,
            "accuracy": state.accuracy,
            "avg_reward": round(avg_reward, 4),
            "score": score,
            "best_score": state.best_score,
            "min_reward": round(min(rewards), 4),
            "max_reward": round(max(rewards), 4),
        }

    # Print results
    print(json.dumps(summary, indent=2))
    print()

    all_pass = True
    for diff, data in summary.items():
        status = "PASS" if data["accuracy"] == 1.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {diff:8s}: {data['correct']}/{data['cases']} correct, "
              f"score={data['score']:.3f}, avg_reward={data['avg_reward']:.3f}")

    print()
    if all_pass:
        print("  BASELINE VERIFIED: All ground-truth answers score correctly.")
        print("  Grading is deterministic and reproducible.")
    else:
        print("  BASELINE FAILED: Check grader logic.")
        sys.exit(1)


if __name__ == "__main__":
    run_baseline()
