"""
Example 3: Classifier Agent — sklearn-style classifier using CogniCore.

No sklearn needed. Shows how a keyword-based classifier gets evaluated
with structured rewards, memory, and reflection.
"""

import sys, io
if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

import cognicore


class KeywordClassifier:
    """Simple keyword-based classifier (stand-in for sklearn/PyTorch).

    In real usage, replace predict() with:
        prediction = model.predict(vectorizer.transform([text]))
    """

    UNSAFE_KEYWORDS = [
        "malware", "hack", "exploit", "attack", "phishing",
        "password", "credential", "inject", "vulnerability",
        "bomb", "weapon", "kill", "drug", "illegal",
    ]

    def predict(self, text):
        text_lower = text.lower()
        score = sum(1 for kw in self.UNSAFE_KEYWORDS if kw in text_lower)
        if score >= 2:
            return "UNSAFE"
        elif score == 1:
            return "NEEDS_REVIEW"
        return "SAFE"

    def predict_with_confidence(self, text):
        text_lower = text.lower()
        score = sum(1 for kw in self.UNSAFE_KEYWORDS if kw in text_lower)
        if score >= 2:
            return "UNSAFE", min(0.5 + score * 0.15, 1.0)
        elif score == 1:
            return "NEEDS_REVIEW", 0.5
        return "SAFE", 0.7


def run_classifier_agent():
    """Evaluate a keyword classifier with CogniCore's structured feedback."""

    env = cognicore.make("SafetyClassification-v1", difficulty="medium")
    clf = KeywordClassifier()

    print("=" * 65)
    print("  Agent Type: Classifier (Keyword-based)")
    print("  Environment: SafetyClassification-v1 (medium)")
    print("=" * 65)

    obs = env.reset()
    results = {"correct": 0, "wrong": 0, "categories": {}}

    while True:
        text = obs.get("prompt", "")
        prediction, confidence = clf.predict_with_confidence(text)

        obs, reward, done, _, info = env.step({
            "classification": prediction,
            "confidence": confidence,  # CogniCore uses this for calibration reward
        })

        correct = info["eval_result"]["correct"]
        category = info["eval_result"].get("category", "?")
        icon = "[OK]" if correct else "[XX]"

        if correct:
            results["correct"] += 1
        else:
            results["wrong"] += 1

        if category not in results["categories"]:
            results["categories"][category] = {"correct": 0, "wrong": 0}
        results["categories"][category]["correct" if correct else "wrong"] += 1

        print(
            f"  {icon} {category:25s} pred={prediction:12s} "
            f"conf={confidence:.1f} reward={reward.total:+.2f} "
            f"conf_cal={reward.confidence_cal:+.3f}"
        )

        if done:
            break

    stats = env.episode_stats()
    print(f"\n  Score: {env.get_score():.4f} | Accuracy: {stats.accuracy:.0%}")
    print(f"\n  Per-category breakdown:")
    for cat, counts in sorted(results["categories"].items()):
        total = counts["correct"] + counts["wrong"]
        acc = counts["correct"] / total if total else 0
        bar = "=" * int(acc * 20) + "-" * (20 - int(acc * 20))
        print(f"    {cat:25s} [{bar}] {acc:.0%} ({counts['correct']}/{total})")
    print()
    return env.get_score()


if __name__ == "__main__":
    run_classifier_agent()
