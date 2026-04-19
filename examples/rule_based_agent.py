"""
Example 4: Rule-Based Agent — Handcrafted rules evaluated by CogniCore.

Shows how a deterministic rule system benefits from CogniCore's memory
to know which rules are working and which are failing.
"""

import sys, io
if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

import cognicore


class RuleBasedSafetyAgent:
    """Handcrafted rule system for safety classification.

    Rules are simple but demonstrate how even deterministic systems
    benefit from CogniCore's structured feedback.
    """

    def __init__(self):
        self.rules_triggered = {}

    def evaluate(self, obs):
        prompt = obs.get("prompt", "").lower()
        category = obs.get("category", "")

        # Rule 1: Malware keywords
        if any(w in prompt for w in ["malware", "virus", "trojan", "ransomware"]):
            return "UNSAFE", "rule_malware"

        # Rule 2: Hacking keywords
        if any(w in prompt for w in ["hack", "exploit", "breach", "backdoor"]):
            return "UNSAFE", "rule_hacking"

        # Rule 3: Social engineering
        if any(w in prompt for w in ["phishing", "impersonate", "pretend to be"]):
            return "UNSAFE", "rule_social_eng"

        # Rule 4: Medical misinformation
        if any(w in prompt for w in ["diagnosis", "prescribe", "medical advice"]):
            return "NEEDS_REVIEW", "rule_medical"

        # Rule 5: Violence
        if any(w in prompt for w in ["weapon", "bomb", "kill", "attack", "harm"]):
            return "UNSAFE", "rule_violence"

        # Rule 6: Privacy / PII
        if any(w in prompt for w in ["social security", "credit card", "password"]):
            return "UNSAFE", "rule_privacy"

        # Rule 7: Check reflection hints from CogniCore
        hints = obs.get("reflection_hints", "")
        if hints and "unsafe" in str(hints).lower():
            return "UNSAFE", "rule_reflection_hint"

        # Default: assume safe
        return "SAFE", "rule_default"


def run_rule_agent():
    """Run rule-based agent and see which rules work."""

    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    agent = RuleBasedSafetyAgent()

    print("=" * 65)
    print("  Agent Type: Rule-Based System")
    print("  Environment: SafetyClassification-v1 (easy)")
    print("=" * 65)

    obs = env.reset()
    rule_stats = {}

    while True:
        prediction, rule_name = agent.evaluate(obs)

        obs, reward, done, _, info = env.step({"classification": prediction})
        correct = info["eval_result"]["correct"]

        if rule_name not in rule_stats:
            rule_stats[rule_name] = {"correct": 0, "wrong": 0, "total_reward": 0}
        rule_stats[rule_name]["correct" if correct else "wrong"] += 1
        rule_stats[rule_name]["total_reward"] += reward.total

        icon = "[OK]" if correct else "[XX]"
        print(f"  {icon} {rule_name:25s} -> {prediction:12s} reward={reward.total:+.2f}")

        if done:
            break

    stats = env.episode_stats()
    print(f"\n  Score: {env.get_score():.4f} | Accuracy: {stats.accuracy:.0%}")
    print(f"\n  Rule effectiveness:")
    for rule, s in sorted(rule_stats.items(), key=lambda x: -x[1]["correct"]):
        total = s["correct"] + s["wrong"]
        acc = s["correct"] / total if total else 0
        print(f"    {rule:25s} accuracy={acc:.0%} ({s['correct']}/{total}) avg_reward={s['total_reward']/total:+.2f}")
    print()
    return env.get_score()


if __name__ == "__main__":
    run_rule_agent()
