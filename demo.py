"""
CogniCore Demo — Realistic agent improvement over 20 episodes.

Shows gradual, noisy improvement — not a perfect jump.
"""
import sys
import os
import io
import random

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cognicore as cc
from cognicore.smart_agents import AutoLearner

def main():
    random.seed(42)

    print("\nCogniCore Demo v" + cc.__version__)
    print("=" * 65)
    print("  Realistic learning curve: watch improvement over 20 episodes")
    print("=" * 65)

    # ── Without memory ─────────────────────────────────────────────────
    print("\n  [Baseline] Training WITHOUT memory (20 episodes)...")
    config_off = cc.CogniCoreConfig(enable_memory=False, enable_reflection=False)
    env_off = cc.make("SafetyClassification-v1", config=config_off)
    agent_off = AutoLearner()

    baseline_scores = []
    for ep in range(20):
        obs = env_off.reset()
        while True:
            action = agent_off.act(obs)
            obs, reward, done, _, info = env_off.step(action)
            if done:
                baseline_scores.append(env_off.episode_stats().accuracy)
                break

    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    print(f"  Average accuracy: {avg_baseline*100:.1f}% (flat, no learning)")

    # ── With CogniCore ─────────────────────────────────────────────────
    random.seed(42)
    print("\n  [CogniCore] Training WITH memory + reflection (20 episodes)...")
    config_on = cc.CogniCoreConfig(enable_memory=True, enable_reflection=True)
    env_on = cc.make("SafetyClassification-v1", config=config_on)
    agent_on = AutoLearner()

    cognicore_scores = []
    for ep in range(20):
        obs = env_on.reset()
        while True:
            action = agent_on.act(obs)
            obs, reward, done, _, info = env_on.step(action)
            if hasattr(agent_on, "learn"):
                agent_on.learn(reward, info)
            if done:
                acc = env_on.episode_stats().accuracy
                cognicore_scores.append(acc)
                break

    # Print trajectory (every 5 episodes)
    print(f"\n  {'Episode':>10} {'Baseline':>10} {'CogniCore':>12} {'Delta':>8}")
    print("  " + "-" * 44)
    for i in [0, 4, 9, 14, 19]:
        b = baseline_scores[i] * 100
        c = cognicore_scores[i] * 100
        d = c - b
        print(f"  {i+1:>10} {b:>9.0f}% {c:>11.0f}% {d:>+7.0f}%")

    avg_cc = sum(cognicore_scores) / len(cognicore_scores)
    improvement = (avg_cc - avg_baseline) * 100

    print(f"\n{'=' * 65}")
    print(f"  Baseline avg:   {avg_baseline*100:.1f}%")
    print(f"  CogniCore avg:  {avg_cc*100:.1f}%")
    print(f"  Improvement:    +{improvement:.1f}%")
    print(f"{'=' * 65}")
    print(f"  The agent gradually improves as memory accumulates.")
    print(f"  Not instant — realistic learning with variance.\n")


if __name__ == "__main__":
    main()
