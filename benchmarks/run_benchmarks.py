"""
CogniCore Benchmark Suite — Reproducible, deterministic benchmarks.

Runs multiple seeds, reports mean +/- std dev, and saves results to JSON.
Usage:
    python benchmarks/run_benchmarks.py
    cognicore benchmark --episodes 5
"""

import sys
import os
import json
import time
import statistics
from pathlib import Path

# Force local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cognicore as cc
from cognicore.smart_agents import AutoLearner

# ── Configuration ──────────────────────────────────────────────────────
NUM_SEEDS = 5
EPISODES_PER_SEED = 10
ENVS_TO_TEST = ["SafetyClassification-v1", "MathReasoning-v1", "CodeDebugging-v1"]

# ── Helpers ────────────────────────────────────────────────────────────

def run_single_seed(agent_class, env_id, seed, episodes, enable_memory):
    """Run one complete experiment with a fixed seed."""
    import random
    random.seed(seed)

    config = cc.CogniCoreConfig(
        enable_memory=enable_memory,
        enable_reflection=enable_memory,
    )
    env = cc.make(env_id, config=config)
    agent = agent_class()

    scores = []
    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if hasattr(agent, "learn"):
                agent.learn(reward, info)
            if done:
                scores.append(env.get_score())
                break
    return statistics.mean(scores)


def benchmark_condition(agent_class, env_id, seeds, episodes, enable_memory):
    """Run across multiple seeds, return mean and std."""
    seed_scores = []
    for seed in seeds:
        score = run_single_seed(agent_class, env_id, seed, episodes, enable_memory)
        seed_scores.append(score)
    return {
        "mean": round(statistics.mean(seed_scores), 4),
        "std": round(statistics.stdev(seed_scores), 4) if len(seed_scores) > 1 else 0.0,
        "min": round(min(seed_scores), 4),
        "max": round(max(seed_scores), 4),
        "seeds": len(seed_scores),
        "episodes_per_seed": episodes,
        "raw": seed_scores,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    seeds = list(range(42, 42 + NUM_SEEDS))
    results = {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 65)
    print("  COGNICORE BENCHMARK SUITE")
    print(f"  {NUM_SEEDS} seeds x {EPISODES_PER_SEED} episodes | {len(ENVS_TO_TEST)} environments")
    print("=" * 65)

    for env_id in ENVS_TO_TEST:
        print(f"\n--- {env_id} ---")

        # Baseline (no memory)
        baseline = benchmark_condition(
            AutoLearner, env_id, seeds, EPISODES_PER_SEED, enable_memory=False
        )
        print(f"  Baseline (no memory): {baseline['mean']*100:.1f}% +/- {baseline['std']*100:.1f}%")

        # CogniCore (with memory)
        cognicore_result = benchmark_condition(
            AutoLearner, env_id, seeds, EPISODES_PER_SEED, enable_memory=True
        )
        print(f"  CogniCore (memory):   {cognicore_result['mean']*100:.1f}% +/- {cognicore_result['std']*100:.1f}%")

        improvement = (cognicore_result["mean"] - baseline["mean"]) * 100
        print(f"  Improvement:          +{improvement:.1f}%")

        results[env_id] = {
            "baseline": baseline,
            "cognicore": cognicore_result,
            "improvement_pct": round(improvement, 2),
        }

    # ── Summary Table ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'Environment':<30} {'Baseline':>10} {'CogniCore':>12} {'Gain':>8}")
    print("-" * 65)
    for env_id, r in results.items():
        b = f"{r['baseline']['mean']*100:.1f}%"
        c = f"{r['cognicore']['mean']*100:.1f}%"
        g = f"+{r['improvement_pct']:.1f}%"
        print(f"  {env_id:<30} {b:>10} {c:>12} {g:>8}")
    print("=" * 65)

    # ── Save Report ────────────────────────────────────────────────────
    report = {
        "timestamp": timestamp,
        "config": {
            "num_seeds": NUM_SEEDS,
            "episodes_per_seed": EPISODES_PER_SEED,
            "seeds": seeds,
        },
        "results": results,
    }
    out_path = Path(__file__).parent / "benchmark_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
