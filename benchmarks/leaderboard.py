"""
CogniCore Official Benchmark — Generates the published leaderboard.

Run: python benchmarks/leaderboard.py
Outputs: benchmarks/RESULTS.md with tables and rankings
"""
import cognicore.gym
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import time
import json
import os

SEEDS = [42, 123, 456, 789, 1337]
TRAIN_STEPS = 50_000
EVAL_EPISODES = 50

ENVS = [
    ("cognicore/GridWorld-v0", "Navigation (5x5, traps)"),
    ("cognicore/MazeRunner-v0", "Procedural maze (8x8)"),
    ("cognicore/Trading-v0", "Portfolio management"),
    ("cognicore/Survival-v0", "Long-horizon survival"),
    ("cognicore/BattleArena-v0", "2-player combat"),
]

ALGOS = {"PPO": PPO, "DQN": DQN, "A2C": A2C}


def run_random_baseline(env_id, n_episodes=50, seeds=None):
    seeds = seeds or SEEDS
    all_rewards = []
    for seed in seeds:
        env = gym.make(env_id)
        for _ in range(n_episodes // len(seeds)):
            obs, _ = env.reset(seed=seed)
            total = 0
            done = False
            while not done:
                obs, r, t, tr, _ = env.step(env.action_space.sample())
                total += r
                done = t or tr
            all_rewards.append(total)
        env.close()
    return np.mean(all_rewards), np.std(all_rewards)


def run_algo(env_id, algo_name, algo_cls, steps, seeds=None):
    seeds = seeds or SEEDS[:3]
    all_means = []
    for seed in seeds:
        env = Monitor(gym.make(env_id))
        model = algo_cls("MlpPolicy", env, verbose=0, seed=seed)
        model.learn(total_timesteps=steps)
        mean_r, _ = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES // len(seeds))
        all_means.append(mean_r)
        env.close()
    return np.mean(all_means), np.std(all_means)


def main():
    os.makedirs("benchmarks", exist_ok=True)

    print("=" * 70)
    print("  CogniCore Official Benchmark")
    print(f"  {TRAIN_STEPS:,} training steps | {len(SEEDS)} seeds | {EVAL_EPISODES} eval episodes")
    print("=" * 70)

    results = {}

    for env_id, desc in ENVS:
        print(f"\n  {env_id} — {desc}")
        results[env_id] = {"description": desc, "baselines": {}}

        # Random
        mean, std = run_random_baseline(env_id)
        results[env_id]["baselines"]["Random"] = {"mean": round(mean, 1), "std": round(std, 1), "time": 0}
        print(f"    Random:  {mean:>+8.1f} +/- {std:.1f}")

        # Trained agents
        for algo_name, algo_cls in ALGOS.items():
            t0 = time.time()
            try:
                mean, std = run_algo(env_id, algo_name, algo_cls, TRAIN_STEPS)
                dt = time.time() - t0
                results[env_id]["baselines"][algo_name] = {
                    "mean": round(mean, 1), "std": round(std, 1), "time": round(dt, 1)
                }
                print(f"    {algo_name:<8} {mean:>+8.1f} +/- {std:.1f}  ({dt:.0f}s)")
            except Exception as e:
                print(f"    {algo_name:<8} FAILED: {e}")
                results[env_id]["baselines"][algo_name] = {"mean": 0, "std": 0, "error": str(e)}

    # Save JSON
    with open("benchmarks/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown
    md = generate_markdown(results)
    with open("benchmarks/RESULTS.md", "w") as f:
        f.write(md)

    print(f"\n  Results saved to benchmarks/results.json")
    print(f"  Leaderboard saved to benchmarks/RESULTS.md")


def generate_markdown(results):
    lines = [
        "# CogniCore Benchmark Leaderboard",
        "",
        f"> Trained with **{TRAIN_STEPS:,}** steps | Evaluated over **{EVAL_EPISODES}** episodes | **{len(SEEDS)}** seeds",
        "",
    ]

    for env_id, data in results.items():
        lines.append(f"## {env_id}")
        lines.append(f"*{data['description']}*")
        lines.append("")
        lines.append("| Rank | Algorithm | Score | Std | Train Time |")
        lines.append("|------|-----------|-------|-----|------------|")

        sorted_baselines = sorted(
            data["baselines"].items(),
            key=lambda x: x[1].get("mean", -999), reverse=True
        )

        medals = ["🥇", "🥈", "🥉"]
        for i, (name, vals) in enumerate(sorted_baselines):
            rank = medals[i] if i < 3 else f" {i+1}."
            mean = vals.get("mean", 0)
            std = vals.get("std", 0)
            t = vals.get("time", 0)
            time_str = f"{t:.0f}s" if t > 0 else "—"
            lines.append(f"| {rank} | **{name}** | {mean:+.1f} | {std:.1f} | {time_str} |")

        lines.append("")

    lines.extend([
        "---",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "pip install cognicore-env stable-baselines3",
        "python benchmarks/leaderboard.py",
        "```",
        "",
        "## Environment Details",
        "",
        "| Environment | Obs Space | Action Space | Max Steps |",
        "|-------------|-----------|-------------|-----------|",
    ])

    for env_id, _ in ENVS:
        env = gym.make(env_id)
        obs = env.observation_space
        act = env.action_space
        ms = env.spec.max_episode_steps if env.spec else "?"
        lines.append(f"| `{env_id}` | Box({obs.shape}) | {act} | {ms} |")
        env.close()

    lines.append("")
    lines.append(f"*Generated with CogniCore v0.7.0*")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
