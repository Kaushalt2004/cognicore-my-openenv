"""
Benchmark: Real ML models vs RL agents — ZERO LLMs.

Tests:
  - RandomForest (scikit-learn)
  - Deep Q-Network (PyTorch)
  - Policy Gradient / REINFORCE (PyTorch)
  - Q-Learning (tabular)
  - SARSA (tabular)
  - Genetic Algorithm (evolutionary)
  - Random (baseline)

Run: python examples/benchmark_ml_agents.py
"""
import cognicore as cc
import random
import time
import sys

random.seed(42)

print()
print("=" * 72)
print("  CogniCore v0.6.0 -- ML Agent Benchmark (ZERO LLMs)")
print("  All models train LOCALLY on your machine.")
print("=" * 72)


def benchmark(agent, env_id, episodes, label):
    env = cc.make(env_id)
    rewards = []
    t0 = time.time()

    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if hasattr(agent, "on_reward"):
                agent.on_reward(reward)
            ep_reward += reward.total
            if done or truncated:
                break
        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end(env.episode_stats())
        rewards.append(ep_reward)

    elapsed = time.time() - t0
    first = sum(rewards[:episodes//2]) / max(1, episodes//2)
    last = sum(rewards[episodes//2:]) / max(1, episodes//2)
    avg = sum(rewards) / len(rewards)
    return {"label": label, "avg": avg, "first": first, "last": last,
            "learn": last - first, "time": elapsed}


def print_table(results, title):
    print()
    print("-" * 72)
    print(f"  {title}")
    print("-" * 72)
    print(f"  {'Agent':<24} {'Avg':>8} {'Start':>8} {'End':>8} {'Learn':>8} {'Time':>6}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for r in sorted(results, key=lambda x: -x["avg"]):
        bar = "#" * min(20, max(0, int(r["avg"])))
        print(f"  {r['label']:<24} {r['avg']:>+7.1f}  {r['first']:>+7.1f}  "
              f"{r['last']:>+7.1f}  {r['learn']:>+7.1f}  {r['time']:>5.1f}s  {bar}")

    winner = max(results, key=lambda x: x["avg"])
    learner = max(results, key=lambda x: x["learn"])
    print(f"\n  Best overall: {winner['label']} ({winner['avg']:+.1f})")
    print(f"  Most learning: {learner['label']} ({learner['learn']:+.1f} improvement)")


EP = 80

# ═══════════════════════════════════════════════════════════════
#  TEST 1: GridWorld — Navigation
# ═══════════════════════════════════════════════════════════════

print("\n  Building agents for GridWorld-v1...")
grid_agents = {
    "Random":           cc.RandomAgent(),
    "Q-Learning":       cc.QLearningAgent(["UP","DOWN","LEFT","RIGHT"],
                                           learning_rate=0.2, epsilon_decay=0.97),
    "SARSA":            cc.SARSAAgent(["UP","DOWN","LEFT","RIGHT"],
                                       epsilon_decay=0.97),
    "Bandit (UCB1)":    cc.BanditAgent(["UP","DOWN","LEFT","RIGHT"]),
    "Genetic":          cc.GeneticAgent(["UP","DOWN","LEFT","RIGHT"],
                                         population_size=10, strategy_length=50),
}

# Add PyTorch agents if available
try:
    import torch
    grid_agents["Deep Q-Network"] = cc.DeepQAgent(
        state_size=8, actions=["UP","DOWN","LEFT","RIGHT"],
        hidden_size=32, learning_rate=0.001, epsilon_decay=0.99,
    )
    grid_agents["Policy Gradient"] = cc.PolicyGradientAgent(
        state_size=8, actions=["UP","DOWN","LEFT","RIGHT"],
        hidden_size=32, learning_rate=0.01,
    )
    print("  [+] PyTorch agents loaded (DQN, REINFORCE)")
except ImportError:
    print("  [-] PyTorch not installed (pip install torch)")

# Add sklearn agent if available
try:
    import sklearn
    grid_agents["RandomForest"] = cc.SklearnAgent(
        actions=["UP","DOWN","LEFT","RIGHT"],
        model_type="random_forest", retrain_every=15,
    )
    print("  [+] scikit-learn agent loaded (RandomForest)")
except ImportError:
    print("  [-] scikit-learn not installed (pip install scikit-learn)")

grid_results = []
for name, agent in grid_agents.items():
    print(f"  Training {name}...", end=" ", flush=True)
    r = benchmark(agent, "GridWorld-v1", EP, name)
    print(f"done ({r['avg']:+.1f})")
    grid_results.append(r)

print_table(grid_results, "GridWorld-v1 (5x5 Navigation, 3 traps)")


# ═══════════════════════════════════════════════════════════════
#  TEST 2: ResourceGathering — Multi-Objective
# ═══════════════════════════════════════════════════════════════

print("\n  Building agents for ResourceGathering-v1...")
res_actions = ["GATHER_FOOD","GATHER_WOOD","GATHER_STONE","BUILD","REST","EXPLORE"]

res_agents = {
    "Random":        cc.RandomAgent(),
    "Q-Learning":    cc.QLearningAgent(res_actions, learning_rate=0.15, epsilon_decay=0.97),
    "Genetic":       cc.GeneticAgent(res_actions, population_size=10,
                                      strategy_length=30),
    "Bandit (UCB1)": cc.BanditAgent(res_actions),
}

try:
    import sklearn
    res_agents["RandomForest"] = cc.SklearnAgent(
        actions=res_actions, model_type="random_forest", retrain_every=20,
    )
except ImportError:
    pass

try:
    import torch
    res_agents["Policy Gradient"] = cc.PolicyGradientAgent(
        state_size=10, actions=res_actions,
        hidden_size=32, learning_rate=0.01,
    )
except ImportError:
    pass

res_results = []
for name, agent in res_agents.items():
    print(f"  Training {name}...", end=" ", flush=True)
    r = benchmark(agent, "ResourceGathering-v1", EP, name)
    print(f"done ({r['avg']:+.1f})")
    res_results.append(r)

print_table(res_results, "ResourceGathering-v1 (Multi-Objective)")


# ═══════════════════════════════════════════════════════════════
#  TEST 3: Safety — Classification (no LLMs!)
# ═══════════════════════════════════════════════════════════════

print("\n  Building agents for RealWorldSafety-v1...")
safety_actions = ["SAFE", "UNSAFE", "NEEDS_REVIEW"]

safety_agents = {
    "Random":        cc.RandomAgent(),
    "Bandit (UCB1)": cc.BanditAgent(safety_actions),
}

try:
    import sklearn
    safety_agents["RandomForest"] = cc.SklearnAgent(
        actions=safety_actions, model_type="random_forest", retrain_every=10,
    )
    safety_agents["SVM"] = cc.SklearnAgent(
        actions=safety_actions, model_type="svm", retrain_every=10,
    )
    safety_agents["Decision Tree"] = cc.SklearnAgent(
        actions=safety_actions, model_type="decision_tree", retrain_every=10,
    )
    safety_agents["MLP (Neural Net)"] = cc.SklearnAgent(
        actions=safety_actions, model_type="mlp", retrain_every=10,
    )
except ImportError:
    pass

safety_results = []
for name, agent in safety_agents.items():
    print(f"  Training {name}...", end=" ", flush=True)
    r = benchmark(agent, "RealWorldSafety-v1", EP, name)
    print(f"done ({r['avg']:+.1f})")
    safety_results.append(r)

print_table(safety_results, "RealWorldSafety-v1 (AI Safety Classification)")


# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 72)
print("  AGENTS TESTED (zero LLMs, zero API keys):")
print("=" * 72)
print("  Tabular RL:    Q-Learning, SARSA, UCB1 Bandit")
print("  Evolutionary:  Genetic Algorithm")
try:
    import torch
    print("  Deep RL:       Deep Q-Network, REINFORCE Policy Gradient (PyTorch)")
except ImportError:
    print("  Deep RL:       [not installed] pip install torch")
try:
    import sklearn
    print("  Classical ML:  RandomForest, SVM, DecisionTree, MLP (scikit-learn)")
except ImportError:
    print("  Classical ML:  [not installed] pip install scikit-learn")
try:
    import xgboost
    print("  Boosting:      XGBoost gradient boosted trees")
except ImportError:
    print("  Boosting:      [not installed] pip install xgboost")
print()
print(f"  CogniCore v{cc.__version__} | {len(cc.list_envs())} envs | pip install cognicore-env")
print("=" * 72)
