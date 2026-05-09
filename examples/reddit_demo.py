"""Reddit demo: Multiple agents on multiple environments."""
import cognicore as cc
import random

random.seed(42)

print()
print("=" * 65)
print("   CogniCore v0.6.0 -- Cognitive RL Training Framework")
print("   github.com/Kaushalt2004/cognicore-my-openenv")
print("=" * 65)

# ============================================================
# DEMO 1: Q-Learning on GridWorld
# ============================================================
print()
print("-" * 65)
print("  DEMO 1: Q-Learning Agent learns GridWorld (5x5)")
print("-" * 65)

agent = cc.QLearningAgent(
    actions=["UP", "DOWN", "LEFT", "RIGHT"],
    learning_rate=0.2,
    discount=0.95,
    epsilon=1.0,
    epsilon_decay=0.99,
)

env = cc.make("GridWorld-v1")
rewards_over_time = []

for ep in range(1, 301):
    obs = env.reset()
    ep_reward = 0.0
    while True:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.on_reward(reward)
        ep_reward += reward.total
        if done or truncated:
            break
    agent.on_episode_end(env.episode_stats())
    rewards_over_time.append(ep_reward)

    if ep in [1, 50, 100, 150, 200, 250, 300]:
        window = rewards_over_time[-20:] if len(rewards_over_time) >= 20 else rewards_over_time
        avg = sum(window) / len(window)
        bar = "#" * int(max(0, avg))
        print(f"  Ep {ep:4d} | Avg Reward: {avg:+6.1f} | {bar}")

first = sum(rewards_over_time[:30]) / 30
last = sum(rewards_over_time[-30:]) / 30
print(f"\n  Learning: {first:+.1f} -> {last:+.1f} ({last-first:+.1f} improvement)")
print(f"  Q-states learned: {len(agent.q_table)}")

# Show the grid
obs = env.reset()
print(f"\n  Grid ({env.grid_size}x{env.grid_size}): A=Agent, G=Goal, X=Trap")
print("  " + env.render().replace("\n", "\n  "))

# ============================================================
# DEMO 2: Genetic Algorithm on ResourceGathering
# ============================================================
print()
print("-" * 65)
print("  DEMO 2: Genetic Algorithm evolves Resource Strategy")
print("-" * 65)

ga = cc.GeneticAgent(
    actions=["GATHER_FOOD", "GATHER_WOOD", "GATHER_STONE", "BUILD", "REST", "EXPLORE"],
    population_size=10,
    mutation_rate=0.15,
    strategy_length=30,
)

env2 = cc.make("ResourceGathering-v1")
gen_rewards = []

for ep in range(1, 51):
    obs = env2.reset()
    ep_reward = 0.0
    while True:
        action = ga.act(obs)
        obs, reward, done, truncated, info = env2.step(action)
        ga.on_reward(reward)
        ep_reward += reward.total
        if done or truncated:
            break
    ga.on_episode_end(env2.episode_stats())
    gen_rewards.append(ep_reward)

    if ep % 10 == 0 or ep == 1:
        window = gen_rewards[-10:] if len(gen_rewards) >= 10 else gen_rewards
        avg = sum(window) / len(window)
        print(f"  Ep {ep:4d} | Gen {ga.stats['generation']:2d} | "
              f"Avg Reward: {avg:+6.1f} | Best: {ga.stats['best_fitness']:+.1f}")

# ============================================================
# DEMO 3: Benchmark -- all agents head-to-head
# ============================================================
print()
print("-" * 65)
print("  DEMO 3: Agent Comparison on GridWorld-v1 (50 episodes each)")
print("-" * 65)

agents_to_test = {
    "Random":     cc.RandomAgent(),
    "Q-Learning": cc.QLearningAgent(["UP","DOWN","LEFT","RIGHT"], epsilon_decay=0.98),
    "SARSA":      cc.SARSAAgent(["UP","DOWN","LEFT","RIGHT"], epsilon_decay=0.98),
    "Bandit":     cc.BanditAgent(["UP","DOWN","LEFT","RIGHT"]),
}

for name, agent in agents_to_test.items():
    env = cc.make("GridWorld-v1")
    total = 0.0
    for ep in range(50):
        obs = env.reset()
        ep_r = 0.0
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            agent.on_reward(reward)
            ep_r += reward.total
            if done or truncated:
                break
        agent.on_episode_end(env.episode_stats())
        total += ep_r
    avg = total / 50
    bar = "#" * int(max(0, avg))
    print(f"  {name:12s} | Avg Reward: {avg:+6.1f} | {bar}")

# ============================================================
# DEMO 4: Real-world safety eval
# ============================================================
print()
print("-" * 65)
print("  DEMO 4: Real-World AI Safety Evaluation")
print("-" * 65)

env3 = cc.make("RealWorldSafety-Hard-v1")
obs = env3.reset()

print(f"\n  Sample real-world test cases:")
for i in range(min(5, len(env3._tasks))):
    task = env3._tasks[i]
    prompt = task["prompt"][:55]
    label = task["label"]
    cat = task["category"]
    sev = task["severity"]
    emoji = {"UNSAFE": "X", "SAFE": "OK", "NEEDS_REVIEW": "??"}
    print(f"    [{emoji.get(label,'?'):>2}] {prompt}...")
    print(f"         Category: {cat} | Severity: {sev}")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 65)
all_envs = cc.list_envs()
print(f"  CogniCore v{cc.__version__}")
print(f"  {len(all_envs)} environments | {len(cc.__all__)} exports")
print(f"  pip install cognicore-env")
print(f"  github.com/Kaushalt2004/cognicore-my-openenv")
print("=" * 65)
