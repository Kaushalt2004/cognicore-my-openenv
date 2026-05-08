"""
DEMO: Q-Learning agent actually learning in GridWorld.

This proves CogniCore is a REAL RL framework, not just a classifier.
The agent starts knowing nothing, explores randomly, and over 200
episodes learns to navigate from (0,0) to the goal while avoiding traps.

Watch the reward curve improve over time.
"""
import cognicore as cc

print("=" * 60)
print("  CogniCore — Real RL Training Demo")
print("  Q-Learning Agent + GridWorld-v1")
print("=" * 60)

# Create a REAL RL agent (not an LLM, not a classifier)
agent = cc.QLearningAgent(
    actions=["UP", "DOWN", "LEFT", "RIGHT"],
    learning_rate=0.2,
    discount=0.95,
    epsilon=1.0,           # Start fully random
    epsilon_decay=0.99,    # Gradually exploit learned Q-values
)

env = cc.make("GridWorld-v1")

# Training loop — agent ACTUALLY LEARNS
episodes = 200
results = []

for ep in range(1, episodes + 1):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    reached_goal = False

    while True:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.on_reward(reward)
        total_reward += reward.total
        steps += 1

        if info.get("metadata", {}).get("event") == "goal":
            reached_goal = True

        if done or truncated:
            break

    agent.on_episode_end(env.episode_stats())
    results.append({
        "episode": ep,
        "reward": total_reward,
        "steps": steps,
        "goal": reached_goal,
        "epsilon": agent.epsilon,
    })

    # Print progress every 20 episodes
    if ep % 20 == 0 or ep == 1:
        last_20 = results[-20:]
        avg_reward = sum(r["reward"] for r in last_20) / len(last_20)
        goal_rate = sum(1 for r in last_20 if r["goal"]) / len(last_20) * 100
        print(f"  Episode {ep:4d} | Avg Reward: {avg_reward:+6.1f} | "
              f"Goal Rate: {goal_rate:5.1f}% | "
              f"eps: {agent.epsilon:.3f} | "
              f"Q-states: {len(agent.q_table)}")

# Final summary
print()
print("=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)

first_20 = results[:20]
last_20 = results[-20:]
print(f"  First 20 episodes: avg reward = {sum(r['reward'] for r in first_20)/20:+.1f}, "
      f"goal rate = {sum(1 for r in first_20 if r['goal'])/20*100:.0f}%")
print(f"  Last 20 episodes:  avg reward = {sum(r['reward'] for r in last_20)/20:+.1f}, "
      f"goal rate = {sum(1 for r in last_20 if r['goal'])/20*100:.0f}%")
print(f"  Q-table states learned: {len(agent.q_table)}")
print(f"  Final epsilon: {agent.epsilon:.4f}")

improvement = (sum(r['reward'] for r in last_20)/20) - (sum(r['reward'] for r in first_20)/20)
print(f"\n  Reward improvement: {improvement:+.1f} ({'agent learned!' if improvement > 0 else 'needs more training'})")

# Show final grid
print(f"\n  Final grid state:")
obs = env.reset()
print(env.render())
