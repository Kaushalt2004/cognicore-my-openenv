"""
Example 2: RL Agent — Uses CogniCore's GymnasiumAdapter with a simple Q-table.

No external dependencies needed. Shows how an RL agent learns from
CogniCore's structured rewards across episodes.
"""

import sys, io, random
if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

import cognicore


class SimpleQLearner:
    """Minimal Q-learning agent that learns action preferences per category."""

    def __init__(self, actions, lr=0.3, epsilon=0.3):
        self.actions = actions
        self.lr = lr
        self.epsilon = epsilon
        self.q_table = {}  # {category: {action: value}}

    def act(self, obs):
        category = obs.get("category", "unknown")
        if category not in self.q_table:
            self.q_table[category] = {a: 0.0 for a in self.actions}

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[category], key=self.q_table[category].get)

    def learn(self, category, action, reward):
        if category not in self.q_table:
            self.q_table[category] = {a: 0.0 for a in self.actions}
        old = self.q_table[category][action]
        self.q_table[category][action] = old + self.lr * (reward - old)


def run_rl_agent():
    """Train an RL agent using CogniCore's structured rewards."""

    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    actions = ["SAFE", "UNSAFE", "NEEDS_REVIEW"]
    agent = SimpleQLearner(actions, lr=0.3, epsilon=0.4)

    print("=" * 65)
    print("  Agent Type: RL (Q-Learning)")
    print("  Environment: SafetyClassification-v1 (easy)")
    print("  Training: 5 episodes with decaying epsilon")
    print("=" * 65)

    for ep in range(1, 6):
        obs = env.reset()
        agent.epsilon = max(0.05, 0.4 - ep * 0.08)  # decay exploration

        while True:
            category = obs.get("category", "unknown")
            action = agent.act(obs)
            obs, reward, done, _, info = env.step({"classification": action})

            # RL agents use the TOTAL reward (all 8 components combined)
            agent.learn(category, action, reward.total)

            if done:
                break

        stats = env.episode_stats()
        print(
            f"  Episode {ep}: score={env.get_score():.4f} "
            f"accuracy={stats.accuracy:.0%} "
            f"epsilon={agent.epsilon:.2f} "
            f"q_entries={len(agent.q_table)}"
        )

    # Show what the agent learned
    print(f"\n  Learned Q-values:")
    for cat in sorted(list(agent.q_table.keys()))[:5]:
        best = max(agent.q_table[cat], key=agent.q_table[cat].get)
        print(f"    {cat:25s} -> {best} (Q={agent.q_table[cat][best]:.2f})")
    print()
    return env.get_score()


if __name__ == "__main__":
    run_rl_agent()
