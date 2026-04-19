"""
Example 6: Compare ALL agent types side-by-side on the same environment.

Runs LLM (mock), RL, Classifier, Rule-based, and Random agents through
the same environment and prints a comparison table.
"""

import sys, io, random
if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

import cognicore
from cognicore.agents.base_agent import RandomAgent


# === Agent Definitions ===

class MockLLMAgent:
    """Simulates an LLM that uses memory context."""
    name = "LLM (Gemini mock)"

    def act(self, obs):
        prompt = obs.get("prompt", "").lower()
        memory = obs.get("memory_context", [])

        # Use memory: if we failed on this category before, flip answer
        for m in memory:
            if not m.get("correct") and m.get("predicted") == "SAFE":
                return {"classification": "UNSAFE"}
            if not m.get("correct") and m.get("predicted") == "UNSAFE":
                return {"classification": "SAFE"}

        # Keyword matching
        if any(w in prompt for w in ["malware", "hack", "phish", "weapon", "bomb", "exploit"]):
            return {"classification": "UNSAFE"}
        if any(w in prompt for w in ["password", "credential", "social security"]):
            return {"classification": "UNSAFE"}
        return {"classification": "SAFE"}


class QLearnerAgent:
    """Simple Q-learner that improves across episodes."""
    name = "RL (Q-Learning)"

    def __init__(self):
        self.q = {}
        self.lr = 0.3
        self.epsilon = 0.2
        self.actions = ["SAFE", "UNSAFE", "NEEDS_REVIEW"]
        self.last_cat = None
        self.last_act = None

    def act(self, obs):
        cat = obs.get("category", "?")
        self.last_cat = cat
        if cat not in self.q:
            self.q[cat] = {a: 0 for a in self.actions}
        if random.random() < self.epsilon:
            self.last_act = random.choice(self.actions)
        else:
            self.last_act = max(self.q[cat], key=self.q[cat].get)
        return {"classification": self.last_act}

    def learn(self, reward):
        if self.last_cat and self.last_act:
            old = self.q[self.last_cat][self.last_act]
            self.q[self.last_cat][self.last_act] = old + self.lr * (reward - old)


class KeywordClassifierAgent:
    """Simple keyword safety classifier."""
    name = "Classifier (Keywords)"

    UNSAFE = ["malware", "hack", "exploit", "phish", "weapon", "bomb", "attack",
              "password", "credential", "drug", "kill", "violence"]

    def act(self, obs):
        text = obs.get("prompt", "").lower()
        hits = sum(1 for kw in self.UNSAFE if kw in text)
        if hits >= 2:
            return {"classification": "UNSAFE"}
        if hits == 1:
            return {"classification": "NEEDS_REVIEW"}
        return {"classification": "SAFE"}


class RuleAgent:
    """Rule-based with reflection hint following."""
    name = "Rule-Based"

    def act(self, obs):
        prompt = obs.get("prompt", "").lower()
        hints = str(obs.get("reflection_hints", "")).lower()

        if "unsafe" in hints:
            return {"classification": "UNSAFE"}
        if any(w in prompt for w in ["malware", "hack", "exploit", "bomb", "weapon"]):
            return {"classification": "UNSAFE"}
        if any(w in prompt for w in ["phish", "password", "credential"]):
            return {"classification": "UNSAFE"}
        return {"classification": "SAFE"}


class WrappedRandomAgent:
    name = "Random (Baseline)"

    def __init__(self):
        self.actions = ["SAFE", "UNSAFE", "NEEDS_REVIEW"]

    def act(self, obs):
        return {"classification": random.choice(self.actions)}


# === Main Comparison ===

def run_comparison():
    """Compare all agent types on the same environment."""

    agents = [
        MockLLMAgent(),
        QLearnerAgent(),
        KeywordClassifierAgent(),
        RuleAgent(),
        WrappedRandomAgent(),
    ]

    env_id = "SafetyClassification-v1"
    difficulty = "easy"
    episodes = 3

    print("\n" + "=" * 75)
    print("  CogniCore Agent Comparison")
    print(f"  Environment: {env_id} ({difficulty}) x {episodes} episodes")
    print("=" * 75)

    results = []

    for agent in agents:
        env = cognicore.make(env_id, difficulty=difficulty)
        best_score = 0
        best_acc = 0
        total_memory_bonus = 0
        total_streak_penalty = 0

        for ep in range(episodes):
            obs = env.reset()
            ep_mem_bonus = 0
            ep_streak = 0

            while True:
                action = agent.act(obs)
                obs, reward, done, _, info = env.step(action)

                ep_mem_bonus += reward.memory_bonus
                ep_streak += reward.streak_penalty

                # RL agent learns from rewards
                if hasattr(agent, 'learn'):
                    agent.learn(reward.total)

                if done:
                    break

            stats = env.episode_stats()
            score = env.get_score()
            best_score = max(best_score, score)
            best_acc = max(best_acc, stats.accuracy)
            total_memory_bonus += ep_mem_bonus
            total_streak_penalty += ep_streak

        results.append({
            "name": agent.name,
            "best_score": best_score,
            "best_acc": best_acc,
            "memory_bonus": total_memory_bonus,
            "streak_penalty": total_streak_penalty,
        })

    # Print comparison table
    print(f"\n  {'Agent':<25s} {'Best Score':<12s} {'Best Acc':<10s} {'Memory+':>10s} {'Streak-':>10s}")
    print("  " + "-" * 70)

    results.sort(key=lambda x: -x["best_score"])
    for i, r in enumerate(results):
        rank = f"#{i+1}"
        print(
            f"  {rank:3s} {r['name']:<22s} "
            f"{r['best_score']:<12.4f} "
            f"{r['best_acc']*100:<9.0f}% "
            f"{r['memory_bonus']:>+10.2f} "
            f"{r['streak_penalty']:>+10.2f}"
        )

    print(f"\n  Winner: {results[0]['name']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_comparison()
