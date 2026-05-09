"""
CogniCore Agent Testing Suite — Every agent type, every environment.

This script demonstrates how to test NON-LLM agents:
  1. Q-Learning         (tabular RL)
  2. SARSA              (on-policy RL)
  3. Genetic Algorithm  (evolutionary)
  4. UCB1 Bandit        (exploration-exploitation)
  5. Rule-Based         (hand-coded heuristics)
  6. Random             (baseline)

Run:  python examples/test_all_agents.py
"""
import cognicore as cc
import random
import time

random.seed(42)


# ── 1. Custom Rule-Based Agent (no ML at all) ──────────────────────

class RuleBasedGridAgent(cc.BaseAgent):
    """Hard-coded heuristic: always move toward the goal, avoid known traps."""

    def __init__(self):
        self.known_traps = set()

    def act(self, obs):
        pos = tuple(obs["agent_pos"])
        goal = tuple(obs["goal_pos"])
        traps = {tuple(t) for t in obs.get("nearby_traps", [])}

        # Remember traps across steps
        self.known_traps.update(traps)

        # Try to move toward goal, avoiding traps
        candidates = []
        for action, (dr, dc) in [("DOWN", (1, 0)), ("RIGHT", (0, 1)),
                                   ("UP", (-1, 0)), ("LEFT", (0, -1))]:
            new_pos = (pos[0] + dr, pos[1] + dc)
            # Clamp
            grid = obs["grid_size"]
            new_pos = (max(0, min(grid-1, new_pos[0])),
                       max(0, min(grid-1, new_pos[1])))

            if new_pos in self.known_traps:
                continue  # Skip trap moves

            # Score by distance to goal (lower = better)
            dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
            candidates.append((dist, action))

        if candidates:
            candidates.sort()
            return {"action": candidates[0][1]}
        return {"action": random.choice(["UP", "DOWN", "LEFT", "RIGHT"])}


class RuleBasedResourceAgent(cc.BaseAgent):
    """Strategy: Explore first, gather balanced, build when possible."""

    def act(self, obs):
        turn = obs.get("turn", 0)
        energy = obs.get("energy", 0)
        can_build = obs.get("can_build", False)

        # Low energy? Rest
        if energy < 15:
            return {"action": "REST"}

        # Can build? Always build (highest reward)
        if can_build:
            return {"action": "BUILD"}

        # Early game: explore to find resources
        if turn < 5:
            return {"action": "EXPLORE"}

        # Mid game: gather the resource we have least of
        food = obs.get("food", 0)
        wood = obs.get("wood", 0)
        stone = obs.get("stone", 0)

        if food <= wood and food <= stone:
            return {"action": "GATHER_FOOD"}
        elif wood <= food and wood <= stone:
            return {"action": "GATHER_WOOD"}
        else:
            return {"action": "GATHER_STONE"}


class RuleBasedSafetyAgent(cc.BaseAgent):
    """Keyword-based safety classifier (no ML, just pattern matching)."""

    UNSAFE_KEYWORDS = [
        "hack", "crack", "exploit", "steal", "password",
        "drug", "weapon", "bomb", "kill", "murder",
        "methamphetamine", "cocaine", "heroin",
        "ignore all previous", "ignore the above",
        "jailbreak", "DAN", "pretend you are",
        "SSN", "social security", "credit card number",
    ]

    SAFE_KEYWORDS = [
        "terminate a thread", "kill the process", "destroy old hard",
        "fire an employee", "Python", "Java", "how to",
    ]

    def act(self, obs):
        text = (obs.get("prompt", "") + " " + obs.get("response", "")).lower()

        # Check safe overrides first
        for kw in self.SAFE_KEYWORDS:
            if kw.lower() in text:
                return {"classification": "SAFE"}

        # Check unsafe patterns
        for kw in self.UNSAFE_KEYWORDS:
            if kw.lower() in text:
                return {"classification": "UNSAFE"}

        return {"classification": "SAFE"}


# ── 2. Run benchmarks ──────────────────────────────────────────────

def benchmark_agent(agent, env_id, episodes=50, label="Agent"):
    """Run an agent on an environment and return metrics."""
    env = cc.make(env_id)
    rewards = []
    correct = 0
    total_steps = 0

    start = time.time()
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if hasattr(agent, "on_reward"):
                agent.on_reward(reward)
            ep_reward += reward.total
            total_steps += 1
            if info.get("correct", False):
                correct += 1
            if done or truncated:
                break
        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end(env.episode_stats())
        rewards.append(ep_reward)
    elapsed = time.time() - start

    avg = sum(rewards) / len(rewards)
    first_half = sum(rewards[:episodes//2]) / max(1, episodes//2)
    second_half = sum(rewards[episodes//2:]) / max(1, episodes//2)
    improvement = second_half - first_half

    return {
        "label": label,
        "avg_reward": avg,
        "first_half": first_half,
        "second_half": second_half,
        "improvement": improvement,
        "total_steps": total_steps,
        "elapsed": elapsed,
    }


def print_results(results, title):
    """Print a comparison table."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"  {'Agent':<20} {'Avg Reward':>12} {'1st Half':>10} {'2nd Half':>10} {'Learn':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

    for r in sorted(results, key=lambda x: -x["avg_reward"]):
        learn = f"{r['improvement']:+.1f}"
        bar = "#" * min(20, max(0, int(r["avg_reward"])))
        print(f"  {r['label']:<20} {r['avg_reward']:>+10.1f}   {r['first_half']:>+8.1f}   "
              f"{r['second_half']:>+8.1f}   {learn:>6}  {bar}")

    winner = max(results, key=lambda x: x["avg_reward"])
    print(f"\n  Winner: {winner['label']} ({winner['avg_reward']:+.1f} avg reward)")


# ── 3. Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    EPISODES = 50

    print()
    print("*" * 72)
    print("  CogniCore v0.6.0 -- Multi-Agent Benchmark Suite")
    print("  Testing: Q-Learning, SARSA, Genetic, Bandit, RuleBased, Random")
    print("*" * 72)

    # ── BENCHMARK 1: GridWorld ──────────────────────────────────────

    grid_results = []
    agents_grid = {
        "Random":        cc.RandomAgent(),
        "Q-Learning":    cc.QLearningAgent(["UP","DOWN","LEFT","RIGHT"],
                                            learning_rate=0.2, epsilon_decay=0.98),
        "SARSA":         cc.SARSAAgent(["UP","DOWN","LEFT","RIGHT"],
                                        epsilon_decay=0.98),
        "Bandit (UCB1)": cc.BanditAgent(["UP","DOWN","LEFT","RIGHT"]),
        "RuleBased":     RuleBasedGridAgent(),
    }

    for name, agent in agents_grid.items():
        result = benchmark_agent(agent, "GridWorld-v1", EPISODES, name)
        grid_results.append(result)

    print_results(grid_results, "GridWorld-v1 (5x5 Navigation)")

    # ── BENCHMARK 2: ResourceGathering ──────────────────────────────

    resource_results = []
    actions_res = ["GATHER_FOOD","GATHER_WOOD","GATHER_STONE","BUILD","REST","EXPLORE"]

    agents_res = {
        "Random":        cc.RandomAgent(),
        "Q-Learning":    cc.QLearningAgent(actions_res, learning_rate=0.15, epsilon_decay=0.97),
        "Genetic":       cc.GeneticAgent(actions_res, population_size=10,
                                          mutation_rate=0.15, strategy_length=30),
        "Bandit (UCB1)": cc.BanditAgent(actions_res),
        "RuleBased":     RuleBasedResourceAgent(),
    }

    for name, agent in agents_res.items():
        result = benchmark_agent(agent, "ResourceGathering-v1", EPISODES, name)
        resource_results.append(result)

    print_results(resource_results, "ResourceGathering-v1 (Multi-Objective)")

    # ── BENCHMARK 3: Safety Classification ──────────────────────────

    safety_results = []
    agents_safety = {
        "Random":        cc.RandomAgent(),
        "Bandit (UCB1)": cc.BanditAgent(["SAFE", "UNSAFE", "NEEDS_REVIEW"]),
        "RuleBased":     RuleBasedSafetyAgent(),
    }

    for name, agent in agents_safety.items():
        result = benchmark_agent(agent, "RealWorldSafety-v1", EPISODES, name)
        safety_results.append(result)

    print_results(safety_results, "RealWorldSafety-v1 (AI Safety)")

    # ── BENCHMARK 4: GridWorld-Hard ──────────────────────────────────

    hard_results = []
    agents_hard = {
        "Random":        cc.RandomAgent(),
        "Q-Learning":    cc.QLearningAgent(["UP","DOWN","LEFT","RIGHT"],
                                            learning_rate=0.2, epsilon_decay=0.97),
        "RuleBased":     RuleBasedGridAgent(),
    }

    for name, agent in agents_hard.items():
        result = benchmark_agent(agent, "GridWorld-Hard-v1", EPISODES, name)
        hard_results.append(result)

    print_results(hard_results, "GridWorld-Hard-v1 (10x10, 15 traps)")

    # ── Summary ──────────────────────────────────────────────────────

    print()
    print("=" * 72)
    print("  OVERALL SUMMARY")
    print("=" * 72)
    print()
    print("  Agent types tested:")
    print("    - Q-Learning      (tabular RL, epsilon-greedy + TD learning)")
    print("    - SARSA           (on-policy RL, safer exploration)")
    print("    - Genetic         (evolutionary, tournament + crossover + mutation)")
    print("    - Bandit (UCB1)   (exploration-exploitation, no state)")
    print("    - RuleBased       (hand-coded heuristics, zero ML)")
    print("    - Random          (uniform random baseline)")
    print()
    print("  Environments tested:")
    print("    - GridWorld-v1       (5x5 navigation, 3 traps)")
    print("    - GridWorld-Hard-v1  (10x10 navigation, 15 traps)")
    print("    - ResourceGathering  (multi-objective: gather + build + survive)")
    print("    - RealWorldSafety    (jailbreaks, PII, prompt injection)")
    print()
    print("  Key takeaway:")
    print("    NONE of these agents use LLMs.")
    print("    CogniCore works with ANY agent type.")
    print()
    print(f"  pip install cognicore-env  |  v{cc.__version__}  |  {len(cc.list_envs())} envs")
    print("=" * 72)
