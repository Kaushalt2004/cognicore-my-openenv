"""
CogniCore v0.7.0 — World's Best Environments Demo

Real decision-making environments, not prompt optimization.
Every env provides Memory + Reflection as infrastructure.

Run: python examples/world_demo.py
"""
import cognicore as cc
from cognicore.core.cognitive_boost import CognitiveBoost, Arena, AutoCurriculum
import random
import time

random.seed(42)

print()
print("*" * 70)
print("  CogniCore — World-Class Environment Framework")
print("  50 environments | 14 agent types | Memory + Reflection built-in")
print("*" * 70)


# ═══════════════════════════════════════════════════════════════
#  1. MazeRunner — Where memory ACTUALLY helps
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  MAZE RUNNER (8x8 procedural maze, fixed walls)")
print("=" * 70)

env = cc.make("MazeRunner-v1")
agent = cc.QLearningAgent(["UP","DOWN","LEFT","RIGHT"],
                           learning_rate=0.3, epsilon_decay=0.99)

obs = env.reset()
print(f"\n  Generated maze:")
print("  " + env.render().replace("\n", "\n  "))

total_reward = 0
steps = 0
while True:
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    agent.on_reward(reward)
    total_reward += reward.total
    steps += 1
    if done or truncated:
        break

event = reward.metadata.get("event", "timeout") if hasattr(reward, "metadata") else "done"
print(f"\n  Result: {event} in {steps} steps | Reward: {total_reward:+.1f}")
print(f"  Agent's explored path:")
print("  " + env.render().replace("\n", "\n  "))


# ═══════════════════════════════════════════════════════════════
#  2. Trading — Financial decisions
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  TRADING (portfolio management, market regimes)")
print("=" * 70)

env = cc.make("Trading-v1")
agent = cc.BanditAgent(["HOLD", "BUY", "SELL"])

obs = env.reset()
print(f"\n  Starting: ${obs['portfolio_value']:,.2f}")

for ep in range(3):
    obs = env.reset()
    total_r = 0
    while True:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.on_reward(reward)
        total_r += reward.total
        if done or truncated:
            break
    pv = obs.get('portfolio_value', obs.get('cash', 0))
    regime = obs.get('volatility_regime', 'unknown')
    print(f"  Episode {ep+1}: Portfolio ${pv:,.2f} | "
          f"Regime: {regime} | Reward: {total_r:+.1f}")


# ═══════════════════════════════════════════════════════════════
#  3. Survival — Long-horizon planning
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  SURVIVAL (health, hunger, crafting, threats, weather)")
print("=" * 70)

env = cc.make("Survival-v1")
actions_surv = ["FORAGE","HUNT","BUILD_SHELTER","CRAFT_TOOL","REST","EXPLORE","DEFEND"]
agent = cc.GeneticAgent(actions_surv, population_size=10, strategy_length=50)

best_survival = 0
for ep in range(5):
    obs = env.reset()
    total_r = 0
    while True:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent.on_reward(reward)
        total_r += reward.total
        if done or truncated:
            break
    agent.on_episode_end(env.episode_stats())
    survived = obs.get("steps_survived", 0)
    best_survival = max(best_survival, survived)
    health = obs.get("health", 0)
    status = "ALIVE" if obs.get("alive", False) else "DEAD"
    print(f"  Episode {ep+1}: Survived {survived} days | Health: {health:.0f} | "
          f"Status: {status} | Reward: {total_r:+.1f}")

print(f"\n  Best survival: {best_survival} days")


# ═══════════════════════════════════════════════════════════════
#  4. Arena — ELO Tournament across ALL new envs
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  ARENA: ELO Tournament (5 agents x 3 environments)")
print("=" * 70)

arena = cc.Arena()
arena.add_agent("Random", cc.RandomAgent())
arena.add_agent("Q-Learning", cc.QLearningAgent(
    ["UP","DOWN","LEFT","RIGHT","HOLD","BUY","SELL",
     "FORAGE","HUNT","BUILD_SHELTER","CRAFT_TOOL","REST","EXPLORE","DEFEND"],
    learning_rate=0.2, epsilon_decay=0.97))
arena.add_agent("SARSA", cc.SARSAAgent(
    ["UP","DOWN","LEFT","RIGHT","HOLD","BUY","SELL",
     "FORAGE","HUNT","BUILD_SHELTER","CRAFT_TOOL","REST","EXPLORE","DEFEND"],
    epsilon_decay=0.97))
arena.add_agent("Bandit", cc.BanditAgent(
    ["UP","DOWN","LEFT","RIGHT","HOLD","BUY","SELL",
     "FORAGE","HUNT","BUILD_SHELTER","CRAFT_TOOL","REST","EXPLORE","DEFEND"]))
arena.add_agent("Genetic", cc.GeneticAgent(
    ["UP","DOWN","LEFT","RIGHT","HOLD","BUY","SELL",
     "FORAGE","HUNT","BUILD_SHELTER","CRAFT_TOOL","REST","EXPLORE","DEFEND"],
    population_size=8, strategy_length=50))

print("\n  Running tournament on MazeRunner + Trading + Survival...")
t0 = time.time()
arena.run_tournament(
    ["MazeRunner-v1", "Trading-v1", "Survival-v1"],
    episodes_per_match=20,
)
print(f"  Completed in {time.time()-t0:.1f}s")

arena.print_leaderboard()


# ═══════════════════════════════════════════════════════════════
#  5. Auto-Curriculum on MazeRunner
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  AUTO-CURRICULUM: MazeRunner Easy -> Medium -> Hard")
print("=" * 70)

curriculum = cc.AutoCurriculum(
    env_base="MazeRunner",
    levels=["Easy", "Medium", "Hard"],
    window=10,
    promote_threshold=0.4,
    demote_threshold=0.1,
)

agent_c = cc.QLearningAgent(
    ["UP","DOWN","LEFT","RIGHT"],
    learning_rate=0.3, epsilon_decay=0.995,
)

events = []
for ep in range(60):
    env = curriculum.get_env()
    obs = env.reset()
    ep_r = 0
    while True:
        action = agent_c.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent_c.on_reward(reward)
        ep_r += reward.total
        if done or truncated:
            break
    agent_c.on_episode_end(env.episode_stats())
    event = curriculum.report(ep_r, max_possible=10.0)
    if event in ("promoted", "demoted"):
        events.append((ep+1, event, curriculum.levels[curriculum.current_level]))
        print(f"  Episode {ep+1}: {event.upper()} to {curriculum.levels[curriculum.current_level]}")

print(f"\n  Journey: {' -> '.join(e[2] for e in events) if events else 'stayed on ' + curriculum.levels[curriculum.current_level]}")
print(f"  Curriculum stats: {curriculum.stats}")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  CogniCore — ENVIRONMENT CATALOGUE")
print("=" * 70)
print()

envs = cc.list_envs()
env_ids = [e["id"] if isinstance(e, dict) else str(e) for e in envs]
categories = {}
for e in env_ids:
    name = e.split("-")[0]
    if name not in categories:
        categories[name] = []
    categories[name].append(e)

for cat, env_list in sorted(categories.items()):
    variants = ", ".join(e.split("-")[1] if "-" in e else "v1" for e in env_list)
    print(f"  {cat:<25} [{variants}]")

print(f"\n  TOTAL: {len(envs)} environments")
print(f"  Decision-making envs: MazeRunner, Trading, Survival, GridWorld, ResourceGathering")
print(f"  Classification envs: Safety, Code, Math, Conversation, Planning, Summarization")
print()
print(f"  14 agent types | Arena + ELO | AutoCurriculum | Transfer Learning")
print(f"  pip install cognicore-env | v{cc.__version__}")
print("=" * 70)
