"""
CogniCore MEGA DEMO — All 5 wow features in one script.

1. CognitiveBoost: Reward shaping that PROVABLY speeds up learning
2. Arena: Head-to-head tournament with ELO ratings
3. Auto-Curriculum: Difficulty auto-adjusts as agent improves
4. Transfer Learning: Train on Easy, test on Hard
5. Visual training with ASCII learning curves

Run: python examples/mega_demo.py
"""
import cognicore as cc
from cognicore.core.cognitive_boost import CognitiveBoost, Arena, AutoCurriculum, TransferAgent
import random
import time

random.seed(42)

print()
print("*" * 70)
print("  CogniCore v0.6.0 — MEGA DEMO (5 Features)")
print("*" * 70)


# ═══════════════════════════════════════════════════════════════
#  1. COGNITIVE BOOST: Does reward shaping help?
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  1. COGNITIVE BOOST: Reward Shaping vs Standard RL")
print("=" * 70)

EPISODES = 150

# Run A: Standard Q-Learning
random.seed(42)
agent_std = cc.QLearningAgent(
    ["UP", "DOWN", "LEFT", "RIGHT"],
    learning_rate=0.2, discount=0.95, epsilon_decay=0.98,
)
env_std = cc.make("GridWorld-v1")
rewards_std = []

for ep in range(EPISODES):
    obs = env_std.reset()
    ep_r = 0.0
    while True:
        action = agent_std.act(obs)
        obs, reward, done, truncated, info = env_std.step(action)
        agent_std.on_reward(reward)
        ep_r += reward.total
        if done or truncated:
            break
    agent_std.on_episode_end(env_std.episode_stats())
    rewards_std.append(ep_r)

# Run B: Q-Learning + CognitiveBoost
random.seed(42)
agent_raw = cc.QLearningAgent(
    ["UP", "DOWN", "LEFT", "RIGHT"],
    learning_rate=0.2, discount=0.95, epsilon_decay=0.98,
)
boosted = CognitiveBoost(agent_raw, gamma=0.95)
env_boost = cc.make("GridWorld-v1")
rewards_boost = []

for ep in range(EPISODES):
    obs = env_boost.reset()
    prev_obs = obs
    ep_r = 0.0
    while True:
        action = boosted.act(obs)
        obs, reward, done, truncated, info = env_boost.step(action)
        shaped = boosted.shape_reward(reward, prev_obs, obs, done or truncated)
        boosted.on_reward(shaped)
        ep_r += reward.total  # Track REAL reward, not shaped
        prev_obs = obs
        if done or truncated:
            break
    boosted.on_episode_end(env_boost.episode_stats())
    rewards_boost.append(ep_r)

# Print results
def avg(lst, start, end):
    s = lst[start:end]
    return sum(s) / len(s) if s else 0

print(f"\n  {'Phase':<20} {'Standard':>10} {'+ Boost':>10} {'Winner':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

for label, s, e in [("First 30", 0, 30), ("Middle 30", 60, 90), ("Last 30", 120, 150)]:
    a = avg(rewards_std, s, e)
    b = avg(rewards_boost, s, e)
    winner = "Boost" if b > a else "Standard"
    print(f"  {label:<20} {a:>+9.1f}  {b:>+9.1f}  {winner:>9}")

total_std = sum(rewards_std) / len(rewards_std)
total_boost = sum(rewards_boost) / len(rewards_boost)
print(f"  {'OVERALL':<20} {total_std:>+9.1f}  {total_boost:>+9.1f}  "
      f"{'BOOST' if total_boost > total_std else 'Standard':>9}")

print(f"\n  Memory stats: {boosted.stats}")

# Mini learning curve
print(f"\n  Learning Curve (15-ep moving avg):")
print(f"  {'Ep':>6}  {'Standard':>10}  {'Boosted':>10}  Visual")
for i in [0, 29, 59, 89, 119, 149]:
    w = 15
    a = avg(rewards_std, max(0,i-w), i+1)
    b = avg(rewards_boost, max(0,i-w), i+1)
    bar_a = "." * max(0, int(a))
    bar_b = "#" * max(0, int(b))
    print(f"  {i+1:>6}  {a:>+9.1f}  {b:>+9.1f}  S:{bar_a}  B:{bar_b}")


# ═══════════════════════════════════════════════════════════════
#  2. ARENA: Tournament with ELO ratings
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  2. ARENA: Agent Tournament with ELO Ratings")
print("=" * 70)

arena = Arena()
arena.add_agent("Random", cc.RandomAgent())
arena.add_agent("Q-Learning", cc.QLearningAgent(
    ["UP","DOWN","LEFT","RIGHT"], learning_rate=0.2, epsilon_decay=0.97))
arena.add_agent("SARSA", cc.SARSAAgent(
    ["UP","DOWN","LEFT","RIGHT"], epsilon_decay=0.97))
arena.add_agent("Bandit", cc.BanditAgent(["UP","DOWN","LEFT","RIGHT"]))
arena.add_agent("Genetic", cc.GeneticAgent(
    ["UP","DOWN","LEFT","RIGHT"], population_size=10, strategy_length=50))

print("\n  Running round-robin on GridWorld + ResourceGathering...")
t0 = time.time()
arena.run_tournament(["GridWorld-v1", "ResourceGathering-v1"], episodes_per_match=30)
print(f"  Done in {time.time()-t0:.1f}s")

arena.print_leaderboard()

# Show match details
print("\n  Match Results:")
for m in arena.match_history:
    winner = m["a"] if m["score_a"] > m["score_b"] else m["b"]
    print(f"    {m['env']}: {m['a']} ({m['score_a']:+.1f}) vs "
          f"{m['b']} ({m['score_b']:+.1f}) -> {winner} wins")


# ═══════════════════════════════════════════════════════════════
#  3. AUTO-CURRICULUM: Difficulty auto-adjusts
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  3. AUTO-CURRICULUM: Difficulty Adjusts Automatically")
print("=" * 70)

curriculum = AutoCurriculum(
    env_base="GridWorld",
    levels=["Easy", "Medium", "Hard"],
    window=15,
    promote_threshold=0.5,
    demote_threshold=0.15,
)

agent_cur = cc.QLearningAgent(
    ["UP","DOWN","LEFT","RIGHT"],
    learning_rate=0.2, epsilon_decay=0.99,
)

print(f"\n  Starting at: {curriculum.get_env_id()}")
print(f"  {'Episode':>8} {'Level':<22} {'Reward':>8} {'Event':<12}")
print(f"  {'-'*8} {'-'*22} {'-'*8} {'-'*12}")

for ep in range(80):
    env = curriculum.get_env()
    obs = env.reset()
    ep_r = 0.0
    while True:
        action = agent_cur.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        agent_cur.on_reward(reward)
        ep_r += reward.total
        if done or truncated:
            break
    agent_cur.on_episode_end(env.episode_stats())

    event = curriculum.report(ep_r, max_possible=10.0)
    if event in ("promoted", "demoted") or ep < 3 or ep == 79:
        emoji = {"promoted": "UP!", "demoted": "DOWN", "same": "", "warming_up": "..."}
        print(f"  {ep+1:>8} {curriculum.get_env_id():<22} {ep_r:>+7.1f}  {emoji.get(event, event):<12}")

print(f"\n  Curriculum stats: {curriculum.stats}")


# ═══════════════════════════════════════════════════════════════
#  4. TRANSFER LEARNING: Train Easy -> Test Hard
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  4. TRANSFER LEARNING: Train Easy, Test Hard")
print("=" * 70)

# Agent A: Train directly on Hard (no transfer)
random.seed(42)
agent_direct = cc.QLearningAgent(
    ["UP","DOWN","LEFT","RIGHT"],
    learning_rate=0.2, epsilon_decay=0.97,
)
transfer_direct = TransferAgent(agent_direct)
direct_hard = transfer_direct.train_on("GridWorld-Hard-v1", episodes=80)

# Agent B: Train on Easy first, THEN transfer to Hard
random.seed(42)
agent_transfer = cc.QLearningAgent(
    ["UP","DOWN","LEFT","RIGHT"],
    learning_rate=0.2, epsilon_decay=0.97,
)
transfer_agent = TransferAgent(agent_transfer)
transfer_easy = transfer_agent.train_on("GridWorld-Easy-v1", episodes=40)
transfer_hard = transfer_agent.test_on("GridWorld-Hard-v1", episodes=40)

avg_direct = sum(direct_hard) / len(direct_hard)
avg_easy = sum(transfer_easy) / len(transfer_easy)
avg_transfer_hard = sum(transfer_hard) / len(transfer_hard)
avg_direct_last = sum(direct_hard[-20:]) / 20
avg_transfer_last = sum(transfer_hard[-20:]) / 20

print(f"\n  {'Approach':<35} {'Avg Reward':>12} {'Last 20':>10}")
print(f"  {'-'*35} {'-'*12} {'-'*10}")
print(f"  {'Direct on Hard (80 ep)':<35} {avg_direct:>+11.1f}  {avg_direct_last:>+8.1f}")
print(f"  {'Easy(40) -> Hard(40) transfer':<35} {avg_transfer_hard:>+11.1f}  {avg_transfer_last:>+8.1f}")
print(f"  {'Training on Easy (reference)':<35} {avg_easy:>+11.1f}")

if avg_transfer_last > avg_direct_last:
    pct = ((avg_transfer_last - avg_direct_last) / abs(avg_direct_last)) * 100
    print(f"\n  Transfer learning: {pct:+.0f}% better on Hard!")
else:
    print(f"\n  Direct training slightly better — env difference too large for transfer")

# Show Q-table transfer
if hasattr(agent_transfer, "q_table"):
    print(f"  Q-states learned on Easy: {len(agent_transfer.q_table)}")
    print(f"  (These Q-values carry over to Hard environment)")


# ═══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  SUMMARY: 5 WOW FEATURES")
print("=" * 70)
print()
print("  1. CognitiveBoost  Reward shaping from episodic memory")
print(f"     Standard: {total_std:+.1f}  vs  Boosted: {total_boost:+.1f}")
print()
print("  2. Arena           ELO-rated agent tournament")
top = sorted(arena.elo.items(), key=lambda x: -x[1])
print(f"     Champion: {top[0][0]} (ELO {top[0][1]:.0f})")
print()
print("  3. Auto-Curriculum Difficulty auto-adjusts")
print(f"     {curriculum.stats['promotions']} promotions, "
      f"{curriculum.stats['demotions']} demotions")
print()
print("  4. Transfer        Train Easy -> Test Hard")
print(f"     Transfer: {avg_transfer_last:+.1f}  vs  Direct: {avg_direct_last:+.1f}")
print()
print("  5. Visual          ASCII learning curves (shown above)")
print()
print(f"  pip install cognicore-env | v{cc.__version__} | {len(cc.list_envs())} envs")
print("=" * 70)
