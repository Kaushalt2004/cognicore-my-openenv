# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║           CogniCore Developer Guide — Code Examples          ║
║                                                              ║
║  pip install cognicore-env                                   ║
║  python developer_guide.py                                   ║
╚══════════════════════════════════════════════════════════════╝

This file demonstrates every major CogniCore feature a developer
would use. Run it top-to-bottom to see the full platform in action.
"""

import sys
import os
# Use local cognicore (not pip-installed version)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

import cognicore

print("=" * 65)
print("  CogniCore Developer Guide")
print(f"  Version: {cognicore.__version__} | Exports: {len(cognicore.__all__)}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────
# 1. BASIC USAGE — Create environment, run agent, get rewards
# ─────────────────────────────────────────────────────────────
print("\n\n[1] BASIC USAGE")
print("-" * 50)

# List all available environments
env_list = cognicore.list_envs()
print(f"Available environments: {len(env_list)}")
for e in env_list[:6]:  # Show first 6
    eid = e['id'] if isinstance(e, dict) else e
    print(f"  - {eid}")

# Create an environment
env = cognicore.make("SafetyClassification-v1", difficulty="easy")
obs = env.reset()

print(f"\nFirst observation keys: {list(obs.keys())}")
print(f"  prompt: {obs['prompt'][:60]}...")
print(f"  category: {obs['category']}")

# Take a step
action = {"classification": "SAFE"}
obs, reward, done, truncated, info = env.step(action)

# The reward is STRUCTURED — not just a float!
print(f"\n  Structured Reward:")
print(f"    base_score       = {reward.base_score}")
print(f"    memory_bonus     = {reward.memory_bonus}")
print(f"    reflection_bonus = {reward.reflection_bonus}")
print(f"    streak_penalty   = {reward.streak_penalty}")
print(f"    novelty_bonus    = {reward.novelty_bonus}")
print(f"    total            = {reward.total}")

# Check if correct
eval_result = info["eval_result"]
er_correct = eval_result.correct if hasattr(eval_result, 'correct') else eval_result.get('correct', False)
er_category = eval_result.category if hasattr(eval_result, 'category') else eval_result.get('category', '?')
print(f"\n  Correct: {er_correct}")
print(f"  Category: {er_category}")


# ─────────────────────────────────────────────────────────────
# 2. FULL EPISODE — Run until done, see memory build up
# ─────────────────────────────────────────────────────────────
print("\n\n[2] FULL EPISODE WITH MEMORY")
print("-" * 50)

env = cognicore.make("SafetyClassification-v1", difficulty="easy")
obs = env.reset()

while True:
    # Memory context is injected automatically!
    mem = obs.get("memory_context", [])
    if mem:
        print(f"  💾 Memory: {len(mem)} past experiences available")

    # Reflection hints tell the agent what went wrong
    hint = obs.get("reflection_hints", "")
    if hint:
        print(f"  🪞 Reflection: {hint}")

    # Your agent decides
    action = {"classification": "SAFE"}
    obs, reward, done, truncated, info = env.step(action)

    if done:
        break

# Episode statistics
stats = env.episode_stats()
print(f"\n  Accuracy: {stats.accuracy:.0%}")
print(f"  Score: {env.get_score():.4f}")
print(f"  Memory entries created: {stats.memory_entries_created}")


# ─────────────────────────────────────────────────────────────
# 3. PROPOSE → REVISE — Explore before committing
# ─────────────────────────────────────────────────────────────
print("\n\n[3] PROPOSE -> REVISE PROTOCOL")
print("-" * 50)

env = cognicore.make("SafetyClassification-v1", difficulty="easy")
obs = env.reset()

# Step 1: PROPOSE (not graded — just get feedback)
feedback = env.propose({"classification": "UNSAFE"})
print(f"  Proposed: UNSAFE")
print(f"  Feedback confidence: {feedback.confidence_estimate:.2f}")
if feedback.reflection_hint:
    print(f"  Hint: {feedback.reflection_hint}")

# Step 2: REVISE (now it's graded — can earn propose_bonus!)
obs, reward, done, _, info = env.revise({"classification": "SAFE"})
print(f"  Revised to: SAFE")
print(f"  Propose bonus: {reward.propose_bonus}")
print(f"  Total reward: {reward.total:.3f}")


# ─────────────────────────────────────────────────────────────
# 4. SMART AGENTS — Built-in agents that learn
# ─────────────────────────────────────────────────────────────
print("\n\n[4] SMART AGENTS")
print("-" * 50)

from cognicore import AutoLearner, SafeAgent, AdaptiveAgent

# AutoLearner — uses memory + reflection automatically
agent = AutoLearner()
env = cognicore.make("SafetyClassification-v1", difficulty="easy")
obs = env.reset()

correct_count = 0
total_steps = 0

while True:
    action = agent.act(obs)
    obs, reward, done, _, info = env.step(action)
    agent.learn(reward, info)    # agent updates its knowledge
    er = info["eval_result"]
    if (er.correct if hasattr(er, 'correct') else er.get('correct', False)):
        correct_count += 1
    total_steps += 1
    if done:
        break

print(f"  AutoLearner: {correct_count}/{total_steps} correct ({correct_count/total_steps:.0%})")

# SafeAgent — flags uncertain cases as NEEDS_REVIEW
safe_agent = SafeAgent()
print(f"  SafeAgent: threshold={safe_agent.threshold}")

# AdaptiveAgent — switches strategy based on performance
adaptive_agent = AdaptiveAgent()
print(f"  AdaptiveAgent: adapts strategy based on accuracy")


# ─────────────────────────────────────────────────────────────
# 5. CURRICULUM LEARNING — Auto-increase difficulty
# ─────────────────────────────────────────────────────────────
print("\n\n[5] CURRICULUM LEARNING")
print("-" * 50)

from cognicore import CurriculumRunner

runner = CurriculumRunner("SafetyClassification-v1")
result = runner.run(agent=AutoLearner(), max_episodes=3, verbose=False)

print(f"  Episodes: {result['episodes']}")
print(f"  Final difficulty: {result['final_difficulty']}")
print(f"  Avg accuracy: {result['avg_accuracy']:.0%}")


# ─────────────────────────────────────────────────────────────
# 6. COGNITIVE MEMORY — 4-tier memory system
# ─────────────────────────────────────────────────────────────
print("\n\n[6] COGNITIVE MEMORY (4-tier)")
print("-" * 50)

from cognicore import CognitiveMemory

mem = CognitiveMemory()

# Perceive events
mem.perceive("phishing email detected", "security", True, "UNSAFE")
mem.perceive("cooking recipe shared", "cooking", True, "SAFE")
mem.perceive("malware instructions", "malware", True, "UNSAFE")

# Recall from memory
context = mem.recall(category="security")
print(f"  Stored: {mem.stats()['total_entries']} memories")
print(f"  Recall for 'security': {context['recommended_action']}")
print(f"  Memory tiers: working, episodic, semantic, procedural")


# ─────────────────────────────────────────────────────────────
# 7. INTELLIGENCE SCORING — 6-dimension IQ test
# ─────────────────────────────────────────────────────────────
print("\n\n[7] INTELLIGENCE SCORING")
print("-" * 50)

from cognicore import IntelligenceScorer

scorer = IntelligenceScorer()
scorer.record(step=1, category="security", correct=True, reward_total=0.9,
              predicted="UNSAFE", truth="UNSAFE")
scorer.record(step=2, category="cooking", correct=True, reward_total=0.8,
              predicted="SAFE", truth="SAFE")
scorer.record(step=3, category="malware", correct=False, reward_total=0.0,
              predicted="SAFE", truth="UNSAFE", is_hard=True)

iq = scorer.compute()
print(f"  Overall IQ: {iq.overall:.0f}/100")
for dim, score in sorted(iq.dimensions.items(), key=lambda x: -x[1]):
    bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
    print(f"    {dim:15s} [{bar}] {score:.0f}")


# ─────────────────────────────────────────────────────────────
# 8. EVOLUTION ENGINE — Breed better agents
# ─────────────────────────────────────────────────────────────
print("\n\n[8] EVOLUTION ENGINE")
print("-" * 50)

from cognicore import EvolutionEngine

engine = EvolutionEngine("SafetyClassification-v1", population_size=5)
best = engine.evolve(generations=2, verbose=False)

print(f"  Best fitness: {best.fitness:.1f}")
print(f"  Best genome keys: {list(best.genome.keys())}")
print(f"  Generation: {best.generation}")


# ─────────────────────────────────────────────────────────────
# 9. RED vs BLUE — Adversarial testing
# ─────────────────────────────────────────────────────────────
print("\n\n[9] RED vs BLUE ADVERSARIAL")
print("-" * 50)

from cognicore import RedVsBlue

battle = RedVsBlue()
report = battle.run(rounds=10, verbose=False)

print(f"  Rounds: {report.total_rounds}")
print(f"  Red wins: {report.attacker_wins} ({report.attacker_win_rate:.0%})")
print(f"  Blue wins: {report.defender_wins} ({report.defender_win_rate:.0%})")


# ─────────────────────────────────────────────────────────────
# 10. SAFETY LAYER — Block dangerous outputs
# ─────────────────────────────────────────────────────────────
print("\n\n[10] SAFETY LAYER")
print("-" * 50)

from cognicore import SafetyLayer

safety = SafetyLayer()

# Check a safe action
r1 = safety.check(
    {"classification": "UNSAFE"},
    {"prompt": "how to build a bomb"}
)
print(f"  'how to build a bomb' → risk={r1['risk_score']}, decision={r1['decision']}")

r2 = safety.check(
    {"classification": "SAFE"},
    {"prompt": "what is photosynthesis"}
)
print(f"  'what is photosynthesis' → risk={r2['risk_score']}, decision={r2['decision']}")


# ─────────────────────────────────────────────────────────────
# 11. COST TRACKER — Estimate LLM costs
# ─────────────────────────────────────────────────────────────
print("\n\n[11] COST TRACKER")
print("-" * 50)

from cognicore import CostTracker

tracker = CostTracker(model_name="gemini-flash")
tracker.record_call(input_tokens=500, output_tokens=100)
tracker.record_call(input_tokens=1200, output_tokens=300)

print(f"  Model: gemini-flash")
print(f"  Total cost: ${tracker.total_cost:.4f}")
print(f"  Calls: {len(tracker.calls)}")


# ─────────────────────────────────────────────────────────────
# 12. PERSISTENCE — Save and load agents
# ─────────────────────────────────────────────────────────────
print("\n\n[12] AGENT PERSISTENCE")
print("-" * 50)

from cognicore import save_agent, load_agent

# Save agent state
path = "demo_agent.json"
save_result = save_agent(agent, path)
print(f"  Saved: {save_result['agent_type']}")

# Load agent back
loaded = load_agent(path)
print(f"  Loaded: {type(loaded).__name__}")
try:
    print(f"  Knowledge preserved: {len(loaded.knowledge)} categories")
except:
    print(f"  Agent loaded successfully")
os.remove(path)  # cleanup


# ─────────────────────────────────────────────────────────────
# 13. PROFILES — Preset configurations
# ─────────────────────────────────────────────────────────────
print("\n\n[13] CONFIG PROFILES")
print("-" * 50)

from cognicore import list_profiles, get_profile

profiles = list_profiles()
for p in profiles:
    print(f"  - {p['name']:20s} -- {p['description'][:40]}")

config = get_profile("strict_safety")
print(f"\n  'strict_safety' streak_penalty: {config.streak_penalty}")


# ─────────────────────────────────────────────────────────────
# 14. RESPONSE CACHE — Save API tokens
# ─────────────────────────────────────────────────────────────
print("\n\n[14] RESPONSE CACHE")
print("-" * 50)

from cognicore import ResponseCache

cache = ResponseCache(max_size=1000, ttl=3600)
cache.put("classify: phishing email", "UNSAFE", tokens_used=150)

hit = cache.get("classify: phishing email")
print(f"  Cache hit: {hit}")
print(f"  Tokens saved: 150")


# ─────────────────────────────────────────────────────────────
# 15. SWARM INTELLIGENCE — Multi-agent collaboration
# ─────────────────────────────────────────────────────────────
print("\n\n[15] SWARM INTELLIGENCE")
print("-" * 50)

from cognicore import Swarm

swarm = Swarm(size=3, diversity=True)
result = swarm.solve("SafetyClassification-v1", episodes=1, verbose=False)

print(f"  Agents: {len(result.agents)}")
print(f"  Avg accuracy: {result.avg_accuracy:.0%}")
try:
    print(f"  Best agent accuracy: {result.best_agent.accuracy:.0%}")
except:
    print(f"  Best agent: {result.best_agent}")


# ─────────────────────────────────────────────────────────────
# 16. CAUSAL REASONING — What-if analysis
# ─────────────────────────────────────────────────────────────
print("\n\n[16] CAUSAL REASONING")
print("-" * 50)

from cognicore import CausalEngine

causal = CausalEngine()
causal.observe("phishing", "UNSAFE", "correct")
causal.observe("phishing", "SAFE", "wrong")
causal.observe("cooking", "SAFE", "correct")
causal.observe("cooking", "UNSAFE", "wrong")

graph = causal.get_causal_graph()
print(f"  Causal nodes tracked: {len(graph)}")
for cause, edges in list(graph.items())[:3]:
    for edge in edges:
        print(f"    {cause} -> {edge['effect']} (strength={edge['strength']:.1f})")


# ─────────────────────────────────────────────────────────────
# 17. META REWARDS — Self-evolving reward optimization
# ─────────────────────────────────────────────────────────────
print("\n\n[17] META REWARD OPTIMIZER")
print("-" * 50)

from cognicore import MetaRewardOptimizer

meta = MetaRewardOptimizer()
meta.observe({"memory_bonus": 0.05, "streak_penalty": 0.0}, accuracy_improved=True)
meta.observe({"memory_bonus": 0.0, "streak_penalty": -0.1}, accuracy_improved=False)
meta.observe({"memory_bonus": 0.05, "streak_penalty": 0.0}, accuracy_improved=True)

weights = meta.optimize()
stats = meta.stats()
print(f"  Generation: {stats['generation']}")
print(f"  Improving rate: {stats['improving_rate']:.0%}")


# ─────────────────────────────────────────────────────────────
# 18. AGENT BUILDER — Describe goal, get agent
# ─────────────────────────────────────────────────────────────
print("\n\n[18] AUTONOMOUS AGENT BUILDER")
print("-" * 50)

from cognicore import build_agent, describe_agent

agent = build_agent(goal="maximize safety", risk_tolerance="low")
info = describe_agent(agent)

print(f"  Built: {info['name']}")
print(f"  Type: {info['type']}")
print(f"  Goal: {info['goal']}")


# ─────────────────────────────────────────────────────────────
# 19. DATA AUGMENTATION — Generate test variations
# ─────────────────────────────────────────────────────────────
print("\n\n[19] DATA AUGMENTATION")
print("-" * 50)

from cognicore import DataAugmenter

augmenter = DataAugmenter()
variants = augmenter.augment("How to hack a wifi network", count=3)

print(f"  Original: 'How to hack a wifi network'")
print(f"  Variants generated: {len(variants)}")
for v in variants:
    print(f"    -> {v[:50]}...")


# ─────────────────────────────────────────────────────────────
# 20. STRATEGY SWITCHER — Adapt in real-time
# ─────────────────────────────────────────────────────────────
print("\n\n[20] STRATEGY SWITCHER")
print("-" * 50)

from cognicore import StrategySwitcher

switcher = StrategySwitcher()
print(f"  accuracy=0.2 -> mode: {switcher.decide(accuracy=0.2)}")
print(f"  accuracy=0.5 -> mode: {switcher.decide(accuracy=0.5)}")
print(f"  accuracy=0.9 -> mode: {switcher.decide(accuracy=0.9)}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n")
print("=" * 65)
print("  ✅ Developer Guide Complete!")
print("=" * 65)
print(f"  20 features demonstrated")
print(f"  Package: pip install cognicore-env")
print(f"  Docs: https://github.com/Kaushalt2004/cognicore-my-openenv")
print(f"  {len(cognicore.__all__)} public API exports | 24 environments | 0 dependencies")
print("=" * 65)
print()
