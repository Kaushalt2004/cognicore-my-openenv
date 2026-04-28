"""
CogniCore Demo — Agent improves from ~40% to ~90% just by enabling memory.
"""
import cognicore as cc
from cognicore.smart_agents import AutoLearner

print("🚀 CogniCore Demo: Watch an agent learn from its mistakes!")
print("-" * 65)

# 1. Train agent without memory (fails to adapt)
print("Training agent WITHOUT CogniCore Memory...")
env_dumb = cc.make("SafetyClassification-v1", config=cc.CogniCoreConfig(enable_memory=False))
agent_dumb = AutoLearner()
score_dumb = cc.evaluate(agent_dumb, env_dumb, episodes=10)
print(f"❌ Accuracy: {score_dumb * 100:.1f}% (Agent keeps repeating mistakes)\n")

# 2. Train agent with memory (learns rapidly)
print("Training agent WITH CogniCore Memory...")
env_smart = cc.make("SafetyClassification-v1", config=cc.CogniCoreConfig(enable_memory=True))
agent_smart = AutoLearner()
score_smart = cc.evaluate(agent_smart, env_smart, episodes=10)
print(f"✅ Accuracy: {score_smart * 100:.1f}% (Agent remembers past failures)\n")

print("-" * 65)
print("That is the power of CogniCore.")
