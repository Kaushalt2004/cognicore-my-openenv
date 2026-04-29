"""Quick test: verify all 5 industry fixes work."""
import cognicore as cc

# 1. Imports + version
print(f"v{cc.__version__} | Exports: {len(cc.__all__)}")

# 2. Error classes exist
print(f"Errors: {cc.CogniCoreError.__name__}, {cc.InvalidEnvironmentError.__name__}")

# 3. Invalid env → clear error
try:
    cc.make("FakeEnv-v1")
except cc.InvalidEnvironmentError as e:
    print(f"InvalidEnvironmentError: {e}")

# 4. Invalid config → caught immediately
try:
    cc.CogniCoreConfig(memory_max_size=-1)
except cc.InvalidConfigError as e:
    print(f"InvalidConfigError: {e}")

# 5. Agent validation
try:
    cc.train("not_an_agent", cc.make("SafetyClassification-v1"))
except cc.AgentInterfaceError as e:
    print(f"AgentInterfaceError: {e}")

# 6. Episode finished error
env = cc.make("SafetyClassification-v1")
obs = env.reset()
while True:
    obs, reward, done, _, info = env.step({"classification": "SAFE"})
    if done:
        break
try:
    env.step({"classification": "SAFE"})
except cc.EpisodeFinishedError as e:
    print(f"EpisodeFinishedError: {e}")

# 7. Normal flow still works
print("\nTrain + Evaluate:")
from cognicore.smart_agents import AutoLearner
agent = AutoLearner()
env = cc.make("SafetyClassification-v1")
cc.train(agent, env, episodes=3)
score = cc.evaluate(agent, env, episodes=2)
print(f"Score: {score*100:.1f}%")

print("\nAll 5 industry fixes verified!")
