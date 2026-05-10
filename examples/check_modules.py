"""Quick integration check for all new modules."""
import shutil

print("Testing new modules...")

# 1. Embedding memory
from cognicore.memory.embedding import EmbeddingMemory
mem = EmbeddingMemory(max_size=100)
mem.store("hit wall at position 3,4", {"reward": -1, "action": "UP"})
mem.store("reached goal via 5,2", {"reward": 10, "action": "RIGHT"})
mem.store("fell into trap at 2,3", {"reward": -5, "action": "DOWN"})
results = mem.retrieve("near wall at position 3,5", top_k=2)
print(f"  EmbeddingMemory: {mem.size} stored, {len(results)} retrieved")
for r in results:
    print(f"    sim={r['_similarity']:.2f}: {r['_matched_text']}")

# 2. HuggingFace
from cognicore.integrations.huggingface import upload_model, download_model
print("  HuggingFace: module loaded")

# 3. Logger
from cognicore.logging import TrainingLogger
log = TrainingLogger("_tmp_log", use_tensorboard=False)
log.log_episode(5.0, 10, True)
print(f"  Logger: {log.stats}")
log.close()
shutil.rmtree("_tmp_log", ignore_errors=True)

# 4. Renderer
from cognicore.rendering import MazeRenderer, SurvivalRenderer
print("  Renderer: loaded")

# 5. CLI
from cognicore.cli import main
print("  CLI: loaded")

# 6. Gymnasium
import cognicore.gym
import gymnasium as gym
gym_envs = sorted(e for e in gym.envs.registry.keys() if e.startswith("cognicore/"))
print(f"  Gymnasium: {len(gym_envs)} envs")

# 7. Version
import cognicore
print(f"  Version: {cognicore.__version__}")

# 8. CognitiveGymWrapper
from cognicore.memory.embedding import CognitiveGymWrapper
env = gym.make("cognicore/GridWorld-v0")
env = CognitiveGymWrapper(env)
obs, info = env.reset(seed=42)
obs, r, t, tr, info = env.step(0)
print(f"  CognitiveGymWrapper: memory_stats={info.get('cognicore_memory_stats', {})}")
env.close()

# 9. SB3 quick check
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
env = Monitor(gym.make("cognicore/GridWorld-v0"))
model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
model.learn(total_timesteps=128)
print("  SB3: PPO trains on CogniCore")
env.close()

print("\nALL MODULES OK")
