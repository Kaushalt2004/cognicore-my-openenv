<h1 align="center">🧠 CogniCore</h1>

<p align="center">
  <strong>Cognitive RL Environments — Memory, Reflection & Reward Shaping built into every env.</strong><br>
  Train smarter agents on procedural mazes, trading, survival & combat — with SB3, RLlib, or any Gymnasium-compatible library.
</p>

<p align="center">
  <a href="https://pypi.org/project/cognicore-env/"><img src="https://img.shields.io/pypi/v/cognicore-env?color=C4703D&label=PyPI" alt="PyPI"/></a>
  <a href="https://github.com/Kaushalt2004/cognicore-my-openenv/actions"><img src="https://github.com/Kaushalt2004/cognicore-my-openenv/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/gymnasium_envs-8-3D6EC4" alt="Gym Envs"/>
  <img src="https://img.shields.io/badge/total_envs-50-3D6EC4" alt="Total Envs"/>
  <img src="https://img.shields.io/badge/SB3-compatible-green" alt="SB3"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License"/>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-environments">Environments</a> •
  <a href="#-benchmark-leaderboard">Benchmarks</a> •
  <a href="#-cognitive-middleware">Middleware</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-integrations">Integrations</a>
</p>

---

## What is CogniCore?

CogniCore provides **cognitive RL environments** — environments where Memory, Reflection, and Reward Shaping are built into the infrastructure, not bolted on as an afterthought.

```python
import cognicore.gym
import gymnasium as gym
from stable_baselines3 import PPO

# Standard Gymnasium API — works with ANY RL library
env = gym.make("cognicore/MazeRunner-v0")
model = PPO("MlpPolicy", env)
model.learn(100_000)
```

**What makes CogniCore different:**

| Feature | Gymnasium | CogniCore |
|---------|-----------|-----------|
| Environments | CartPole, MountainCar | Procedural Mazes, Trading, Survival, Combat |
| Memory | None | Embedding-based retrieval across episodes |
| Difficulty | Fixed | Auto-curriculum (Easy → Hard) |
| Agent comparison | Manual | Built-in Arena with ELO ratings |
| Deployment | None | HuggingFace Hub upload/download |

---

## 🚀 Quickstart

```bash
pip install cognicore-env
pip install stable-baselines3  # optional, for training
```

### Train a PPO agent on a procedural maze:

```python
import cognicore.gym
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("cognicore/MazeRunner-v0")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate
obs, info = env.reset(seed=42)
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(f"Result: {info['event']} in {info['steps']} steps")
```

### Add cognitive memory to ANY environment:

```python
from cognicore.memory import EmbeddingMemory

memory = EmbeddingMemory()
memory.store("hit wall at (3,4)", {"action": "UP", "reward": -0.3})
memory.store("reached goal via (5,2)", {"action": "RIGHT", "reward": +10})

# Semantic retrieval — finds similar experiences by meaning
results = memory.retrieve("near wall at (3,5)", top_k=3)
# Returns the wall experience (semantically similar), not the goal one
```

---

## 🌍 Environments

### Gymnasium-Native (SB3/RLlib compatible)

| Environment | Type | Obs Space | Actions | Description |
|------------|------|-----------|---------|-------------|
| `cognicore/MazeRunner-v0` | Navigation | Box(11) | Discrete(4) | 8×8 procedural maze with fixed walls |
| `cognicore/MazeRunner-Medium-v0` | Navigation | Box(11) | Discrete(4) | 12×12 maze |
| `cognicore/MazeRunner-Hard-v0` | Navigation | Box(11) | Discrete(4) | 16×16 maze |
| `cognicore/GridWorld-v0` | Navigation | Box(11) | Discrete(4) | 5×5 grid with traps |
| `cognicore/GridWorld-Hard-v0` | Navigation | Box(35) | Discrete(4) | 10×10, 15 traps |
| `cognicore/Trading-v0` | Finance | Box(8) | Discrete(3) | Portfolio management |
| `cognicore/Survival-v0` | Planning | Box(9) | Discrete(7) | Long-horizon survival |
| `cognicore/BattleArena-v0` | Combat | Box(9) | Discrete(6) | 2-player grid battle |

All 8 environments pass `gymnasium.utils.env_checker.check_env()` ✅

### Legacy CogniCore Environments (50 total)

GridWorld, MazeRunner, Trading, Survival, ResourceGathering, SafetyClassification, MathReasoning, CodeDebugging, Conversation, Planning, Summarization — each with Easy/Medium/Hard variants.

---

## 📊 Benchmark Leaderboard

Trained with 50K steps | Evaluated over 50 episodes | 3 seeds

| Environment | PPO | DQN | A2C | Random |
|------------|-----|-----|-----|--------|
| GridWorld-v0 | +1.5 | **+1.5** | +0.8 | -4.2 |
| Trading-v0 | **+0.0** | -0.1 | -0.2 | -0.5 |
| Survival-v0 | **+199.9** | +120.3 | +145.2 | +15.0 |
| BattleArena-v0 | **+19.2** | +12.1 | +15.8 | -3.5 |

Run your own: `python benchmarks/leaderboard.py`

### Arena ELO Ratings

```
  Rank  Agent                    ELO    W    L
  [1st] Q-Learning             1232    2    0
  [2nd] SARSA                  1199    1    1
  [3rd] Random                 1169    0    2
```

---

## 🧠 Cognitive Middleware

### 1. Embedding-Based Memory

```python
from cognicore.memory import EmbeddingMemory

memory = EmbeddingMemory(model_name="all-MiniLM-L6-v2")
memory.store("dead end at (3,4)", {"reward": -1, "action": "UP"})

# Retrieves by SEMANTIC similarity, not exact match
results = memory.retrieve("wall near (3,5)")  # Finds the dead end experience
```

### 2. Auto-Curriculum

```python
from cognicore import AutoCurriculum

curriculum = AutoCurriculum(
    env_base="MazeRunner",
    levels=["Easy", "Medium", "Hard"],
    promote_threshold=0.4,
)
# Automatically promotes agent from Easy → Medium → Hard
```

### 3. Arena (ELO Tournament)

```python
from cognicore import Arena

arena = Arena()
arena.add_agent("PPO", ppo_agent)
arena.add_agent("DQN", dqn_agent)
arena.run_tournament(["MazeRunner-v1", "GridWorld-v1"])
arena.print_leaderboard()
```

### 4. CognitiveGymWrapper

```python
from cognicore.memory.embedding import CognitiveGymWrapper

env = gym.make("cognicore/MazeRunner-v0")
env = CognitiveGymWrapper(env)  # Adds memory to ANY env

obs, info = env.reset()
print(info["cognicore_memory"])  # Past similar experiences
print(info["cognicore_advice"])  # Human-readable memory advice
```

---

## 💻 CLI

```bash
# List all environments
python -m cognicore.cli list

# Train an agent
python -m cognicore.cli train --env MazeRunner-v0 --algo PPO --steps 100000

# Benchmark all algorithms
python -m cognicore.cli benchmark --env GridWorld-v0

# Run ELO tournament
python -m cognicore.cli arena --envs MazeRunner-v1,GridWorld-v1
```

---

## 🔗 Integrations

### HuggingFace Hub

```python
from cognicore.integrations.huggingface import upload_model, download_model

# Upload trained model
upload_model(model, "username/ppo-mazerunner", env_id="cognicore/MazeRunner-v0")

# Download and use
model = download_model("username/ppo-mazerunner", algo="PPO")
```

### TensorBoard

```python
from cognicore.logging import CogniCoreCallback

model = PPO("MlpPolicy", env)
model.learn(100_000, callback=CogniCoreCallback("runs/maze_ppo"))
# Then: tensorboard --logdir runs/
```

### Stable Baselines3

All gymnasium envs work with SB3 out of the box:
```python
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
env = gym.make("cognicore/MazeRunner-v0")
model = PPO("MlpPolicy", env)  # Just works
```

---

## 📁 Project Structure

```
cognicore/
├── gym/                    # Gymnasium-native environments (8 envs)
│   ├── __init__.py         # MazeRunner, GridWorld, Trading, Survival
│   └── battle_arena.py     # BattleArena (multi-agent)
├── envs/                   # Legacy CogniCore environments (50 envs)
├── core/                   # CognitiveBoost, Arena, AutoCurriculum
├── memory/                 # EmbeddingMemory (sentence-transformers)
├── middleware/              # Memory, Reflection, Safety, Rewards
├── agents/                 # 14 built-in agent types
├── integrations/           # HuggingFace Hub
├── rendering/              # Pygame renderer
├── logging/                # TensorBoard + CSV logger
├── cli.py                  # Command-line interface
└── server/                 # REST API (FastAPI)
```

---

## 🧪 Testing

```bash
pytest tests/ -x -q          # 425 tests
python examples/full_test.py  # Full system test
python examples/train_sb3.py  # SB3 baselines
```

---

## ⚠️ Known Limitations

- **CognitiveBoost reward shaping** needs tuning — current implementation can hurt exploration in some envs
- **MazeRunner PPO** requires 100K+ steps to learn effectively (8x8 is genuinely hard)
- **Embedding memory** uses random fallback when sentence-transformers isn't installed
- **Pygame rendering** requires `pip install pygame`

---

## 📜 License

MIT — free for research and commercial use.

## 🤝 Contributing

Issues, PRs, and benchmark submissions welcome at [github.com/Kaushalt2004/cognicore-my-openenv](https://github.com/Kaushalt2004/cognicore-my-openenv).
