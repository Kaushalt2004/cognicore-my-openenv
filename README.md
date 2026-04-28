<h1 align="center">🧠 CogniCore</h1>

<p align="center">
  <strong>Debug and test AI agents like code.</strong><br>
  Train, evaluate, and improve AI systems using memory, feedback, and structured environments.
</p>

<p align="center">
  <a href="https://pypi.org/project/cognicore-env/"><img src="https://img.shields.io/pypi/v/cognicore-env?color=C4703D&label=PyPI" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/tests-passing-3D8C6C" alt="Tests"/>
  <img src="https://img.shields.io/badge/environments-24-3D6EC4" alt="Environments"/>
</p>

---

## 🚀 Quickstart

```bash
pip install cognicore-env
```

```python
import cognicore as cc
from cognicore.agents import RandomAgent

# 1. Create your agent
agent = RandomAgent()

# 2. Create the environment
env = cc.make("SafetyClassification-v1", difficulty="easy")

# 3. Train the agent
cc.train(agent, env, episodes=5)

# 4. Evaluate performance
score = cc.evaluate(agent, env, episodes=3)
print(f"Agent Accuracy: {score * 100:.1f}%")
```

---

## 🎯 What problem it solves

Building an AI agent is easy. **Fixing it when it fails is hard.**

When your agent makes a mistake (e.g., misclassifying a prompt, generating bad code), you usually have to dig through logs, rewrite the prompt, and pray it doesn't break something else.

**CogniCore provides an environment that gives your agent:**
1. **Memory:** Past mistakes are injected directly into the agent's context.
2. **Feedback:** The environment explains *why* the agent failed.
3. **Structured Rewards:** A detailed 8-component reward signal (streak penalty, novelty bonus, etc.) instead of a generic pass/fail float.

---

## 📊 Benchmark Results

Agents plugged into CogniCore's cognitive middleware (Memory + Reflection) show massive improvements over baseline agents running in standard environments.

| Agent Type | Standard Environment | **CogniCore Environment** | Improvement |
|------------|----------------------|---------------------------|-------------|
| Random | 33.0% | **33.0%** | +0% |
| Rule-Based | 65.0% | **82.0%** | **+17%** |
| LLM (GPT-4) | 78.5% | **94.2%** | **+15.7%** |

*(Run `python benchmarks/run_benchmarks.py` to reproduce locally).*

---

## 🧠 Features

- **24 Built-In Environments:** Safety Classification, Math Reasoning, Code Debugging, Summarization, and more.
- **Cognitive Middleware:** Automatic memory storage/retrieval, reflection hints, and causal analysis.
- **PROPOSE → Revise Pipeline:** Let agents explore tentatively before committing to an action.
- **Gymnasium-Compatible:** Drop-in replacement for OpenAI Gym / Gymnasium.
- **Zero Dependencies:** Pure Python standard library (core framework).

---

## 🔌 API Consistency

CogniCore follows a dead-simple, highly consistent API:

- `cc.make("Env-v1")` → Create an environment
- `cc.train(agent, env)` → Train an agent
- `cc.evaluate(agent, env)` → Evaluate an agent
- `cc.debug(agent, env)` → Debug an agent failure (coming soon)

---

## 📦 Installation & Packaging

CogniCore is built for industry-grade reliability:

```bash
# Basic install
pip install cognicore-env

# Install with development tools (Pytest, Ruff, Mypy, Bandit)
pip install cognicore-env[dev]
```

## 🧑🤝🧑 Open Source Readiness

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on how to get started.
