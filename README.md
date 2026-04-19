<p align="center">
  <img src="https://img.shields.io/badge/CogniCore-AI%20Operating%20System-C4703D?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyek0xMiAyMGMtNC40MiAwLTgtMy41OC04LThzMy41OC04IDgtOCA4IDMuNTggOCA4LTMuNTggOC04IDh6Ii8+PC9zdmc+" alt="CogniCore"/>
</p>

<h1 align="center">🧠 CogniCore</h1>

<p align="center">
  <strong>The AI Operating System — Memory, Reflection, and Structured Rewards for Every Agent</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/cognicore-env/"><img src="https://img.shields.io/pypi/v/cognicore-env?color=C4703D&label=PyPI" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/tests-277%20passing-3D8C6C" alt="Tests"/>
  <img src="https://img.shields.io/badge/exports-71-C4703D" alt="Exports"/>
  <img src="https://img.shields.io/badge/environments-24-3D6EC4" alt="Environments"/>
  <img src="https://img.shields.io/badge/CLI-22%20commands-6B6560" alt="CLI"/>
  <img src="https://img.shields.io/badge/dependencies-0%20(stdlib)-3D8C6C" alt="Zero deps"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License"/>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-why-cognicore">Why CogniCore</a> •
  <a href="#-environments">Environments</a> •
  <a href="#-features">Features</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-api">API</a> •
  <a href="#-gemini-integration">Gemini</a>
</p>

---

## ⚡ Quick Start

```bash
pip install cognicore-env
```

```python
import cognicore

env = cognicore.make("SafetyClassification-v1", difficulty="easy")
obs = env.reset()

while True:
    action = {"classification": "SAFE"}  # your agent decides
    obs, reward, done, _, info = env.step(action)

    print(f"Reward: {reward.total:.2f}")        # 8-component structured reward
    print(f"Memory: {reward.memory_bonus}")      # bonus for remembering past cases
    print(f"Streak: {reward.streak_penalty}")    # penalty for consecutive errors

    if done:
        break

stats = env.episode_stats()
print(f"Accuracy: {stats.accuracy:.0%} | Memory entries: {stats.memory_entries_created}")
```

---

## 🤔 Why CogniCore

> **"Give developers capabilities they don't want to build themselves."**

| Capability | Gymnasium | CogniCore |
|-----------|-----------|-----------|
| Memory across episodes | ❌ | ✅ Agents learn from past mistakes |
| Reflection hints | ❌ | ✅ Environment tells agents *what* went wrong |
| Structured rewards | Single float | **8-component** signal (memory, streak, novelty...) |
| PROPOSE → Revise | ❌ | ✅ Explore before committing |
| Agent types | RL only | **LLM, RL, classifier, rule-based** |
| Swarm intelligence | ❌ | ✅ Multi-agent shared memory |
| Self-evolving rewards | ❌ | ✅ Meta-learning optimizes reward weights |
| Causal reasoning | ❌ | ✅ What-if counterfactual analysis |
| Lifelong learning | ❌ | ✅ Persistent agent identity across sessions |

---

## 🌍 Environments

**24 environments** across 6 domains, each with Easy / Medium / Hard difficulty:

| Domain | Description | Example Action |
|--------|-------------|----------------|
| 🛡️ **Safety Classification** | Classify AI responses as SAFE/UNSAFE/NEEDS_REVIEW | `{"classification": "UNSAFE"}` |
| 🔢 **Math Reasoning** | Arithmetic → number theory | `{"answer": 42}` |
| 🐛 **Code Debugging** | Find and fix Python bugs | `{"bug_line": 4, "fix_type": "off_by_one"}` |
| 💬 **Conversation** | Dialogue and negotiation | `{"response": "empathetic_action"}` |
| 📋 **Multi-Step Planning** | Task ordering and scheduling | `{"order": ["A", "B", "C"]}` |
| 📝 **Summarization** | Key-point coverage | `{"summary": "..."}` |

```python
# All environments work the same way
for env_id in cognicore.list_envs():
    env = cognicore.make(env_id)
    obs = env.reset()
    # ... your agent here
```

---

## 🎯 The 8-Component Structured Reward

Every `env.step()` returns a `StructuredReward` — not just a float:

```
StructuredReward(
    base_score       = 1.0     # Environment grader
    memory_bonus     = +0.05   # Recognized past category correctly
    reflection_bonus = +0.03   # Followed the reflection hint
    streak_penalty   = -0.10   # 2+ consecutive wrong answers
    propose_bonus    = +0.06   # Improved via PROPOSE → Revise
    novelty_bonus    = +0.04   # First time handling this category
    confidence_cal   = +0.02   # Well-calibrated confidence
    time_decay       = -0.01   # Speed penalty
    ──────────────────────────
    total            = 1.09    # Sum of all components
)
```

---

## 🔥 Features

### Phase 1-3: Core Framework
```python
# Memory — agents remember past cases
obs["memory_context"]  # similar past experiences injected

# Reflection — environment explains failures
obs["reflection_hints"]  # "you got 'malware' wrong last time"

# PROPOSE → Revise — explore before committing
feedback = env.propose({"classification": "UNSAFE"})
obs, reward, done, _, info = env.revise({"classification": "SAFE"})
```

### Phase 4: Benchmark & Analytics
```python
from cognicore import benchmark_agent, CurriculumRunner

# Benchmark across all 24 environments
result = benchmark_agent(my_agent, episodes=3)
print(result.overall_score)

# Adaptive difficulty progression
runner = CurriculumRunner("SafetyClassification-v1", my_agent)
runner.run(max_episodes=20)  # auto-increases difficulty
```

### Phase 5: Smart Agents & Safety
```python
from cognicore import AutoLearner, SafeAgent, AdaptiveAgent

agent = AutoLearner()        # learns from memory + reflection
agent = SafeAgent()          # conservative, flags uncertain cases
agent = AdaptiveAgent()      # switches strategy based on performance
```

### Phase 6: Research-Grade
```python
from cognicore import CognitiveMemory, IntelligenceScorer, EvolutionEngine

# 4-tier cognitive memory (working, episodic, semantic, procedural)
mem = CognitiveMemory()
mem.perceive("phishing email detected", "security", True, "UNSAFE")

# Evolution — population-based agent training
engine = EvolutionEngine("SafetyClassification-v1", pop_size=20)
best = engine.evolve(generations=10)

# Red vs Blue — adversarial simulation
from cognicore import RedVsBlue
battle = RedVsBlue()
report = battle.simulate(rounds=50)
```

### Phase 7: Platform & DevTools
```python
from cognicore import save_agent, load_agent, get_profile, ResponseCache

# Agent persistence
save_agent(my_agent, "agent_v1.json")
agent = load_agent("agent_v1.json")

# 7 config presets
config = get_profile("strict_safety")  # or: fast_explore, production, research...

# Response caching (save tokens!)
cache = ResponseCache(max_size=1000, ttl=3600)
```

### Phase 8: AI Operating System
```python
from cognicore import (
    MetaRewardOptimizer, CausalEngine, build_agent,
    StrategySwitcher, LifelongAgent, Swarm
)

# Self-evolving rewards — the system learns how to reward better
meta = MetaRewardOptimizer()
meta.observe(reward_components, accuracy_improved=True)
new_weights = meta.optimize()

# Causal reasoning — learn cause→effect, not just correlations
engine = CausalEngine()
engine.observe("phishing", action="UNSAFE", outcome="correct")
result = engine.what_if("phishing", "SAFE")  # prediction: wrong!

# Autonomous agent builder — describe goal, get agent
agent = build_agent(goal="maximize safety", risk_tolerance="low")

# Real-time strategy switching
switcher = StrategySwitcher()
mode = switcher.decide(accuracy=0.2)  # → "explore"
mode = switcher.decide(accuracy=0.9)  # → "safe"

# Lifelong learning — persistent identity across sessions
agent = LifelongAgent("agent-001")
agent.run_session("SafetyClassification-v1", episodes=10)
agent.run_session("CodeDebugging-v1", episodes=10)  # knowledge carries over!
agent.save()

# Swarm intelligence — multi-agent collaboration
swarm = Swarm(size=5, diversity=True)
result = swarm.solve("SafetyClassification-v1", episodes=3)
result.print_report()
```

---

## 💻 CLI

**22 commands** covering the entire lifecycle:

```bash
# Core
cognicore list                                    # List 24 environments
cognicore run SafetyClassification-v1 -v          # Run with verbose output
cognicore info MathReasoning-v1                   # Environment details
cognicore benchmark --episodes 3                  # Benchmark all envs

# Training
cognicore curriculum SafetyClassification-v1      # Adaptive difficulty
cognicore evolve SafetyClassification-v1          # Evolutionary training
cognicore improve SafetyClassification-v1         # Self-improvement loop

# Analysis
cognicore explain SafetyClassification-v1         # Explainable AI report
cognicore iq SafetyClassification-v1              # 6-dimension IQ test
cognicore debug SafetyClassification-v1           # AI debugger
cognicore cost --model gemini-flash               # Cost estimation

# Advanced
cognicore battle --rounds 50                      # Red vs Blue simulation
cognicore stress SafetyClassification-v1          # Adversarial stress test
cognicore swarm --size 5                          # Multi-agent swarm
cognicore build --goal "maximize safety"          # Auto-build agent
cognicore lifelong agent-001                      # Persistent agent
cognicore transfer                                # Knowledge distillation
cognicore report --episodes 3                     # HTML report

# Infrastructure
cognicore serve --port 8000                       # REST API server
cognicore dashboard                               # Web dashboard
cognicore doctor                                  # 55 health checks
```

---

## 🔌 API

```bash
cognicore serve --port 8000
```

```bash
# Create session
curl -X POST http://localhost:8000/envs/SafetyClassification-v1/create \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Step
curl -X POST http://localhost:8000/sessions/{sid}/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"classification": "UNSAFE"}}'
```

Interactive docs at `http://localhost:8000/docs`

---

## 🤖 Gemini Integration

```python
import google.generativeai as genai
import cognicore

genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

env = cognicore.make("SafetyClassification-v1")
obs = env.reset()

while True:
    # CogniCore injects memory + reflection into the prompt
    prompt = f"""Classify as SAFE/UNSAFE/NEEDS_REVIEW:
    {obs['prompt']}
    Memory: {obs.get('memory_context', [])}
    Hint: {obs.get('reflection_hints', '')}"""

    response = model.generate_content(prompt)
    obs, reward, done, _, info = env.step({"classification": response.text.strip()})
    if done: break
```

---

## 🏗️ Build Your Own Environment

```python
from cognicore import CogniCoreEnv, EvalResult

class MyEnv(CogniCoreEnv):
    def _setup(self, **kwargs):
        pass  # define spaces, load data

    def _generate_tasks(self):
        return [{"q": "2+2", "a": 4, "category": "math"}]

    def _evaluate(self, action):
        task = self._tasks[self._current_step]
        correct = action.get("answer") == task["a"]
        return EvalResult(base_score=1.0 if correct else 0.0, correct=correct, category=task["category"])

    def _get_obs(self):
        return {"question": self._tasks[self._current_step]["q"]}

# Memory, reflection, rewards all work automatically!
```

---

## 📊 Architecture

```
cognicore/
├── core/               # Base env, types, spaces
├── middleware/          # Memory, Reflection, Rewards, Propose-Revise, Safety Monitor
├── envs/               # 6 domains × 4 difficulty variants = 24 environments
├── agents/             # BaseAgent + RandomAgent
├── server/             # FastAPI REST API (12 endpoints)
├── cli.py              # 22 CLI commands
│
├── smart_agents.py     # AutoLearner, SafeAgent, AdaptiveAgent
├── evolution.py        # Population-based training
├── intelligence.py     # 6-dimension IQ scoring
├── red_blue.py         # Adversarial Red vs Blue
├── causal.py           # Cause→effect reasoning
├── meta_rewards.py     # Self-evolving reward optimization
├── strategy.py         # Real-time mode switching
├── swarm.py            # Multi-agent shared memory
├── lifelong.py         # Persistent agent identity
├── agent_builder.py    # Autonomous agent configuration
└── ... 30+ more modules
```

---

## 📈 Project Stats

| Metric | Value |
|--------|-------|
| Tests | **277 passing** (1.2s) |
| Exports | **71** public API symbols |
| Environments | **24** (6 domains × 4 difficulties) |
| CLI Commands | **22** |
| Doctor Checks | **55/55** |
| Dependencies | **0** (pure stdlib core) |
| Hand-crafted cases | **180** |
| Examples | **11** |

---

## 📦 Installation Options

```bash
# Core (zero dependencies)
pip install cognicore-env

# With API server (FastAPI + Uvicorn)
pip install cognicore-env[server]

# With LLM support (OpenAI client)
pip install cognicore-env[llm]

# Everything
pip install cognicore-env[all]
```

---

## 🗺️ Roadmap

- [x] 8-component structured rewards
- [x] 24 environments (6 domains)
- [x] Memory + Reflection middleware
- [x] PROPOSE → Revise protocol
- [x] CLI with 22 commands
- [x] REST API + Dashboard
- [x] Smart agents (AutoLearner, SafeAgent, AdaptiveAgent)
- [x] Evolutionary training
- [x] Red vs Blue adversarial
- [x] Swarm intelligence
- [x] Causal reasoning
- [x] Self-evolving rewards
- [x] Lifelong learning
- [x] Gemini API integration
- [ ] Hosted cloud platform
- [ ] Plugin ecosystem
- [ ] Docker deployment
- [ ] Database backend

---

## 📄 License

MIT — use it however you want.

---

<p align="center">
  Built with 🧠 by <a href="https://github.com/Kaushalt2004">Kaushalt2004</a>
</p>
