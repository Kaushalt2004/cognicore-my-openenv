# CogniCore

**Cognitive Environments for AI** — Memory, Reflection, and Structured Rewards built into every environment.

CogniCore is a Python framework where every environment comes with built-in cognitive infrastructure that no other framework provides:

| Feature | Gymnasium | CogniCore |
|---------|-----------|-----------|
| Memory across episodes | No | **Yes** — agents learn from past mistakes |
| Reflection hints | No | **Yes** — environment tells agents what they're doing wrong |
| Structured rewards | Single float | **8-component** reward signal |
| PROPOSE → Revise | No | **Yes** — explore before committing |
| Works with any AI type | RL only | **LLM, RL, classifier, rule-based** |

## Installation

```bash
# Core framework (zero dependencies)
pip install cognicore

# With LLM support
pip install cognicore[llm]

# With API server
pip install cognicore[server]

# Everything
pip install cognicore[all]
```

## Quick Start

```python
import cognicore

# Create an environment
env = cognicore.make("SafetyClassification-v1", difficulty="easy")
obs = env.reset()

while True:
    action = {"classification": "SAFE"}  # your agent here
    obs, reward, done, truncated, info = env.step(action)

    # 8-component structured reward
    print(f"Total: {reward.total:.2f}")
    print(f"  Base: {reward.base_score}")
    print(f"  Memory bonus: {reward.memory_bonus}")
    print(f"  Streak penalty: {reward.streak_penalty}")

    if done:
        break

print(env.episode_stats())
```

## Environments

CogniCore ships with **5 environment domains** (20 registered IDs):

### Safety Classification
Classify AI responses as SAFE / UNSAFE / NEEDS_REVIEW.
```python
env = cognicore.make("SafetyClassification-v1", difficulty="hard")
obs, reward, done, _, info = env.step({"classification": "UNSAFE"})
```

### Math Reasoning
Solve arithmetic, algebra, and advanced math problems.
```python
env = cognicore.make("MathReasoning-v1", difficulty="medium")
obs, reward, done, _, info = env.step({"answer": 42})
```

### Code Debugging
Find and fix bugs in Python code snippets.
```python
env = cognicore.make("CodeDebugging-v1", difficulty="hard")
obs, reward, done, _, info = env.step({"bug_line": 4, "fix_type": "security_vulnerability"})
```

### Conversation / Negotiation
Choose the best response in dialogue scenarios.
```python
env = cognicore.make("Conversation-v1", difficulty="medium")
obs, reward, done, _, info = env.step({"response": "empathetic_action"})
```

### Multi-Step Planning
Order steps correctly to solve planning problems.
```python
env = cognicore.make("Planning-v1", difficulty="hard")
obs, reward, done, _, info = env.step({"order": ["A", "B", "C", "D", "E"]})
```

## The 8-Component Structured Reward

Every `step()` returns a `StructuredReward` — not just a float:

```
StructuredReward(
    base_score       = 1.0    # From environment grader
    memory_bonus     = 0.05   # Consistency with past successes
    reflection_bonus = 0.03   # Followed a reflection hint
    streak_penalty   = 0.00   # Penalty for consecutive failures
    propose_bonus    = 0.05   # Improved via PROPOSE → Revise
    novelty_bonus    = 0.04   # Correctly handled new category
    confidence_cal   = 0.02   # Well-calibrated confidence
    time_decay       = -0.01  # Speed penalty
    ─────────────────────────
    total            = 1.18   # Sum of all components
)
```

## PROPOSE → Revise Protocol

Agents can explore before committing:

```python
# 1. Propose (no grading)
feedback = env.propose({"classification": "UNSAFE"})
print(feedback.reflection_hint)     # "This category was often SAFE"
print(feedback.confidence_estimate) # 0.34

# 2. Revise (graded)
obs, reward, done, _, info = env.revise({"classification": "SAFE"})
# If improved → propose_bonus in reward
```

## Build Your Own Environment

Subclass `CogniCoreEnv` and implement 4 methods:

```python
from cognicore import CogniCoreEnv, EvalResult

class MyEnv(CogniCoreEnv):
    def _setup(self, **kwargs):
        pass  # Define spaces, load data

    def _generate_tasks(self):
        return [{"q": "2+2", "a": 4, "category": "math"}]

    def _evaluate(self, action):
        task = self._tasks[self._current_step]
        correct = action.get("answer") == task["a"]
        return EvalResult(
            base_score=1.0 if correct else 0.0,
            correct=correct,
            category=task["category"],
        )

    def _get_obs(self):
        return {"question": self._tasks[self._current_step]["q"]}

# That's it! Memory, reflection, rewards all work automatically.
```

## CLI

```bash
# List environments
cognicore list

# Run with a random agent
cognicore run SafetyClassification-v1 --difficulty hard --episodes 3 -v

# Show environment info
cognicore info MathReasoning-v1

# Start API server
cognicore serve --port 8000
```

## REST API

```bash
# Start server
cognicore serve

# Create session
curl -X POST http://localhost:8000/envs/SafetyClassification-v1/create \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Reset
curl -X POST http://localhost:8000/sessions/{sid}/reset

# Step
curl -X POST http://localhost:8000/sessions/{sid}/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"classification": "SAFE"}}'
```

Interactive docs at `http://localhost:8000/docs`.

## Architecture

```
cognicore/
├── core/           # Base env, types, spaces
├── middleware/      # Memory, Reflection, Rewards, Propose-Revise, Safety Monitor
├── envs/           # 5 built-in environments + registry
├── agents/         # BaseAgent ABC + RandomAgent
├── server/         # FastAPI REST API
├── cli.py          # Command-line interface
└── utils/          # Logging utilities
```

## License

MIT
