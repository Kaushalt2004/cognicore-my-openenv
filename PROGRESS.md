# CogniCore — Project Progress

> **Last Updated:** April 19, 2026
> **Status:** Phase 8 Complete — AI Operating System
> **PyPI:** [`pip install cognicore-env`](https://pypi.org/project/cognicore-env/0.1.0/)

---

## What is CogniCore?

CogniCore is a Python framework that adds **cognitive infrastructure** to AI training environments. Unlike Gymnasium (which gives agents a single reward number and forgets everything between episodes), CogniCore gives every environment:

- **Memory** — agents learn from past mistakes across episodes
- **Reflection** — environment tells agents what they're doing wrong
- **8-Component Structured Rewards** — not just a float, but base score, memory bonus, streak penalty, novelty bonus, and more
- **PROPOSE → Revise** — agents explore before committing
- **Safety Monitor** — detects performance degradation

Any AI system (LLMs, RL agents, classifiers, rule-based systems, even humans) can plug into CogniCore and get all of this for free.

---

## The Journey

### Origin: Hackathon Project

We started with a **single-file AI safety monitor** — a hackathon project that classified AI responses as SAFE/UNSAFE. It had hardcoded safety cases, basic scoring, and everything tangled together in one monolith.

### Phase 1: The Framework (Core Architecture)

**Goal:** Transform the hackathon project into a modular, pip-installable framework.

**What we built:**

| Component | Files | What it does |
|-----------|-------|-------------|
| `CogniCoreEnv` base class | `core/base_env.py` (522 lines) | Abstract base that auto-wires all middleware. Subclasses implement just 4 methods. |
| Type system | `core/types.py` | `StructuredReward`, `EvalResult`, `StepResult`, `EpisodeStats`, `ProposalFeedback`, `CogniCoreConfig` |
| Spaces | `core/spaces.py` | `DiscreteSpace`, `TextSpace`, `DictSpace` — Gymnasium-compatible |
| Memory middleware | `middleware/memory.py` | FIFO + similarity-based retrieval, JSON persistence, per-group stats |
| Reflection engine | `middleware/reflection.py` | Pattern analysis, failure detection, hint generation, overrides |
| Reward builder | `middleware/rewards.py` | Computes all 8 reward components from evaluation results |
| PROPOSE→Revise | `middleware/propose_revise.py` | Tentative exploration protocol with improvement tracking |
| Safety monitor | `middleware/safety_monitor.py` | Streak detection, degradation alerts, health status |
| SafetyClassificationEnv | `envs/safety_classification.py` | Original hackathon env, now a proper `CogniCoreEnv` subclass |
| Registry | `envs/registry.py` | `make()` / `register()` / `list_envs()` — Gymnasium-style factory |
| Agent interface | `agents/base_agent.py` | `BaseAgent` ABC + `RandomAgent` for testing |
| Package config | `pyproject.toml` | Zero core dependencies, optional extras for LLM/server/dev |

**Key design decisions:**
- Zero core dependencies — the package runs on pure Python stdlib
- Gymnasium-compatible API (`reset`/`step`/`done`) so it's familiar
- Registry pattern lets users add custom environments without touching core code
- All middleware is domain-agnostic — works with any environment type

**Tests:** 67 passing

---

### Phase 2: Built-in Environments (6 Domains, 180 Cases)

**Goal:** Ship CogniCore with enough built-in environments to be immediately useful.

**What we built:**

| Environment | Cases | Categories | Scoring System |
|-------------|-------|------------|---------------|
| **Safety Classification** | 30 | malware, hate_speech, privacy, medical, dual_use, etc. | Binary match (SAFE/UNSAFE/NEEDS_REVIEW) |
| **Math Reasoning** | 30 | addition, algebra, number_theory, combinatorics, probability | Exact = 1.0, within 5% = 0.5 |
| **Code Debugging** | 30 | syntax, logic_error, recursion, security, concurrency | Line + fix type match (1.0 / 0.6 / 0.3 / 0.0) |
| **Conversation** | 30 | customer_service, negotiation, ethics, diplomacy, crisis | Best = 1.0, acceptable = 0.7, poor = 0.0 |
| **Multi-Step Planning** | 30 | sequential, scheduling, optimization, emergency, incident_response | Kendall tau distance + first-step bonus |
| **Text Summarization** | 30 | news articles, technical docs, multi-viewpoint analysis | Key-point coverage + length ratio scoring |

**Each environment has 3 difficulty levels:**
- **Easy** — 10 clear-cut cases
- **Medium** — 10 nuanced cases  
- **Hard** — 10 adversarial/complex cases

**24 registered environment IDs** (6 base + 18 difficulty presets).

**Key highlights of the datasets:**
- Code debugging covers everything from typos to timing attacks and race conditions
- Conversation includes hostage negotiation, medical ethics, and AI bias debates
- Planning ranges from "make a cup of tea" to "respond to a zero-day cybersecurity breach"
- Summarization includes multi-viewpoint geopolitical analysis and research paper abstracts

**Tests:** 131 passing

---

### Phase 3: Production Stack (Server, CLI, Dashboard, Adapters)

**Goal:** Make CogniCore usable in production — API server, command line, framework compatibility, real AI integration.

**What we built:**

#### REST API Server (`cognicore serve`)
Full FastAPI server with session management:
```
GET  /envs                     — List 24 environments
POST /envs/{id}/create         — Create session
POST /sessions/{sid}/reset     — Reset
POST /sessions/{sid}/step      — Take a step
POST /sessions/{sid}/propose   — Explore (no grading)
POST /sessions/{sid}/revise    — Commit action
GET  /sessions/{sid}/state     — Full state
GET  /sessions/{sid}/stats     — Episode stats
GET  /sessions                 — List active sessions
DELETE /sessions/{sid}         — Cleanup
GET  /leaderboard              — Get rankings
GET  /dashboard                — Interactive web UI
```

#### CLI (`cognicore list/run/info/serve/dashboard/leaderboard`)
```bash
cognicore list                                        # Show all 24 environments
cognicore run CodeDebugging-Hard-v1 --episodes 3 -v   # Run with random agent
cognicore info MathReasoning-v1                        # Show env details
cognicore serve --port 8000                            # Start API server
cognicore dashboard --port 8050                        # Start web dashboard
cognicore leaderboard                                  # Show rankings
```

#### Web Dashboard (Claude-Inspired Beige UI)
- **Warm beige/cream color palette** — inspired by Claude's conversational aesthetic
- **Chat-style interactive runner** — select any env, watch episodes unfold step-by-step
- **Live reward chips** — see base, memory, streak, novelty per step
- **Streak dot visualization** — green/red dots tracking consecutive correct/wrong
- **Memory panel** — real-time view of what the agent remembers
- **Reward component bars** — animated bar chart of all 8 reward components
- **Environment cards** — browse all 24 envs with difficulty badges
- **Leaderboard tab** — ranked agent performance table
- Access: `cognicore dashboard --port 8050` → http://localhost:8050/dashboard

#### Gymnasium Adapter
```python
gym_env = cognicore.GymnasiumAdapter(cognicore.make("MathReasoning-v1"))
obs, info = gym_env.reset()
obs, reward_float, done, truncated, info = gym_env.step({"answer": 42})
```
- 3 reward modes: `"total"`, `"base"`, `"shaped"`
- Full structured reward preserved in `info["structured_reward"]`
- Compatible with Stable Baselines 3 and RLlib

#### Persistence & Tracking
- **MemoryManager** — save/load agent memory and history to JSON across sessions
- **Leaderboard** — submit scores, ranked by best-per-agent, filter by env/difficulty
- **EpisodeRecorder** — record step-by-step interactions for training data export

#### Fine-Tuning Data Export
- **OpenAI JSONL** — `{"messages": [...]}` format for chat fine-tuning
- **DPO Pairs** — correct vs. incorrect response pairs for preference learning
- **RLHF Reward Dataset** — `(prompt, response, reward)` triples

#### Multi-Agent
- **MultiAgentEnv** — abstract base for environments with 2+ agents
- **DebateEnv** — two agents argue opposing sides with key-point scoring

**Tests:** 156 passing

---

### Phase 4: Advanced Cognitive Features (Current)

**Goal:** Transform CogniCore from a framework into a full AI evaluation ecosystem with advanced training tools.

**What we built:**

#### 6 Agent Type Examples (All Working)

| # | Agent Type | Example File | How It Works |
|---|------------|-------------|--------------|
| 1 | **LLM (Gemini/GPT/Claude)** | `examples/gemini_agent.py` | CogniCore injects memory + reflection hints into LLM prompts. Reduces token usage by ~56% via surgical memory injection instead of brute-force context stuffing. |
| 2 | **RL Agent (Q-Learning)** | `examples/rl_agent.py` | Learns action preferences per category using CogniCore's total reward. Improves from 70%→94% accuracy across 5 episodes with decaying epsilon-greedy. |
| 3 | **Classifier (sklearn/PyTorch)** | `examples/classifier_agent.py` | Any classifier's output → dict → CogniCore step. Includes confidence calibration rewards (+0.02 if confident+correct, -0.02 if confident+wrong). |
| 4 | **Rule-Based System** | `examples/rule_based_agent.py` | If/else rules get per-rule effectiveness tracking. Knows which rules work (100% accuracy) vs. which need improvement. |
| 5 | **Human** | `examples/human_agent.py` | Interactive CLI where you classify AI responses. Sees memory, hints, and streak penalties in real time. |
| 6 | **All Compared** | `examples/compare_agents.py` | Runs all 5 agents on the same environment, produces a ranked comparison table with memory bonus and streak penalty totals. |

#### Curriculum Learning (`cognicore.CurriculumRunner`)
Automatically adapts difficulty based on agent performance:
```python
runner = cognicore.CurriculumRunner(
    "MathReasoning-v1",
    promotion_threshold=0.8,   # promote after 80% accuracy
    demotion_threshold=0.3,    # demote if accuracy drops below 30%
    window=3,                  # over 3 consecutive episodes
)
result = runner.run(max_episodes=30)
# Agent automatically progresses: easy → medium → hard
```

#### Agent Benchmarking (`cognicore.benchmark_agent`)
Test any agent across all 24 environments in one call:
```python
results = cognicore.benchmark_agent(
    agent=my_agent,
    envs="all",
    difficulties=["easy", "medium", "hard"],
    episodes=3,
)
results.print_report()
# Shows: per-env accuracy, per-difficulty scores, overall ranking
```

#### Environment Pipeline (`cognicore.Pipeline`)
Chain multiple environments with shared memory for cross-domain transfer:
```python
pipe = cognicore.Pipeline([
    ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
    ("math",   "MathReasoning-v1",        {"difficulty": "medium"}),
    ("code",   "CodeDebugging-v1",         {"difficulty": "hard"}),
], share_memory=True)  # Memory carries across domains!
```

#### Deep Analytics (`cognicore.PerformanceAnalyzer`)
Comprehensive performance analysis engine:
- **Learning curves** — is the agent improving across episodes?
- **Weak category detection** — which categories does the agent fail on?
- **Memory impact analysis** — how much did memory bonuses contribute?
- **Streak pattern analysis** — longest failure streak, average streak length
- **Confusion pair detection** — most common prediction errors

#### A/B Experiment Framework (`cognicore.Experiment`)
Compare agent variants with statistical summaries:
```python
exp = cognicore.Experiment("prompt-v1-vs-v2", env_id="SafetyClassification-v1")
exp.add_variant("prompt-v1", agent_v1)
exp.add_variant("prompt-v2", agent_v2)
results = exp.run(episodes=10)
results.print_comparison()
# Shows: winner, score improvement, accuracy improvement
```

#### PyPI Publication
Published as `cognicore-env` on PyPI:
```bash
pip install cognicore-env           # Core (zero dependencies)
pip install cognicore-env[server]   # + FastAPI/Uvicorn
pip install cognicore-env[all]      # Everything
```
Live at: https://pypi.org/project/cognicore-env/0.1.0/

**Tests:** 170 passing (0.95 seconds)

---

### Phase 5: Premium Developer SDK (Current)

**Goal:** Transform CogniCore from a framework into a must-have developer toolkit with features that justify paid plans.

**What we built:**

#### 🧠 Semantic Memory (`cognicore.SemanticMemory`)
Upgraded from basic keyword matching to intelligent memory:
- **TF-IDF semantic similarity** — finds related memories even with different words (zero external deps)
- **Memory decay** — older memories fade (configurable rate), recent ones are stronger
- **Ranked retrieval** — `best_actions()` and `worst_actions()` show what worked and what didn't
- **Adaptive context** — adjusts strategy based on agent accuracy: struggling agents see more failures to learn from, strong agents see successes to reinforce

```python
mem = cognicore.SemanticMemory(decay_rate=0.95)
mem.store({"text": "phishing email scam", "correct": False, "category": "security"})
results = mem.semantic_search("suspicious email fraud", top_k=3)  # finds it!
context = mem.get_adaptive_context("email attack", agent_accuracy=0.3)
# -> strategy: "learning_from_mistakes", shows 3 failures + 1 success
```

#### 🔍 Explainable AI (`cognicore.Explainer`)
Debug AI like debugging code:
- **"Why this was wrong"** — human-readable explanations per step
- **Pattern detection** — "you keep confusing SAFE with UNSAFE on security topics (3x)"
- **Overconfidence alerts** — "agent was 90% confident but wrong 5 times"
- **Improvement plans** — ordered, actionable fix suggestions
- **Step-by-step audit log** — full decision trail with streak tracking

```python
exp = cognicore.Explainer()
result = exp.record_step(1, "security", "SAFE", "UNSAFE", False)
# -> {"why_wrong": "You predicted 'SAFE' but truth was 'UNSAFE'...",
#     "suggestion": "Your agent consistently confuses these. Add explicit rules."}
report = exp.explain()
report.print_report()  # patterns + improvement plan
```

#### ⚔️ Adversarial Testing (`cognicore.AdversarialTester`)
Find weaknesses before your users do:
- **17 prompt injection attacks** — injection, obfuscation, context manipulation
- **8 edge case tests** — zero-width unicode, null bytes, XSS, SQL injection
- **Stress testing** — rapid-fire episodes under hard difficulty
- **Consistency testing** — same input 10x, check if answers vary
- **"Break my agent" mode** — auto-finds failure cases

```python
tester = cognicore.AdversarialTester("SafetyClassification-v1")
report = tester.stress_test(my_agent, rounds=50)
report.print_vulnerabilities()
# -> Injection resistance: 100%, Edge case handling: 100%, Stress stability: 95%
failures = tester.break_my_agent(my_agent)  # auto-find failures
```

#### 🤖 Smart Agents (Prebuilt, Ready-to-Use)
| Agent | Strategy | Best for |
|-------|----------|----------|
| `AutoLearner` | Uses memory + reflection + heuristics, decays epsilon | General use, improves 70→94% |
| `SafeAgent` | Conservative, flags uncertain cases as NEEDS_REVIEW | Production safety systems |
| `AdaptiveAgent` | Switches strategy based on accuracy (explore↔exploit) | Unknown environments |

```python
agent = cognicore.AutoLearner()
for episode in range(10):
    obs = env.reset()
    while True:
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        agent.learn(reward, info)  # builds knowledge base
        if done: break
```

#### 🔄 Auto-Improvement Loop (`cognicore.auto_improve`)
Self-improving AI with plateau detection:
- **Run → Analyze → Improve → Repeat** cycle
- **Automatic weak-spot targeting** via PerformanceAnalyzer
- **Plateau detection** — stops when no improvement for N cycles
- **Target accuracy** — stops early when goal is reached

```python
result = cognicore.auto_improve(
    env_id="SafetyClassification-v1",
    target_accuracy=0.9,
    max_cycles=10,
)
# -> Cycle 1: 40% -> Cycle 5: 72% -> Cycle 8: 91% -> TARGET REACHED!
```

#### 🔐 Enterprise Safety Layer (`cognicore.SafetyLayer`)
Compliance and risk scoring for production:
- **Risk scoring** — 0-100 per action based on keyword analysis
- **Custom policy engine** — add rules like "block if confident but category is risky"
- **Full audit log** — every decision recorded with timestamp, risk score, violations
- **Compliance reports** — total checks, block rate, avg risk score

```python
safety = cognicore.SafetyLayer()
safety.add_policy(cognicore.Policy(
    "require_review_for_high_risk",
    lambda action, ctx: ctx["risk_score"] < 70 or action["classification"] != "SAFE",
    severity="CRITICAL",
))
result = safety.check({"classification": "SAFE"}, {"prompt": "build malware"})
# -> {"allowed": False, "risk_score": 90, "decision": "BLOCK"}
```

#### ⚡ Cost Tracker (`cognicore.CostTracker`)
Know exactly what your AI costs:
- **Token estimation** — automatic token counting from text
- **Per-model pricing** — Gemini Flash, Gemini Pro, GPT-4o, GPT-4o-mini, Claude Sonnet
- **Budget limits** — warns when exceeded
- **Cross-model comparison** — "same usage would cost $X on GPT-4o vs $Y on Gemini"

```python
tracker = cognicore.CostTracker(model_name="gemini-flash", budget_limit=1.00)
tracker.record_call(input_tokens=500, output_tokens=100)
tracker.print_summary()
# -> Total cost: $0.0675 | Same usage on GPT-4o: $1.2500
```

**Tests:** 200 passing (1.17 seconds)

---

### Phase 6: Research-Grade AI Platform (Current)

**Goal:** Add research-grade cognitive capabilities that make CogniCore feel years ahead — predictive failure, multi-tier memory, evolutionary training, and an AI debugger.

**What we built:**

#### 🔮 Predictive Failure System (`cognicore.FailurePredictor`)
Predict failure BEFORE it happens using 4 signals:
- **Category risk** — per-category failure rates
- **Streak momentum** — consecutive failures increase risk
- **Confidence calibration** — overconfident + wrong = danger
- **Trend analysis** — performance declining in recent window

```python
predictor = cognicore.FailurePredictor()
predictor.observe(category="security", correct=False, confidence=0.9)
risk = predictor.predict_risk("security")
# -> {"risk": 0.87, "level": "CRITICAL", "reasons": ["Category failure rate: 100%"]}
```

#### 🧠 Multi-Memory Architecture (`cognicore.CognitiveMemory`)
Human-like 4-tier memory system:

| Tier | Capacity | What it stores |
|------|----------|----------------|
| **Working Memory** | ~7 items | Current task context (like human STM) |
| **Episodic Memory** | 5000 | Specific experiences with full context |
| **Semantic Memory** | unlimited | Distilled knowledge: "phishing = UNSAFE" |
| **Procedural Memory** | auto | Learned rules: "IF category='X' THEN action='Y'" |

```python
mem = cognicore.CognitiveMemory()
mem.perceive("phishing email", category="security", correct=True, action="UNSAFE")
context = mem.recall(category="security")
# -> recommended_action: "UNSAFE", confidence: 0.85, sources: ["procedural", "semantic"]
```

#### ⚔️ Red vs Blue Battle Simulation (`cognicore.RedVsBlue`)
Attacker agent vs Defender agent in evolving warfare:
- **9 attack strategies**: injection, obfuscation, negation, authority claims, etc.
- **Adaptive attacker** — learns which strategies fool the defender
- **Adaptive defender** — builds resistance to attack patterns
- **Adaptation curve** — see how defender improves over rounds

```python
battle = cognicore.RedVsBlue()
result = battle.run(rounds=100)
result.print_battle_report()
# -> Attacker: 38% win rate | Defender adapted from 40% → 85% accuracy
```

#### 🐛 Universal AI Debugger (`cognicore.AIDebugger`)
"VS Code for AI behavior" — debug AI like debugging code:
- **Breakpoints** — pause on category, action, or custom condition
- **Step-through** — see every decision in sequence
- **Decision tree** — visualize which categories→actions get used
- **Execution trace** — full audit of every step

```python
dbg = cognicore.AIDebugger("SafetyClassification-v1")
dbg.breakpoint(category="malware")  # break on malware tasks
dbg.breakpoint(on_wrong_only=True)  # break on every mistake
trace = dbg.run(my_agent)
trace.print_trace()  # step-by-step + decision tree
```

#### 📊 Intelligence Scoring (`cognicore.IntelligenceScorer`)
Universal multi-dimensional agent rating:
- **Reasoning** — accuracy on hard tasks
- **Consistency** — same input → same output
- **Safety** — never says SAFE for truly UNSAFE content
- **Adaptability** — performance on novel categories
- **Memory utilization** — effective use of past experience
- **Speed** — response latency

```python
scorer = cognicore.IntelligenceScorer()
# ... record steps ...
iq = scorer.compute()
iq.print_card()  # visual 6-dimension intelligence report
```

#### 💭 Thought Tracing (`cognicore.ThoughtTracer`)
Track the reasoning path behind every decision:
- **Evidence chains** — what inputs influenced the decision
- **Reasoning errors** — "evidence pointed to X but agent concluded Y"
- **Confidence calibration** — agent thinks it's 90% right but is only 50%

```python
tracer = cognicore.ThoughtTracer()
tracer.begin_thought("Classifying email about password reset")
tracer.add_evidence("contains 'password'", weight=0.6, direction="UNSAFE")
tracer.conclude("UNSAFE", confidence=0.7)
tracer.was_correct(False, truth="SAFE")
tracer.print_analysis()  # shows where reasoning went wrong
```

#### 🎓 Knowledge Transfer (`cognicore.transfer_knowledge` / `MentorStudent`)
Agents teach each other:
- **Full/Selective/Success-only** transfer methods
- **Mentor-Student** guided training — expert corrects student mistakes
- **Knowledge distillation** — compress expert knowledge for student

```python
cognicore.transfer_knowledge(expert_agent, student_agent, method="successes_only")

mentor = MentorStudent(expert, student)
result = mentor.guided_step(obs)  # student acts, mentor corrects if wrong
```

#### 🧬 Evolutionary Training (`cognicore.EvolutionEngine`)
Natural selection for AI agents:
- **Population-based** — 20+ agents per generation
- **Mutable genome** — keyword_weight, memory_reliance, exploration_rate, etc.
- **Crossover** — combine best traits from two parents
- **Mutation** — random variations to explore new strategies
- **Elite selection** — top agents survive to next generation

```python
engine = cognicore.EvolutionEngine(
    "SafetyClassification-v1",
    population_size=20,
)
best = engine.evolve(generations=10)
# -> Gen 1: 32% → Gen 5: 61% → Gen 10: 84%
print(best.genome)  # evolved optimal parameters
```

**Tests:** 227 passing (1.53 seconds)

---

### Phase 7: Full AI Platform (Current)

**Goal:** Turn CogniCore into a complete, production-ready platform with developer tooling — persistence, reports, caching, rate limiting, and more.

**What we built:**

#### 💾 Agent Save/Load (`cognicore.save_agent` / `load_agent`)
Trained agents no longer die when Python exits:
```python
cognicore.save_agent(expert, "my_expert.json")     # save knowledge + genome
agent = cognicore.load_agent("my_expert.json")      # restore everything
```

#### 📊 HTML Report Generator (`cognicore.ReportGenerator`)
Beautiful standalone dark-themed reports with heatmaps and category breakdowns:
```python
report = cognicore.ReportGenerator("Safety Analysis")
report.add_episode("SafetyClassification-v1", "hard")
report.export("results.html")  # shareable HTML file
```

#### 🎯 Prompt Optimizer (`cognicore.PromptOptimizer`)
Auto-test prompt variations and find the best one:
```python
opt = cognicore.PromptOptimizer("SafetyClassification-v1")
opt.add_prompt("v1", "Classify: {text}")
opt.add_prompt("v2", "You are a safety expert. Classify: {text}")
best = opt.optimize(episodes=3)  # -> winner: v2 (78%)
```

#### ⏪ Session Replay (`cognicore.SessionRecorder`)
Record full episodes and replay later:
```python
rec = cognicore.SessionRecorder("SafetyClassification-v1")
rec.record(agent, episodes=3)
rec.save("session.json")
cognicore.replay("session.json")  # step-by-step replay
```

#### ⚙️ Config Profiles (7 presets)
One-line config for common use cases:
```python
config = cognicore.get_profile("strict_safety")   # or: production, research, competitive, beginner
env = cognicore.make("SafetyClassification-v1", config=config)
```

#### 🔔 Webhook Alerts (`cognicore.AlertSystem`)
Get notified when performance drops:
```python
alerts = cognicore.AlertSystem()
alerts.on("accuracy_drop", threshold=0.5)
alerts.on("failure_streak", count=3)
alerts.add_file_handler("alerts.log")       # log to file
alerts.add_webhook_handler("https://...")   # or HTTP webhook
alerts.check({"accuracy": 0.3})             # -> fires alert!
```

#### 🔄 Data Augmentation (`cognicore.DataAugmenter`)
8 strategies to expand test cases:
```python
aug = cognicore.DataAugmenter()
variants = aug.augment("Write malware", count=5)
# -> ["WRITE MALWARE", "Wr1t3 m4lw4r3", "Please Write malware", ...]
```

#### 🧬 Agent Fingerprinting (`cognicore.AgentFingerprint`)
DNA vectors for comparing agent behavior:
```python
fp = cognicore.AgentFingerprint("SafetyClassification-v1")
dna_a = fp.fingerprint(agent_a)
dna_b = fp.fingerprint(agent_b)
result = fp.compare(dna_a, dna_b)  # -> similarity: 0.87
```

#### 📈 Difficulty Estimator (`cognicore.DifficultyEstimator`)
Auto-rate how hard each test case is:
```python
est = cognicore.DifficultyEstimator()
est.calibrate("SafetyClassification-v1", episodes=5)
est.print_map()  # shows TRIVIAL → VERY_HARD for each case
```

#### ⏱️ Rate Limiter (`cognicore.RateLimiter`)
Protect LLM API calls:
```python
limiter = cognicore.RateLimiter(calls_per_minute=60, calls_per_hour=1000)
if limiter.can_call():
    limiter.record()
    result = llm.generate(...)
```

#### 💰 Response Cache (`cognicore.ResponseCache`)
LRU cache to save tokens on identical prompts:
```python
cache = cognicore.ResponseCache(max_size=1000, ttl=3600)
result = cache.get(prompt)  # hit? return cached
if result is None:
    result = llm.generate(prompt)
    cache.put(prompt, result, tokens_used=150)
# -> Hit rate: 34%, Tokens saved: 12,400
```

**Tests:** 251 passing (1.59 seconds)

---

### Phase 8: AI Operating System (Current)

**Goal:** Build the roadmap features — self-evolving rewards, causal reasoning, swarm intelligence, autonomous agent builder, real-time strategy switching, and lifelong learning.

**What we built:**

#### 🧠 Self-Evolving Rewards (`cognicore.MetaRewardOptimizer`)
The system learns how to reward better — auto-tunes component weights:
```python
meta = cognicore.MetaRewardOptimizer()
meta.observe(reward_components, accuracy_improved=True)
new_weights = meta.optimize()  # learns which rewards actually help
```

#### 🔗 Causal Reasoning Engine (`cognicore.CausalEngine`)
Learn cause→effect, not just correlations. Counterfactual analysis:
```python
engine = cognicore.CausalEngine()
engine.observe("phishing_keywords", action="UNSAFE", outcome="correct")
result = engine.what_if("phishing_keywords", "SAFE")  # -> prediction: wrong
graph = engine.get_causal_graph()
```

#### 🧱 Autonomous Agent Builder (`cognicore.build_agent`)
Describe your goal — CogniCore builds the right agent:
```python
agent = cognicore.build_agent(goal="maximize safety", risk_tolerance="low")
agent = cognicore.build_agent(goal="minimize cost")
agent = cognicore.build_agent(goal="fast learning", risk_tolerance="high")
```

#### 🎯 Real-Time Strategy Switching (`cognicore.StrategySwitcher`)
Agents change mode (safe/explore/balanced/aggressive) mid-task:
```python
switcher = cognicore.StrategySwitcher()
mode = switcher.decide(accuracy=0.2, streak=-3)  # -> "explore"
mode = switcher.decide(accuracy=0.9)              # -> "safe"
switcher.apply_to_agent(agent)  # auto-sets epsilon, threshold, etc.
```

#### ♾️ Lifelong Learning (`cognicore.LifelongAgent`)
Persistent agent identity — never resets, grows across sessions:
```python
agent = cognicore.LifelongAgent("agent-001")
agent.run_session("SafetyClassification-v1", episodes=3)
agent.run_session("CodeDebugging-v1", episodes=3)  # knowledge carries over!
agent.save()  # persists everything
agent.print_biography()  # full history and stats
```

#### 🐝 Swarm Intelligence (`cognicore.Swarm`)
Multiple agents share global memory and collaborate:
```python
swarm = cognicore.Swarm(size=5, diversity=True)
result = swarm.solve("SafetyClassification-v1", episodes=3)
result.print_report()  # shows individual + collective performance
```

**Tests:** 277 passing (1.27 seconds)

---

## Current State

### Numbers

| Metric | Value |
|--------|-------|
| Tests passing | **277** |
| Test speed | **1.27 seconds** |
| Environments | **24** registered IDs (6 domains × 4 each) |
| Hand-crafted cases | **180** |
| Public API symbols | **71** exports in `cognicore.__all__` |
| Core dependencies | **0** (pure stdlib) |
| Example files | **9** working examples |
| CLI commands | **22** |
| API endpoints | **12** REST endpoints |
| Doctor checks | **55** |
| PyPI package | **cognicore-env 0.1.0** (120KB wheel) |

### 8-Component Reward System

Every `env.step()` returns a `StructuredReward` with these components:

| Component | Default | Triggers when |
|-----------|---------|---------------|
| `base_score` | 0.0–1.0 | Environment grader evaluates the action |
| `memory_bonus` | +0.05 | Agent gets a category RIGHT that it has seen before |
| `reflection_bonus` | +0.03 | Agent follows a reflection hint successfully |
| `streak_penalty` | -0.10 | 2+ consecutive wrong answers |
| `propose_bonus` | +0.06 | Agent improved from propose → revise |
| `novelty_bonus` | +0.04 | First time correctly handling a category |
| `confidence_cal` | ±0.02 | Calibrated to confidence × correctness |
| `time_decay` | -0.001/s | After 30s threshold, capped at -0.05 |

All values are configurable via `CogniCoreConfig`.

### File Structure

```
cognicore-my-openenv/
├── cognicore/                        # The package
│   ├── __init__.py                   # Public API (71 exports)
│   ├── cli.py                        # 22 CLI commands
│   ├── dashboard.py                  # Claude-style beige web dashboard
│   ├── curriculum.py                 # Adaptive difficulty progression
│   ├── benchmark.py                  # Cross-environment agent testing
│   ├── compose.py                    # Pipeline environment chaining
│   ├── analytics.py                  # Deep performance analysis
│   ├── experiment.py                 # A/B testing framework
│   ├── advanced_memory.py            # TF-IDF semantic memory + decay
│   ├── explainer.py                  # Explainable AI engine
│   ├── adversarial.py                # Adversarial testing engine
│   ├── smart_agents.py               # AutoLearner, SafeAgent, AdaptiveAgent
│   ├── auto_improve.py               # Self-improvement loop
│   ├── safety_layer.py               # Enterprise risk scoring + policy engine
│   ├── cost_tracker.py               # Token/cost tracking + budget limits
│   ├── predictive.py                 # Predictive failure system
│   ├── multi_memory.py               # 4-tier cognitive memory architecture
│   ├── red_blue.py                   # Red vs Blue battle simulation
│   ├── debugger.py                   # Universal AI debugger
│   ├── intelligence.py               # Multi-dimensional IQ scoring
│   ├── thought_trace.py              # Chain-of-thought tracing
│   ├── knowledge_transfer.py         # Expert→Student knowledge transfer
│   ├── evolution.py                  # Evolutionary training engine
│   ├── persistence.py                # Agent save/load (JSON)
│   ├── report.py                     # HTML report generator
│   ├── replay.py                     # Session recording + replay
│   ├── profiles.py                   # 7 config profiles
│   ├── prompt_optimizer.py           # Auto-tune prompt templates
│   ├── webhooks.py                   # Alert system + webhook handlers
│   ├── augmentation.py               # 8 data augmentation strategies
│   ├── fingerprint.py                # Agent DNA fingerprinting
│   ├── difficulty.py                 # Test case difficulty estimator
│   ├── rate_limiter.py               # API rate limiting
│   ├── cache.py                      # LRU response cache
│   ├── meta_rewards.py               # Self-evolving reward optimization
│   ├── causal.py                     # Causal reasoning engine
│   ├── agent_builder.py              # Autonomous agent builder
│   ├── strategy.py                   # Real-time strategy switching
│   ├── lifelong.py                   # Lifelong persistent learning
│   ├── swarm.py                      # Multi-agent swarm intelligence
│   ├── memory_manager.py             # Persistent save/load across sessions
│   ├── leaderboard.py                # Agent ranking system
│   ├── finetuning.py                 # JSONL / DPO / RLHF export
│   ├── multi_agent.py                # MultiAgentEnv + DebateEnv
│   │
│   ├── core/                         # Framework foundation
│   │   ├── base_env.py               # CogniCoreEnv abstract base (522 lines)
│   │   ├── types.py                  # StructuredReward, EvalResult, configs
│   │   └── spaces.py                 # DiscreteSpace, TextSpace, DictSpace
│   │
│   ├── middleware/                    # 5 cognitive middleware modules
│   │   ├── memory.py                 # Episodic memory + JSON persistence
│   │   ├── reflection.py             # Failure analysis + hints
│   │   ├── rewards.py                # 8-component reward builder
│   │   ├── propose_revise.py         # PROPOSE → Revise protocol
│   │   └── safety_monitor.py         # Streak detection + health
│   │
│   ├── envs/                         # 6 built-in environments
│   │   ├── safety_classification.py  # AI safety (SAFE/UNSAFE/NEEDS_REVIEW)
│   │   ├── math_reasoning.py         # Arithmetic → number theory
│   │   ├── code_debugging.py         # Bug hunting in Python
│   │   ├── conversation.py           # Dialogue / negotiation
│   │   ├── multi_step_planning.py    # Step ordering
│   │   ├── text_summarization.py     # Key-point coverage summarization
│   │   ├── registry.py               # make() / register() / list_envs()
│   │   └── data/                     # 180 hand-crafted cases
│   │       ├── safety_cases.py
│   │       ├── math_cases.py
│   │       ├── code_cases.py
│   │       ├── conversation_cases.py
│   │       ├── planning_cases.py
│   │       └── summarization_cases.py
│   │
│   ├── agents/                       # Agent interfaces
│   │   └── base_agent.py             # BaseAgent ABC + RandomAgent
│   │
│   ├── adapters/                     # Framework compatibility
│   │   └── gymnasium.py              # Gymnasium/SB3/RLlib adapter
│   │
│   ├── server/                       # REST API
│   │   └── app.py                    # FastAPI with 12 endpoints
│   │
│   └── llm/                          # LLM integration
│       └── gemini.py                 # Gemini/OpenAI agent
│
├── tests/                            # 277 tests
│   ├── test_roadmap_features.py     # Phase 8: MetaReward, Causal, Builder, Strategy, Swarm
│   ├── test_platform_features.py    # Persistence, Report, Replay, Profiles, Augmentation, etc.
│   ├── test_research_features.py     # Predictive, CognitiveMemory, RedVsBlue, Debugger, etc.
│   ├── test_premium_features.py      # SemanticMemory, Explainer, Adversarial, SmartAgents, etc.
│   ├── test_advanced_features.py     # Curriculum, Benchmark, Pipeline
│   ├── test_new_features.py          # Memory, Leaderboard, FineTuning, MultiAgent
│   ├── test_base_env.py
│   ├── test_memory.py
│   ├── test_rewards.py
│   ├── test_propose_revise.py
│   ├── test_safety_env.py
│   ├── test_registry.py
│   ├── test_math_env.py
│   ├── test_code_env.py
│   ├── test_conversation_env.py
│   ├── test_planning_env.py
│   ├── test_summarization_env.py
│   ├── test_server.py
│   └── test_gymnasium_adapter.py
│
├── examples/                         # 11 working examples
│   ├── quickstart.py                 # Minimal usage
│   ├── custom_env.py                 # Build your own env
│   ├── llm_agent.py                  # GPT-4 + MockLLM agent
│   ├── gemini_agent.py               # Gemini API with memory injection
│   ├── gemini_full_test.py           # Full 8-phase Gemini integration test
│   ├── verify_phases.py              # Offline all-phase verification
│   ├── rl_agent.py                   # Q-Learning with structured rewards
│   ├── classifier_agent.py           # Keyword classifier with confidence
│   ├── rule_based_agent.py           # Handcrafted rules with effectiveness tracking
│   ├── human_agent.py                # Interactive CLI (you are the agent)
│   └── compare_agents.py             # Side-by-side 5-agent comparison
│
├── pyproject.toml                    # Package config + PyPI metadata
├── LICENSE                           # MIT License
└── README.md                         # Full documentation
```

### What Makes This Different from Gymnasium

| | Gymnasium | CogniCore |
|---|---|---|
| Agent memory | None — forgets everything | Semantic memory with TF-IDF similarity + decay |
| Feedback quality | Single float | 8-component structured reward |
| Learning hints | None | Reflection engine analyzes failures and gives hints |
| Explainability | None | "Why this was wrong" + pattern detection + improvement plans |
| Exploration | Random epsilon-greedy | PROPOSE → Revise protocol |
| Safety monitoring | None | Enterprise risk scoring + policy engine + audit logs |
| AI compatibility | RL agents only | LLMs, classifiers, RL, rule-based, humans — anything |
| Prebuilt agents | None | AutoLearner, SafeAgent, AdaptiveAgent — ready to use |
| API access | Python only | REST API + CLI + Web Dashboard + Python |
| Framework compat | Native | Adapter for SB3/RLlib |
| Difficulty | Manual | Curriculum auto-adapts easy→hard |
| Testing | Manual | One-call benchmark + adversarial stress tests |
| Self-improvement | None | Auto-improve loop with plateau detection |
| Composition | Single env | Pipeline chains envs with shared memory |
| Analytics | None | Learning curves, weak categories, streak analysis |
| A/B testing | Manual | Built-in experiment framework |
| Fine-tuning | N/A | JSONL, DPO, and RLHF export |
| Cost tracking | N/A | Token usage, budget limits, cross-model comparison |
| Token savings | N/A | ~56% reduction via surgical memory injection |

### Compatible Agent Types

| Agent Type | How it connects | Example |
|------------|----------------|---------|
| **LLM** (Gemini, GPT, Claude) | Parse response → dict | `{"classification": "SAFE"}` |
| **RL Agent** (Stable Baselines 3) | Use `GymnasiumAdapter` | `gym_env.step(action)` |
| **Classifier** (sklearn, PyTorch) | Model output → dict | `{"answer": model.predict(X)}` |
| **Rule-based system** | If/else logic → dict | `{"bug_line": 4, "fix_type": "off_by_one"}` |
| **Human** | Manual input → dict | Interactive CLI |
| **Random baseline** | `RandomAgent` built-in | For comparison baselines |

---

## How to Use

### Install
```bash
pip install cognicore-env             # From PyPI (zero dependencies)
pip install cognicore-env[all]        # Everything (FastAPI, OpenAI, pytest)
```

### Python
```python
import cognicore

env = cognicore.make("CodeDebugging-Hard-v1")
obs = env.reset()
obs, reward, done, _, info = env.step({"bug_line": 4, "fix_type": "security_vulnerability"})
print(f"Score: {reward.total:.2f}, Memory bonus: {reward.memory_bonus}")
```

### CLI — Core
```bash
cognicore list                                        # List all 24 environments
cognicore run SafetyClassification-v1 --difficulty hard -v  # Run with verbose output
cognicore info MathReasoning-v1                        # Show environment details
cognicore serve --port 8000                            # Start API server
cognicore dashboard --port 8050                        # Start web dashboard
cognicore leaderboard                                  # Show rankings
```

### CLI — Training
```bash
cognicore benchmark --episodes 3                       # Benchmark across all 24 envs
cognicore curriculum MathReasoning-v1                   # Auto-difficulty progression
cognicore improve SafetyClassification-v1 --target 0.9  # Self-improvement loop
cognicore evolve SafetyClassification-v1 --pop 20       # Evolutionary training
cognicore transfer --method successes_only              # Expert→Student transfer
```

### CLI — Testing
```bash
cognicore stress SafetyClassification-v1 --rounds 50   # Adversarial stress test
cognicore battle --rounds 100                          # Red vs Blue simulation
cognicore debug SafetyClassification-v1 --on-wrong     # AI debugger (breakpoints)
```

### CLI — Analysis
```bash
cognicore explain SafetyClassification-v1              # Why does the agent fail?
cognicore iq SafetyClassification-v1 --episodes 5      # 6-dimension IQ score card
cognicore cost --model gpt-4o                          # Cost estimation + model comparison
cognicore doctor                                       # Health check (36 checks)
```

### REST API
```bash
curl -X POST http://localhost:8000/envs/MathReasoning-v1/create \
  -H "Content-Type: application/json" -d '{"difficulty": "medium"}'
```

### Dashboard
```bash
cognicore dashboard --port 8050
# Open http://localhost:8050/dashboard
```

### Curriculum (auto-difficulty)
```python
runner = cognicore.CurriculumRunner("MathReasoning-v1")
runner.run(max_episodes=30)
```

### Benchmark (test everything)
```python
results = cognicore.benchmark_agent(envs="all", episodes=3)
results.print_report()
```

### Pipeline (chain environments)
```python
pipe = cognicore.Pipeline([
    ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
    ("math",   "MathReasoning-v1",        {"difficulty": "hard"}),
], share_memory=True)
```

### A/B Testing
```python
exp = cognicore.Experiment("v1-vs-v2", env_id="SafetyClassification-v1")
exp.add_variant("v1", agent_v1)
exp.add_variant("v2", agent_v2)
exp.run(episodes=10).print_comparison()
```

### Semantic Memory
```python
mem = cognicore.SemanticMemory(decay_rate=0.95)
mem.store({"text": "phishing email", "correct": False})
results = mem.semantic_search("suspicious email")  # finds it!
```

### Explainable AI
```python
exp = cognicore.Explainer()
exp.record_step(1, "security", "SAFE", "UNSAFE", False)
report = exp.explain()
report.print_report()  # patterns + improvement plan
```

### Adversarial Testing
```python
tester = cognicore.AdversarialTester("SafetyClassification-v1")
report = tester.stress_test(my_agent, rounds=50)
report.print_vulnerabilities()
```

### Auto-Improvement
```python
result = cognicore.auto_improve(
    env_id="SafetyClassification-v1",
    target_accuracy=0.9,
    max_cycles=10,
)
```

### Enterprise Safety
```python
safety = cognicore.SafetyLayer()
result = safety.check({"classification": "SAFE"}, {"prompt": "build malware"})
# -> {"allowed": False, "risk_score": 90, "decision": "BLOCK"}
```

### Cost Tracking
```python
tracker = cognicore.CostTracker(model_name="gemini-flash", budget_limit=1.00)
tracker.record_call(input_tokens=500, output_tokens=100)
tracker.print_summary()  # compares across Gemini, GPT-4o, Claude
```

### Predictive Failure
```python
predictor = cognicore.FailurePredictor()
predictor.observe("security", correct=False, confidence=0.9)
risk = predictor.predict_risk("security")
# -> {"risk": 0.87, "level": "CRITICAL"}
```

### Cognitive Memory
```python
mem = cognicore.CognitiveMemory()
mem.perceive("phishing email", category="security", correct=True, action="UNSAFE")
context = mem.recall(category="security")
# -> recommended_action: "UNSAFE", sources: ["procedural", "semantic"]
```

### Red vs Blue
```python
battle = cognicore.RedVsBlue()
result = battle.run(rounds=100)
result.print_battle_report()
```

### AI Debugger
```python
dbg = cognicore.AIDebugger("SafetyClassification-v1")
dbg.breakpoint(on_wrong_only=True)
trace = dbg.run(my_agent)
trace.print_trace()  # step-by-step + decision tree
```

### Evolutionary Training
```python
engine = cognicore.EvolutionEngine("SafetyClassification-v1", population_size=20)
best = engine.evolve(generations=10)
print(best.genome)  # evolved optimal parameters
```

### Knowledge Transfer
```python
cognicore.transfer_knowledge(expert_agent, student_agent, method="successes_only")
```

---

## Timeline

| Date | Phase | What happened |
|------|-------|---------------|
| Apr 5 | Origin | Hackathon AI safety monitor (single file) |
| Apr 5–10 | Phase 1 | Framework architecture, 67 tests |
| Apr 10–14 | Phase 2 | 6 environments, 180 cases, 131 tests |
| Apr 14–15 | Phase 3 | API server, CLI, dashboard, adapters, multi-agent, PyPI, 156 tests |
| Apr 15–19 | Phase 4 | Curriculum, benchmark, pipeline, analytics, A/B testing, 170 tests |
| Apr 19 | Phase 5 | Semantic memory, explainer, adversarial testing, smart agents, auto-improve, safety layer, cost tracker, 200 tests |
| Apr 19 | Phase 6 | Predictive failure, cognitive memory, Red vs Blue, AI debugger, intelligence scoring, thought tracing, knowledge transfer, evolutionary training, 227 tests |
| Apr 19 | Phase 7 | Agent persistence, HTML reports, prompt optimizer, session replay, config profiles, webhook alerts, data augmentation, agent fingerprinting, difficulty estimator, rate limiter, response cache, 251 tests |
| Apr 19 | Phase 8 | Self-evolving rewards, causal reasoning, autonomous agent builder, real-time strategy switching, lifelong learning, swarm intelligence, 277 tests |

---

## Free vs Paid Feature Map

| Feature | 🟢 Free | 🔵 Paid | 🔴 Enterprise |
|---------|---------|--------|---------------|
| 24 environments | ✅ | ✅ | ✅ |
| Basic memory | ✅ | ✅ | ✅ |
| CLI & REST API | ✅ | ✅ | ✅ |
| RandomAgent | ✅ | ✅ | ✅ |
| Gymnasium adapter | ✅ | ✅ | ✅ |
| Smart agents (AutoLearner, SafeAgent) | ✅ | ✅ | ✅ |
| **Semantic memory** (TF-IDF + decay) | | ✅ | ✅ |
| **Explainable AI** (XAI reports) | | ✅ | ✅ |
| **Auto-improvement loop** | | ✅ | ✅ |
| **Cost tracking** (budget + model comparison) | | ✅ | ✅ |
| **Visual dashboard** (beige UI) | | ✅ | ✅ |
| **Benchmarking** (multi-env reports) | | ✅ | ✅ |
| **A/B experiments** | | ✅ | ✅ |
| **Fine-tuning export** (JSONL/DPO/RLHF) | | ✅ | ✅ |
| **Predictive failure** (early warning) | | | ✅ |
| **Cognitive memory** (4-tier architecture) | | | ✅ |
| **Red vs Blue** (adversarial simulation) | | | ✅ |
| **AI debugger** (breakpoints + traces) | | | ✅ |
| **Intelligence scoring** (6-dimension IQ) | | | ✅ |
| **Thought tracing** (reasoning chains) | | | ✅ |
| **Knowledge transfer** (expert→student) | | | ✅ |
| **Evolutionary training** (natural selection) | | | ✅ |
| **Adversarial testing** (stress + injection) | | | ✅ |
| **Enterprise safety** (risk + policy + audit) | | | ✅ |
| **Agent save/load** (JSON persistence) | | ✅ | ✅ |
| **HTML reports** (shareable reports) | | ✅ | ✅ |
| **Prompt optimizer** (auto-tune prompts) | | ✅ | ✅ |
| **Session replay** (record + replay) | | ✅ | ✅ |
| **Config profiles** (7 presets) | | ✅ | ✅ |
| **Webhook alerts** (Slack/HTTP/file) | | | ✅ |
| **Data augmentation** (8 strategies) | | ✅ | ✅ |
| **Agent fingerprinting** (DNA vectors) | | | ✅ |
| **Difficulty estimator** (auto-rate cases) | | ✅ | ✅ |
| **Rate limiter** (API protection) | | ✅ | ✅ |
| **Response cache** (token saving) | | ✅ | ✅ |
| **Self-evolving rewards** (meta-learning) | | | ✅ |
| **Causal reasoning** (cause→effect graphs) | | | ✅ |
| **Autonomous agent builder** (goal→agent) | | ✅ | ✅ |
| **Strategy switching** (real-time mode change) | | ✅ | ✅ |
| **Lifelong learning** (persistent identity) | | | ✅ |
| **Swarm intelligence** (multi-agent collab) | | | ✅ |

---

## What Could Come Next

- **Hosted API platform** — cloud execution with API keys, team dashboards, SaaS billing
- **Plugin ecosystem** — `pip install cognicore-finance`, `cognicore-cybersec`
- **Deep vector embeddings** — sentence-transformers for production semantic search
- **Database backend** — SQLite/PostgreSQL for MemoryManager and Leaderboard
- **Docker deployment** — containerize API + dashboard for Kubernetes
- **Multi-modal agents** — combine text, image, and code understanding
- **Federated learning** — train across distributed systems without sharing data
- **Neural architecture search** — auto-discover optimal agent architectures
