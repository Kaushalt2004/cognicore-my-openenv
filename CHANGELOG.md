# Changelog

All notable changes to CogniCore are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.4.0] — 2026-04-30

### Added
- **Custom error hierarchy** — 7 exception classes with actionable messages and suggestions
  - `InvalidEnvironmentError` — shows similar env names ("Did you mean?")
  - `InvalidConfigError` — catches bad config on construction
  - `AgentInterfaceError` — clear message when agent lacks `act()`
  - `EpisodeFinishedError` — replaces silent failure on double-step
- **AgentProtocol** — runtime-checkable Protocol for duck-typing agent validation
- **Config validation** — `CogniCoreConfig.__post_init__()` validates all fields immediately

### Changed
- **Replaced 300+ `print()` calls with `logging`** across 31 modules
- **Type-safe API** — `train()` and `evaluate()` validate agent/env/episodes before running
- `make()` now raises `InvalidEnvironmentError` instead of generic `KeyError`
- `step()` raises `EpisodeFinishedError` instead of silently returning empty data

---

## [0.3.0] — 2026-04-28

### Added
- **CLI commands** — `cognicore train`, `cognicore demo`, `cognicore metrics`
- **Config-driven training** — YAML config files (`configs/default.yaml`, `configs/strict_safety.yaml`)
- **Deterministic benchmarks** — 5 seeds × 10 episodes, mean ± std dev, saved JSON reports
- **Real-world use case** — `examples/chatbot_safety_eval.py` (chatbot safety evaluation)
- **Learning curve graph** — `docs/learning_curve.png` embedded in README
- **README overhaul** — tagline, before/after output, comparison table, how-it-works diagram
- **Known limitations section** — 5 honest limitations documented
- **Roadmap** — plugin ecosystem vision (cybersec, finance, eval)
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`

---

## [0.2.0] — 2026-04-15

### Added
- **24 environments** across 6 domains (safety, math, code, conversation, planning, summarization)
- **`cc.train()` / `cc.evaluate()`** — clean 2-line API
- **22 CLI commands** — `cognicore list`, `run`, `benchmark`, `serve`, `dashboard`, etc.
- **Structured Rewards** — 8-component reward signal per step
- **PROPOSE → Revise protocol** — tentative exploration before commitment
- **Safety Monitor** — streak detection and health status
- **Gymnasium adapter** — `CogniCoreGymAdapter` for RL compatibility
- **API server** — FastAPI-based REST API
- **GitHub Actions CI** — tests on Python 3.9/3.11/3.12, linting, security scan

---

## [0.1.0] — 2026-04-05

### Added
- Initial release
- Core `CogniCoreEnv` base class with Memory, Reflection, and Structured Rewards
- `SafetyClassification-v1` environment
- Basic agent interface (`BaseAgent`, `RandomAgent`)
- `cognicore.make()` factory function

---

## Roadmap

| Target | Feature | Status |
|--------|---------|--------|
| v0.5.0 (June 2026) | Embedding-based semantic memory (optional `sentence-transformers`) | 🔜 Planned |
| v0.5.0 (June 2026) | Parallel episode execution (`asyncio`) | 🔜 Planned |
| v0.6.0 (Aug 2026) | Real-world dataset loader (HuggingFace datasets integration) | 📋 Backlog |
| v0.6.0 (Aug 2026) | `cognicore-eval` — LLM evaluation suite | 📋 Backlog |
| v0.7.0 (Oct 2026) | `cognicore debug agent.py` — CLI debugger with breakpoints | 📋 Backlog |
| v1.0.0 (Dec 2026) | Stable API, full documentation, production-ready | 📋 Backlog |

[0.4.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Kaushalt2004/cognicore-my-openenv/releases/tag/v0.1.0
