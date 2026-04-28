# Contributing to CogniCore

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/Kaushalt2004/cognicore-my-openenv.git
cd cognicore-my-openenv
pip install -e ".[dev]"
python -m cognicore.cli doctor
```

## Development Workflow

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Push and open a Pull Request

## Code Style

- Python 3.9+ compatible
- Use type hints
- Run `ruff check cognicore/` before submitting
- Keep zero external dependencies in the core package

## What to Contribute

- **Bug fixes** — check open issues labeled `🐛 bug`
- **New environments** — inherit from `CogniCoreEnv`, implement 4 abstract methods
- **Tests** — edge cases, failure scenarios, integration flows
- **Documentation** — examples, tutorials, docstrings

## Adding a New Environment

```python
from cognicore.core.base_env import CogniCoreEnv

class MyEnv(CogniCoreEnv):
    def _setup(self, **kwargs): ...
    def _generate_tasks(self): ...
    def _evaluate(self, action): ...
    def _get_obs(self): ...
```

Then register it in `cognicore/envs/registry.py`.

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`

## Questions?

Open an issue with the `❓ question` label.
