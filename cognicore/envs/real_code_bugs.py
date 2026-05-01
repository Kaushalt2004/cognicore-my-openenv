"""
RealWorldCodeBugs — Environment using real Python bugs from production.

Unlike CodeDebugging-v1 (synthetic), this uses actual bug patterns
that developers encounter: SQL injection, mutable defaults, race
conditions, resource leaks, and more.

Usage::

    import cognicore as cc
    env = cc.make("RealWorldCodeBugs-v1")
    obs = env.reset()
    # obs["code"] contains REAL buggy Python code
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult
from cognicore.core.spaces import DictSpace, DiscreteSpace
from cognicore.envs.data.real_code_cases import REAL_CODE_CASES


class RealWorldCodeBugsEnv(CogniCoreEnv):
    """Real-world code bug detection environment.

    The agent must identify the bug type in real Python code snippets
    taken from common production bug patterns.
    """

    BUG_TYPES = [
        "off_by_one", "mutable_default", "closure_bug", "bare_except",
        "silent_failure", "race_condition", "sql_injection", "type_error",
        "resource_leak", "logic_error", "hardcoded_secret", "async_antipattern",
    ]

    def _setup(self, **kwargs: Any) -> None:
        self.difficulty: str = kwargs.get("difficulty", "easy")
        self.num_tasks: int = kwargs.get("num_tasks", 10)

        self.action_space = DiscreteSpace(
            n=len(self.BUG_TYPES),
            labels=self.BUG_TYPES,
        )
        self.observation_space = DictSpace(fields={
            "code": "The buggy Python code",
            "language": "Programming language",
        })

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        cases = REAL_CODE_CASES.copy()
        n = min(self.num_tasks, len(cases))
        selected = random.sample(cases, n)
        return selected

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        task = self._tasks[self._current_step]
        predicted = action.get("bug_type", "").lower().strip()
        correct_type = task["bug_type"]
        is_correct = predicted == correct_type

        # Partial credit for related categories
        partial = 0.0
        related = {
            ("bare_except", "silent_failure"): 0.5,
            ("silent_failure", "bare_except"): 0.5,
            ("type_error", "logic_error"): 0.3,
            ("logic_error", "type_error"): 0.3,
        }
        if not is_correct:
            partial = related.get((predicted, correct_type), 0.0)

        base_score = 1.0 if is_correct else partial

        return EvalResult(
            base_score=base_score,
            correct=is_correct,
            category=correct_type,
            ground_truth=correct_type,
            predicted=predicted,
            metadata={
                "explanation": task["explanation"],
                "fix": task["fix"],
                "code_preview": task["code"][:80],
            },
        )

    def _get_obs(self) -> Dict[str, Any]:
        task = self._tasks[self._current_step]
        obs: Dict[str, Any] = {
            "code": task["code"],
            "language": task["language"],
        }
        if self.difficulty == "easy":
            obs["hint"] = f"Look for: {task['bug_type'].replace('_', ' ')}"
        elif self.difficulty == "medium":
            obs["possible_types"] = random.sample(
                self.BUG_TYPES, min(4, len(self.BUG_TYPES))
            )
            if task["bug_type"] not in obs["possible_types"]:
                obs["possible_types"][-1] = task["bug_type"]
                random.shuffle(obs["possible_types"])
        return obs
