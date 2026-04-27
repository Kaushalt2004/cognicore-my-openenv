"""
CodeDebuggingEnv — Find and fix bugs in code snippets.

Agents receive buggy code and must identify the bug line and fix type.
The cognitive middleware tracks performance by bug category (syntax,
logic_error, security, concurrency, etc.).

Usage::

    import cognicore

    env = cognicore.make("CodeDebugging-v1", difficulty="medium")
    obs = env.reset()
    obs, reward, done, truncated, info = env.step({
        "bug_line": 4,
        "fix_type": "wrong_operator",
    })
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace, TextSpace
from cognicore.envs.data.code_cases import (
    CodeCase,
    get_code_cases,
    grade_code_answer,
)


class CodeDebuggingEnv(CogniCoreEnv):
    """Code debugging environment.

    Agents inspect buggy code snippets and must identify:
    1. The line number containing the bug
    2. The type of fix required

    Parameters
    ----------
    difficulty : str
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    config : CogniCoreConfig or None
        Middleware configuration overrides.
    """

    def __init__(
        self,
        difficulty: str = "easy",
        config: Optional[CogniCoreConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.difficulty = difficulty
        super().__init__(config=config, **kwargs)

    def _setup(self, **kwargs: Any) -> None:
        self.observation_space = DictSpace(
            fields={
                "buggy_code": TextSpace(),
                "bug_description": TextSpace(),
                "language": TextSpace(),
                "category": TextSpace(),
            }
        )
        self.action_space = DictSpace(
            fields={
                "bug_line": DiscreteSpace(100),
                "fix_type": TextSpace(),
            }
        )

    def _generate_tasks(self) -> List[CodeCase]:
        return get_code_cases(self.difficulty)

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        case: CodeCase = self._tasks[self._current_step]

        predicted_line = action.get("bug_line", -1)
        predicted_fix = action.get("fix_type", "")

        try:
            predicted_line = int(predicted_line)
        except (ValueError, TypeError):
            predicted_line = -1

        base_score = grade_code_answer(
            predicted_line,
            case.bug_line,
            str(predicted_fix),
            case.fix_type,
        )
        correct = base_score >= 1.0

        return EvalResult(
            base_score=base_score,
            correct=correct,
            ground_truth=f"line {case.bug_line}: {case.fix_type}",
            predicted=f"line {predicted_line}: {predicted_fix}",
            category=case.category,
            metadata={
                "case_id": case.id,
                "explanation": case.explanation,
                "correct_fix": case.correct_fix,
                "difficulty": case.difficulty,
                "language": case.language,
            },
        )

    def _get_obs(self) -> Dict[str, Any]:
        case: CodeCase = self._tasks[self._current_step]
        return {
            "buggy_code": case.buggy_code,
            "bug_description": case.bug_description,
            "language": case.language,
            "category": case.category,
        }
