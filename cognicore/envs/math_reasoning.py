"""
MathReasoningEnv — Solve arithmetic, algebra, and logic problems.

Agents receive math questions and must provide numerical answers.
The cognitive middleware tracks performance by category (addition,
algebra, number_theory, etc.) to help agents learn patterns.

Usage::

    import cognicore

    env = cognicore.make("MathReasoning-v1", difficulty="medium")
    obs = env.reset()
    obs, reward, done, truncated, info = env.step({"answer": 42})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult
from cognicore.core.spaces import DictSpace, TextSpace
from cognicore.envs.data.math_cases import (
    MathCase,
    get_math_cases,
    grade_math,
)


class MathReasoningEnv(CogniCoreEnv):
    """Math reasoning environment.

    Agents solve arithmetic, algebra, and advanced math problems.
    Cognitive middleware tracks per-category accuracy to provide
    targeted memory context and reflection hints.

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
                "question": TextSpace(),
                "category": TextSpace(),
                "difficulty": TextSpace(),
                "answer_type": TextSpace(),
            }
        )
        self.action_space = DictSpace(
            fields={"answer": TextSpace()}
        )

    def _generate_tasks(self) -> List[MathCase]:
        return get_math_cases(self.difficulty)

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        case: MathCase = self._tasks[self._current_step]
        predicted = action.get("answer")

        base_score = grade_math(predicted, case.answer, case.answer_type)
        correct = base_score >= 1.0

        return EvalResult(
            base_score=base_score,
            correct=correct,
            ground_truth=case.answer,
            predicted=predicted,
            category=case.category,
            metadata={
                "case_id": case.id,
                "explanation": case.explanation,
                "difficulty": case.difficulty,
                "answer_type": case.answer_type,
            },
        )

    def _get_obs(self) -> Dict[str, Any]:
        case: MathCase = self._tasks[self._current_step]
        return {
            "question": case.question,
            "category": case.category,
            "difficulty": case.difficulty,
            "answer_type": case.answer_type,
        }
