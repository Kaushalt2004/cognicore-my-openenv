"""
SafetyClassificationEnv — Classify AI responses as SAFE / UNSAFE / NEEDS_REVIEW.

The original CogniCore hackathon environment, refactored as a ``CogniCoreEnv``
subclass.  All cognitive middleware (memory, reflection, structured rewards,
PROPOSE→Revise) is inherited automatically from the base class.

Usage::

    import cognicore

    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    obs = env.reset()
    obs, reward, done, truncated, info = env.step({"classification": "SAFE"})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace, TextSpace
from cognicore.envs.data.safety_cases import (
    SafetyCase,
    SafetyLabel,
    get_cases,
)
from cognicore.envs.data.graders import get_grader


class SafetyClassificationEnv(CogniCoreEnv):
    """AI Safety classification environment.

    Agents classify AI-generated responses as SAFE, UNSAFE, or
    NEEDS_REVIEW across three difficulty levels (easy / medium / hard).

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

    # ------------------------------------------------------------------
    # CogniCoreEnv abstract methods
    # ------------------------------------------------------------------

    def _setup(self, **kwargs: Any) -> None:
        """Configure observation/action spaces."""
        self.observation_space = DictSpace(
            fields={
                "case_id": TextSpace(),
                "prompt": TextSpace(),
                "response": TextSpace(),
                "difficulty": DiscreteSpace(3, labels=["easy", "medium", "hard"]),
                "category": TextSpace(),
            }
        )
        self.action_space = DiscreteSpace(3, labels=["SAFE", "UNSAFE", "NEEDS_REVIEW"])

    def _generate_tasks(self) -> List[SafetyCase]:
        """Return safety cases for the current difficulty."""
        return get_cases(self.difficulty)

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        """Grade the agent's classification."""
        # Parse action
        classification_str = action.get("classification", "NEEDS_REVIEW")
        if isinstance(classification_str, str):
            classification_str = classification_str.upper()
        try:
            predicted = SafetyLabel(classification_str)
        except ValueError:
            predicted = SafetyLabel.NEEDS_REVIEW

        # Get current case
        case: SafetyCase = self._tasks[self._current_step]
        ground_truth = case.ground_truth

        # Grade
        grader = get_grader(self.difficulty)
        base_score = grader(predicted, ground_truth)
        correct = predicted == ground_truth

        return EvalResult(
            base_score=base_score,
            correct=correct,
            ground_truth=ground_truth.value,
            predicted=predicted.value,
            category=case.category,
            metadata={
                "case_id": case.id,
                "explanation": case.explanation,
                "difficulty": case.difficulty,
            },
        )

    def _get_obs(self) -> Dict[str, Any]:
        """Build the raw observation for the current case."""
        case: SafetyCase = self._tasks[self._current_step]
        return {
            "case_id": case.id,
            "prompt": case.prompt,
            "response": case.response,
            "difficulty": case.difficulty,
            "category": case.category,
        }
