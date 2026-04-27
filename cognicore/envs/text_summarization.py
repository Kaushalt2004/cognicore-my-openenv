"""
TextSummarizationEnv — Summarize passages with key point coverage.

Usage::

    env = cognicore.make("Summarization-v1", difficulty="medium")
    obs = env.reset()
    obs, reward, done, _, info = env.step({"summary": "MIT created efficient solar panels..."})
"""

from __future__ import annotations

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult
from cognicore.core.spaces import DictSpace, TextSpace
from cognicore.envs.data.summarization_cases import (
    get_summarization_cases, grade_summary,
)


class TextSummarizationEnv(CogniCoreEnv):
    """Text summarization environment.

    Agents receive passages and must produce concise summaries
    capturing all key points.
    """

    def __init__(self, difficulty="easy", config=None, **kwargs):
        self.difficulty = difficulty
        super().__init__(config=config, **kwargs)

    def _setup(self, **kwargs):
        self.observation_space = DictSpace(fields={
            "text": TextSpace(), "category": TextSpace(),
            "max_summary_length": TextSpace(),
        })
        self.action_space = DictSpace(fields={"summary": TextSpace()})

    def _generate_tasks(self):
        return get_summarization_cases(self.difficulty)

    def _evaluate(self, action):
        case = self._tasks[self._current_step]
        predicted = action.get("summary", "")
        score = grade_summary(str(predicted), case.reference_summary, case.key_points)
        return EvalResult(
            base_score=score, correct=score >= 0.7,
            ground_truth=case.reference_summary, predicted=str(predicted),
            category=case.category,
            metadata={"case_id": case.id, "key_points": case.key_points, "difficulty": case.difficulty},
        )

    def _get_obs(self):
        case = self._tasks[self._current_step]
        return {"text": case.text, "category": case.category,
                "max_summary_length": case.max_summary_length}
