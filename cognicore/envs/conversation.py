"""
ConversationEnv — Choose the best conversational response in dialogue scenarios.

Agents receive a scenario context, a user message, and response options.
They must select the most appropriate response. Cognitive middleware tracks
performance by conversation type (customer_service, negotiation, etc.).

Usage::

    import cognicore

    env = cognicore.make("Conversation-v1", difficulty="medium")
    obs = env.reset()
    obs, reward, done, truncated, info = env.step({
        "response": "empathetic_action",
    })
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult
from cognicore.core.spaces import DictSpace, TextSpace
from cognicore.envs.data.conversation_cases import (
    ConversationCase,
    get_conversation_cases,
    grade_conversation,
)


class ConversationEnv(CogniCoreEnv):
    """Conversation / negotiation environment.

    Agents evaluate dialogue scenarios and choose the most
    appropriate response from multiple options. Covers customer
    service, negotiation, conflict resolution, medical ethics,
    diplomacy, and more.

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
                "scenario": TextSpace(),
                "user_message": TextSpace(),
                "response_options": DictSpace(fields={}),
                "category": TextSpace(),
            }
        )
        self.action_space = DictSpace(fields={"response": TextSpace()})

    def _generate_tasks(self) -> List[ConversationCase]:
        return get_conversation_cases(self.difficulty)

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        case: ConversationCase = self._tasks[self._current_step]
        predicted = action.get("response", "")

        base_score = grade_conversation(
            str(predicted),
            case.best_response,
            case.acceptable_responses,
        )
        correct = base_score >= 1.0

        return EvalResult(
            base_score=base_score,
            correct=correct,
            ground_truth=case.best_response,
            predicted=str(predicted),
            category=case.category,
            metadata={
                "case_id": case.id,
                "explanation": case.explanation,
                "difficulty": case.difficulty,
                "acceptable_responses": case.acceptable_responses,
            },
        )

    def _get_obs(self) -> Dict[str, Any]:
        case: ConversationCase = self._tasks[self._current_step]
        return {
            "scenario": case.scenario,
            "user_message": case.user_message,
            "response_options": case.response_options,
            "category": case.category,
        }
