"""
MultiStepPlanningEnv — Order steps to solve planning problems correctly.

Agents receive a scenario with constraints and must order the steps
in the correct sequence. Cognitive middleware tracks performance by
planning category (sequential, scheduling, optimization, etc.).

Usage::

    import cognicore

    env = cognicore.make("Planning-v1", difficulty="hard")
    obs = env.reset()
    obs, reward, done, truncated, info = env.step({
        "order": ["A", "B", "C", "D", "E"],
    })
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult
from cognicore.core.spaces import DictSpace, TextSpace
from cognicore.envs.data.planning_cases import (
    PlanningCase,
    get_planning_cases,
    grade_plan_order,
)


class MultiStepPlanningEnv(CogniCoreEnv):
    """Multi-step planning environment.

    Agents receive a scenario with steps and constraints, and must
    order the steps in the correct sequence. Covers sequential tasks,
    project management, resource allocation, emergency response, and
    complex optimization problems.

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
                "steps": DictSpace(fields={}),
                "constraints": TextSpace(),  # serialized list
                "category": TextSpace(),
            }
        )
        self.action_space = DictSpace(
            fields={"order": TextSpace()}  # list of step IDs
        )

    def _generate_tasks(self) -> List[PlanningCase]:
        return get_planning_cases(self.difficulty)

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        case: PlanningCase = self._tasks[self._current_step]
        predicted_order = action.get("order", [])

        # Accept both list and comma-separated string
        if isinstance(predicted_order, str):
            predicted_order = [s.strip() for s in predicted_order.split(",")]

        base_score = grade_plan_order(predicted_order, case.correct_order)
        correct = base_score >= 1.0

        return EvalResult(
            base_score=base_score,
            correct=correct,
            ground_truth=",".join(case.correct_order),
            predicted=",".join(predicted_order)
            if isinstance(predicted_order, list)
            else str(predicted_order),
            category=case.category,
            metadata={
                "case_id": case.id,
                "explanation": case.explanation,
                "difficulty": case.difficulty,
                "num_steps": case.num_steps,
            },
        )

    def _get_obs(self) -> Dict[str, Any]:
        case: PlanningCase = self._tasks[self._current_step]
        return {
            "scenario": case.scenario,
            "steps": case.steps,
            "constraints": case.constraints,
            "num_steps": case.num_steps,
            "category": case.category,
        }
