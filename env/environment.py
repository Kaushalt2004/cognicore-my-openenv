"""
AISafetyEnv — The core OpenEnv environment for AI Safety monitoring.

Implements the standard OpenEnv interface: reset(), step(), state().
Integrates CogniCore's memory, reflection, and safety layers.
"""

from typing import Dict, Any, Optional, Tuple
from env.models import SafetyLabel, Observation, Action, RewardInfo, EnvState, SafetyCase
from env.dataset import get_cases, CASES_BY_DIFFICULTY
from env.graders import grader_easy, grader_medium, grader_hard
from cognicore.memory.vector_memory import VectorMemory
from cognicore.reflection.reflection import Reflection
from cognicore.safety.safety import Safety


# Map difficulty to grader
GRADERS = {
    "easy": grader_easy,
    "medium": grader_medium,
    "hard": grader_hard,
}


class AISafetyEnv:
    """AI Safety Monitor environment.

    An OpenEnv environment where agents classify AI responses as
    SAFE, UNSAFE, or NEEDS_REVIEW across three difficulty levels.

    CogniCore integration:
      - VectorMemory: provides context from similar past cases
      - Reflection: hints when the agent makes repeated mistakes
      - Safety: penalizes wrong streaks (3+ consecutive errors)
    """

    def __init__(self):
        # CogniCore components
        self.memory = VectorMemory()
        self.reflection = Reflection(self.memory)
        self.safety = Safety(streak_threshold=3, streak_penalty=-0.1)

        # Environment state
        self.current_task: Optional[str] = None
        self.difficulty: Optional[str] = None
        self.cases: list = []
        self.current_step: int = 0
        self.max_steps: int = 0
        self.total_reward: float = 0.0
        self.done: bool = True
        self.rewards: list = []

        # Tracking
        self.correct_count: int = 0
        self.episode_count: int = 0

    def reset(self, task: str = "binary_safety_classification", difficulty: str = None) -> Dict[str, Any]:
        """Reset the environment for a new episode.

        Args:
            task: Task name (determines which cases to load).
            difficulty: Override difficulty level. If None, inferred from task.

        Returns:
            First observation as a dict.
        """
        # Infer difficulty from task name
        if difficulty is None:
            if "binary" in task or "easy" in task:
                difficulty = "easy"
            elif "nuanced" in task or "medium" in task:
                difficulty = "medium"
            elif "adversarial" in task or "hard" in task:
                difficulty = "hard"
            else:
                difficulty = "easy"

        self.current_task = task
        self.difficulty = difficulty
        self.cases = get_cases(difficulty)
        self.max_steps = len(self.cases)
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False
        self.rewards = []
        self.correct_count = 0
        self.episode_count += 1

        # Reset safety streak
        self.safety.reset()

        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take one step in the environment.

        Args:
            action: Dict with at minimum {"classification": "SAFE"|"UNSAFE"|"NEEDS_REVIEW"}.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self.done:
            return self._get_observation(), 0.0, True, {"error": "Episode is done. Call reset()."}

        if self.current_step >= self.max_steps:
            self.done = True
            return self._get_observation(), 0.0, True, {"error": "Max steps reached."}

        # Parse action
        classification_str = action.get("classification", "NEEDS_REVIEW").upper()
        try:
            predicted = SafetyLabel(classification_str)
        except ValueError:
            predicted = SafetyLabel.NEEDS_REVIEW

        # Get current case
        case = self.cases[self.current_step]
        ground_truth = case.ground_truth

        # Grade the classification
        grader = GRADERS.get(self.difficulty, grader_easy)
        base_reward = grader.grade(predicted, ground_truth)
        correct = (predicted == ground_truth)

        # CogniCore: Memory bonus
        memory_bonus = 0.0
        if correct:
            past_entries = self.memory.retrieve(case.category, top_k=3)
            if past_entries:
                past_correct = sum(1 for e in past_entries if e["correct"])
                if past_correct > 0:
                    memory_bonus = 0.05  # Consistency bonus

        # CogniCore: Safety streak penalty
        streak_penalty = self.safety.check(correct)

        # Total reward
        total_reward = base_reward + memory_bonus + streak_penalty

        # Update tracking
        if correct:
            self.correct_count += 1
        self.total_reward += total_reward
        self.rewards.append(round(total_reward, 2))

        # CogniCore: Store in memory
        self.memory.store(
            case_id=case.id,
            category=case.category,
            predicted=predicted.value,
            ground_truth=ground_truth.value,
            reward=total_reward,
            correct=correct,
            episode=self.episode_count,
        )

        # Build reward info
        reward_info = RewardInfo(
            reward=total_reward,
            correct=correct,
            ground_truth=ground_truth,
            predicted=predicted,
            memory_bonus=memory_bonus,
            streak_penalty=streak_penalty,
            details=case.explanation,
        )

        # Advance step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Build info
        info = {
            "reward_info": reward_info.model_dump(),
            "case_id": case.id,
            "category": case.category,
            "wrong_streak": self.safety.get_wrong_streak(),
        }

        return self._get_observation(), total_reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return the full environment state."""
        accuracy = self.correct_count / self.current_step if self.current_step > 0 else 0.0

        # Phase 2 bounding: strict (0, 1) limit on final calculated score (reward/max_steps)
        bounded_reward = min(max(self.total_reward, 0.01 * self.max_steps), 0.99 * self.max_steps)

        env_state = EnvState(
            current_step=self.current_step,
            max_steps=self.max_steps,
            total_reward=round(bounded_reward, 4),
            done=self.done,
            task_name=self.current_task or "",
            difficulty=self.difficulty or "",
            cases_seen=self.current_step,
            correct_count=self.correct_count,
            wrong_streak=self.safety.get_wrong_streak(),
            accuracy=round(accuracy, 4),
        )
        return env_state.model_dump()

    def _get_observation(self) -> Dict[str, Any]:
        """Build the current observation for the agent."""
        if self.done or self.current_step >= len(self.cases):
            return Observation(
                case_id="done",
                prompt="",
                response="",
                difficulty=self.difficulty or "",
                category="",
                step=self.current_step,
                max_steps=self.max_steps,
            ).model_dump()

        case = self.cases[self.current_step]

        # CogniCore: Memory context
        memory_context = self.memory.get_context_for_observation(
            case.category, top_k=3
        )

        # CogniCore: Reflection hint
        reflection_hint = self.reflection.get_reflection_hint(case.category)

        obs = Observation(
            case_id=case.id,
            prompt=case.prompt,
            response=case.response,
            difficulty=case.difficulty,
            category=case.category,
            memory_context=memory_context,
            reflection_hint=reflection_hint,
            step=self.current_step,
            max_steps=self.max_steps,
        )
        return obs.model_dump()

    def get_score(self) -> float:
        """Return the normalized score for the episode."""
        if self.max_steps == 0:
            return 0.0
        score = self.total_reward / self.max_steps
        return round(min(max(score, 0.01), 0.99), 4)
