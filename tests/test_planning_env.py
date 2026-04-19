"""Tests for MultiStepPlanningEnv."""

import cognicore
from cognicore.envs.multi_step_planning import MultiStepPlanningEnv


class TestPlanningEnvBasics:
    def test_create_via_make(self):
        env = cognicore.make("Planning-v1", difficulty="easy")
        assert isinstance(env, MultiStepPlanningEnv)
        assert env.difficulty == "easy"

    def test_create_all_difficulties(self):
        for d in ("easy", "medium", "hard"):
            env = cognicore.make(f"Planning-{d.capitalize()}-v1")
            assert env.difficulty == d

    def test_reset_returns_observation(self):
        env = cognicore.make("Planning-v1")
        obs = env.reset()
        assert "scenario" in obs
        assert "steps" in obs
        assert "constraints" in obs
        assert "num_steps" in obs
        assert obs["step"] == 0

    def test_observation_steps_are_dict(self):
        env = cognicore.make("Planning-v1")
        obs = env.reset()
        assert isinstance(obs["steps"], dict)
        assert len(obs["steps"]) >= 4

    def test_step_perfect_order(self):
        env = cognicore.make("Planning-v1", difficulty="easy")
        env.reset()
        # First easy case: make tea — A, B, C, D, E
        obs, reward, done, truncated, info = env.step({
            "order": ["A", "B", "C", "D", "E"],
        })
        assert info["eval_result"]["correct"] is True
        assert reward.base_score == 1.0

    def test_step_string_order(self):
        env = cognicore.make("Planning-v1", difficulty="easy")
        env.reset()
        # Accept comma-separated string
        obs, reward, done, truncated, info = env.step({
            "order": "A,B,C,D,E",
        })
        assert reward.base_score == 1.0

    def test_step_reversed_order(self):
        env = cognicore.make("Planning-v1", difficulty="easy")
        env.reset()
        obs, reward, done, truncated, info = env.step({
            "order": ["E", "D", "C", "B", "A"],
        })
        assert reward.base_score == 0.0  # fully reversed

    def test_step_partial_order(self):
        env = cognicore.make("Planning-v1", difficulty="easy")
        env.reset()
        # Swap first two and last two — partial credit
        obs, reward, done, truncated, info = env.step({
            "order": ["B", "A", "C", "E", "D"],
        })
        assert 0.0 < reward.base_score < 1.0  # partial credit

    def test_episode_completes(self):
        env = cognicore.make("Planning-v1", difficulty="easy")
        env.reset()
        for _ in range(10):
            env.step({"order": ["A", "B", "C"]})
        assert env._done is True

    def test_memory_grows(self):
        env = cognicore.make("Planning-v1", difficulty="easy")
        env.reset()
        env.step({"order": ["A", "B", "C", "D", "E"]})
        env.step({"order": ["A", "B", "C", "D", "E"]})
        assert len(env.memory.entries) == 2


class TestPlanGrading:
    def test_perfect_order(self):
        from cognicore.envs.data.planning_cases import grade_plan_order
        assert grade_plan_order(["A", "B", "C"], ["A", "B", "C"]) == 1.0

    def test_reversed_order(self):
        from cognicore.envs.data.planning_cases import grade_plan_order
        score = grade_plan_order(["C", "B", "A"], ["A", "B", "C"])
        assert score == 0.0

    def test_one_swap(self):
        from cognicore.envs.data.planning_cases import grade_plan_order
        score = grade_plan_order(["A", "C", "B"], ["A", "B", "C"])
        assert 0.5 < score < 1.0  # mostly correct

    def test_first_step_bonus(self):
        from cognicore.envs.data.planning_cases import grade_plan_order
        # Same number of inversions but one starts with correct first step
        score_right_start = grade_plan_order(["A", "C", "B", "D"], ["A", "B", "C", "D"])
        score_wrong_start = grade_plan_order(["B", "A", "C", "D"], ["A", "B", "C", "D"])
        assert score_right_start > score_wrong_start

    def test_empty_order(self):
        from cognicore.envs.data.planning_cases import grade_plan_order
        assert grade_plan_order([], ["A", "B", "C"]) == 0.0
