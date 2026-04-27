"""Tests for MathReasoningEnv."""

import cognicore
from cognicore.envs.math_reasoning import MathReasoningEnv


class TestMathEnvBasics:
    def test_create_via_make(self):
        env = cognicore.make("MathReasoning-v1", difficulty="easy")
        assert isinstance(env, MathReasoningEnv)
        assert env.difficulty == "easy"

    def test_create_all_difficulties(self):
        for d in ("easy", "medium", "hard"):
            env = cognicore.make(f"MathReasoning-{d.capitalize()}-v1")
            assert env.difficulty == d

    def test_reset_returns_observation(self):
        env = cognicore.make("MathReasoning-v1")
        obs = env.reset()
        assert "question" in obs
        assert "category" in obs
        assert "answer_type" in obs
        assert obs["step"] == 0

    def test_step_correct_answer(self):
        env = cognicore.make("MathReasoning-v1", difficulty="easy")
        env.reset()
        # First easy case: 247 + 389 = 636
        obs, reward, done, truncated, info = env.step({"answer": 636})
        assert info["eval_result"]["correct"] is True
        assert reward.base_score == 1.0

    def test_step_wrong_answer(self):
        env = cognicore.make("MathReasoning-v1", difficulty="easy")
        env.reset()
        obs, reward, done, truncated, info = env.step({"answer": 999})
        assert info["eval_result"]["correct"] is False
        assert reward.base_score == 0.0

    def test_ten_easy_cases(self):
        env = cognicore.make("MathReasoning-v1", difficulty="easy")
        env.reset()
        answers = [636, 216, 391, 12, 30, 40, 30, 256, 13, 60]
        for ans in answers:
            obs, reward, done, truncated, info = env.step({"answer": ans})
            assert info["eval_result"]["correct"] is True
        assert done is True
        assert env.episode_stats().accuracy == 1.0

    def test_episode_completes(self):
        env = cognicore.make("MathReasoning-v1", difficulty="easy")
        env.reset()
        for _ in range(10):
            env.step({"answer": 0})
        assert env._done is True

    def test_memory_grows(self):
        env = cognicore.make("MathReasoning-v1", difficulty="easy")
        env.reset()
        env.step({"answer": 636})
        env.step({"answer": 216})
        assert len(env.memory.entries) == 2


class TestMathGrading:
    def test_partial_credit_close_answer(self):
        from cognicore.envs.data.math_cases import grade_math
        # Within 5% of correct answer
        assert grade_math(630, 636) == 0.5  # ~1% off
        assert grade_math(100, 636) == 0.0  # way off
        assert grade_math(636, 636) == 1.0  # exact

    def test_string_answers(self):
        from cognicore.envs.data.math_cases import grade_math
        assert grade_math("636", 636) == 1.0
        assert grade_math("abc", 636) == 0.0
