"""Tests for TextSummarizationEnv."""

import cognicore
from cognicore.envs.text_summarization import TextSummarizationEnv


class TestSummarizationEnv:
    def test_create_via_make(self):
        env = cognicore.make("Summarization-v1", difficulty="easy")
        assert isinstance(env, TextSummarizationEnv)

    def test_all_difficulties(self):
        for d in ("easy", "medium", "hard"):
            env = cognicore.make(f"Summarization-{d.capitalize()}-v1")
            assert env.difficulty == d

    def test_reset_returns_text(self):
        env = cognicore.make("Summarization-v1")
        obs = env.reset()
        assert "text" in obs
        assert "category" in obs
        assert len(obs["text"]) > 50

    def test_good_summary(self):
        env = cognicore.make("Summarization-v1", difficulty="easy")
        env.reset()
        # Summarize first case with key points
        obs, reward, done, _, info = env.step(
            {
                "summary": "MIT scientists created a solar panel 30% more efficient using perovskite material, halving solar energy costs within two years."
            }
        )
        assert reward.base_score > 0.5

    def test_empty_summary(self):
        env = cognicore.make("Summarization-v1", difficulty="easy")
        env.reset()
        obs, reward, done, _, info = env.step({"summary": ""})
        assert reward.base_score == 0.0

    def test_episode_completes(self):
        env = cognicore.make("Summarization-v1", difficulty="easy")
        env.reset()
        for _ in range(10):
            env.step({"summary": "test summary"})
        assert env._done is True


class TestSummarizationGrading:
    def test_grade_with_all_key_points(self):
        from cognicore.envs.data.summarization_cases import grade_summary

        score = grade_summary(
            "MIT created 30% more efficient solar panel using perovskite material within two years at half cost",
            "reference",
            ["MIT", "30% more efficient", "perovskite", "two years", "half cost"],
        )
        assert score >= 0.7

    def test_grade_with_no_key_points(self):
        from cognicore.envs.data.summarization_cases import grade_summary

        score = grade_summary(
            "Hello world", "reference", ["MIT", "perovskite", "solar"]
        )
        assert score < 0.3
