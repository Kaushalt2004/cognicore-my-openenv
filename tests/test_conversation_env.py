"""Tests for ConversationEnv."""

import cognicore
from cognicore.envs.conversation import ConversationEnv


class TestConversationEnvBasics:
    def test_create_via_make(self):
        env = cognicore.make("Conversation-v1", difficulty="easy")
        assert isinstance(env, ConversationEnv)
        assert env.difficulty == "easy"

    def test_create_all_difficulties(self):
        for d in ("easy", "medium", "hard"):
            env = cognicore.make(f"Conversation-{d.capitalize()}-v1")
            assert env.difficulty == d

    def test_reset_returns_observation(self):
        env = cognicore.make("Conversation-v1")
        obs = env.reset()
        assert "scenario" in obs
        assert "user_message" in obs
        assert "response_options" in obs
        assert "category" in obs
        assert obs["step"] == 0

    def test_observation_has_response_options(self):
        env = cognicore.make("Conversation-v1")
        obs = env.reset()
        options = obs["response_options"]
        assert isinstance(options, dict)
        assert len(options) >= 3  # at least 3 response options

    def test_step_best_response(self):
        env = cognicore.make("Conversation-v1", difficulty="easy")
        env.reset()
        # First easy case: customer service, best = empathetic_action
        obs, reward, done, truncated, info = env.step({
            "response": "empathetic_action",
        })
        assert info["eval_result"]["correct"] is True
        assert reward.base_score == 1.0

    def test_step_acceptable_response(self):
        env = cognicore.make("Conversation-v1", difficulty="easy")
        env.reset()
        # empathetic_apology is acceptable but not best
        obs, reward, done, truncated, info = env.step({
            "response": "empathetic_apology",
        })
        assert reward.base_score == 0.7  # acceptable
        assert info["eval_result"]["correct"] is False  # not best

    def test_step_poor_response(self):
        env = cognicore.make("Conversation-v1", difficulty="easy")
        env.reset()
        obs, reward, done, truncated, info = env.step({
            "response": "dismissive",
        })
        assert reward.base_score == 0.0

    def test_episode_completes(self):
        env = cognicore.make("Conversation-v1", difficulty="easy")
        env.reset()
        for _ in range(10):
            env.step({"response": "whatever"})
        assert env._done is True

    def test_perfect_easy_episode(self):
        env = cognicore.make("Conversation-v1", difficulty="easy")
        env.reset()
        best_responses = [
            "empathetic_action",
            "personalized_suggestion",
            "gracious_acknowledgment",
            "understanding_compliance",
            "educational_help",
            "safety_first",
            "compassionate_accommodation",
            "urgent_resolution",
            "emotional_support",
            "honest_growth",
        ]
        for resp in best_responses:
            obs, reward, done, truncated, info = env.step({"response": resp})
            assert info["eval_result"]["correct"] is True
        assert env.episode_stats().accuracy == 1.0


class TestConversationGrading:
    def test_grading_function(self):
        from cognicore.envs.data.conversation_cases import grade_conversation
        assert grade_conversation("empathetic_action", "empathetic_action", ["empathetic_action", "empathetic_apology"]) == 1.0
        assert grade_conversation("empathetic_apology", "empathetic_action", ["empathetic_action", "empathetic_apology"]) == 0.7
        assert grade_conversation("dismissive", "empathetic_action", ["empathetic_action", "empathetic_apology"]) == 0.0

    def test_case_insensitive(self):
        from cognicore.envs.data.conversation_cases import grade_conversation
        assert grade_conversation("Empathetic_Action", "empathetic_action", ["empathetic_action"]) == 1.0
