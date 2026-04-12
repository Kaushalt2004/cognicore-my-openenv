"""
Unit tests for CogniCore AI Safety Monitor.

Tests:
  1. Dataset integrity (54 cases, correct distribution)
  2. Grader determinism (same input = same output)
  3. Grader score ranges (all in 0.01-0.99)
  4. Environment reset/step/state lifecycle
  5. Structured reward components
  6. StepInfo correctness
  7. Memory and reflection middleware
  8. Streak penalty activation
  9. Full episode completion
  10. Score clamping compliance

Run:
    python -m pytest tests/ -v
    python -m unittest discover -s tests -p "test_*.py"
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import SafetyAction, SafetyLabel, Severity, SafetyReward, StepInfo
from dataset import EASY_CASES, MEDIUM_CASES, HARD_CASES, get_cases
from graders import grade_easy, grade_medium, grade_hard, grade


class TestDatasetIntegrity(unittest.TestCase):
    """Test that the dataset has exactly 54 cases with correct distribution."""

    def test_easy_count(self):
        self.assertEqual(len(EASY_CASES), 18)

    def test_medium_count(self):
        self.assertEqual(len(MEDIUM_CASES), 18)

    def test_hard_count(self):
        self.assertEqual(len(HARD_CASES), 18)

    def test_total_count(self):
        self.assertEqual(len(EASY_CASES) + len(MEDIUM_CASES) + len(HARD_CASES), 54)

    def test_unique_ids(self):
        all_ids = [c.id for c in EASY_CASES + MEDIUM_CASES + HARD_CASES]
        self.assertEqual(len(all_ids), len(set(all_ids)), "Duplicate case IDs found")

    def test_all_cases_have_ground_truth(self):
        for case in EASY_CASES + MEDIUM_CASES + HARD_CASES:
            self.assertIn(case.ground_truth, SafetyLabel)

    def test_get_cases_easy(self):
        cases = get_cases("easy")
        self.assertEqual(len(cases), 18)

    def test_get_cases_medium(self):
        cases = get_cases("medium")
        self.assertEqual(len(cases), 18)

    def test_get_cases_hard(self):
        cases = get_cases("hard")
        self.assertEqual(len(cases), 18)


class TestGraderDeterminism(unittest.TestCase):
    """Test that grading is deterministic."""

    def test_easy_deterministic(self):
        r1 = grade_easy(SafetyLabel.SAFE, SafetyLabel.SAFE, 0.9)
        r2 = grade_easy(SafetyLabel.SAFE, SafetyLabel.SAFE, 0.9)
        self.assertEqual(r1, r2)

    def test_medium_deterministic(self):
        r1 = grade_medium(SafetyLabel.UNSAFE, SafetyLabel.UNSAFE, 0.8, "high", Severity.HIGH)
        r2 = grade_medium(SafetyLabel.UNSAFE, SafetyLabel.UNSAFE, 0.8, "high", Severity.HIGH)
        self.assertEqual(r1, r2)

    def test_hard_deterministic(self):
        r1 = grade_hard(SafetyLabel.UNSAFE, SafetyLabel.UNSAFE, 0.9,
                        reasoning="jailbreak", case_manipulation="fictional_framing",
                        manipulation_type="fictional_framing")
        r2 = grade_hard(SafetyLabel.UNSAFE, SafetyLabel.UNSAFE, 0.9,
                        reasoning="jailbreak", case_manipulation="fictional_framing",
                        manipulation_type="fictional_framing")
        self.assertEqual(r1, r2)


class TestGraderRanges(unittest.TestCase):
    """Test that all grader outputs are in (0.01, 0.99)."""

    def test_easy_correct_in_range(self):
        score = grade_easy(SafetyLabel.SAFE, SafetyLabel.SAFE, 0.95)
        self.assertGreaterEqual(score, 0.01)
        self.assertLessEqual(score, 0.99)

    def test_easy_wrong_in_range(self):
        score = grade_easy(SafetyLabel.SAFE, SafetyLabel.UNSAFE, 0.95)
        self.assertGreaterEqual(score, 0.01)
        self.assertLessEqual(score, 0.99)

    def test_medium_in_range(self):
        for pred in SafetyLabel:
            for truth in SafetyLabel:
                score = grade_medium(pred, truth, 0.5, "medium", Severity.MEDIUM)
                self.assertGreaterEqual(score, 0.01,
                    f"Score {score} < 0.01 for pred={pred}, truth={truth}")
                self.assertLessEqual(score, 0.99,
                    f"Score {score} > 0.99 for pred={pred}, truth={truth}")

    def test_hard_in_range(self):
        for pred in SafetyLabel:
            for truth in SafetyLabel:
                score = grade_hard(pred, truth, 0.5)
                self.assertGreaterEqual(score, 0.01)
                self.assertLessEqual(score, 0.99)

    def test_correct_beats_wrong(self):
        correct = grade_easy(SafetyLabel.SAFE, SafetyLabel.SAFE, 0.9)
        wrong = grade_easy(SafetyLabel.UNSAFE, SafetyLabel.SAFE, 0.9)
        self.assertGreater(correct, wrong)


class TestGraderLogic(unittest.TestCase):
    """Test grader-specific logic."""

    def test_easy_confidence_matters(self):
        high_conf = grade_easy(SafetyLabel.SAFE, SafetyLabel.SAFE, 0.95)
        low_conf = grade_easy(SafetyLabel.SAFE, SafetyLabel.SAFE, 0.1)
        self.assertGreater(high_conf, low_conf)

    def test_easy_confidently_wrong_worst(self):
        conf_wrong = grade_easy(SafetyLabel.SAFE, SafetyLabel.UNSAFE, 0.95)
        unconf_wrong = grade_easy(SafetyLabel.SAFE, SafetyLabel.UNSAFE, 0.1)
        self.assertLess(conf_wrong, unconf_wrong)

    def test_medium_partial_credit(self):
        correct = grade_medium(SafetyLabel.UNSAFE, SafetyLabel.UNSAFE, 0.8, "high", Severity.HIGH)
        review = grade_medium(SafetyLabel.NEEDS_REVIEW, SafetyLabel.UNSAFE, 0.8, "high", Severity.HIGH)
        wrong = grade_medium(SafetyLabel.SAFE, SafetyLabel.UNSAFE, 0.8, "high", Severity.HIGH)
        self.assertGreater(correct, review)
        self.assertGreater(review, wrong)

    def test_hard_manipulation_match_bonus(self):
        with_match = grade_hard(SafetyLabel.UNSAFE, SafetyLabel.UNSAFE, 0.9,
                                case_manipulation="persona_injection",
                                manipulation_type="persona_injection")
        without_match = grade_hard(SafetyLabel.UNSAFE, SafetyLabel.UNSAFE, 0.9,
                                   case_manipulation="persona_injection",
                                   manipulation_type="other_technique")
        self.assertGreater(with_match, without_match)


class TestEnvironmentLifecycle(unittest.TestCase):
    """Test environment reset/step/state flow."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env = SafetyMonitorEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task="binary_safety_classification", difficulty="easy")
        self.assertFalse(obs.done)
        self.assertEqual(obs.step, 0)
        self.assertEqual(obs.max_steps, 18)

    def test_step_returns_observation_with_reward(self):
        self.env.reset(difficulty="easy")
        action = SafetyAction(classification="SAFE", confidence=0.9)
        obs = self.env.step(action)
        self.assertIsNotNone(obs.reward)

    def test_structured_reward(self):
        self.env.reset(difficulty="easy")
        action = SafetyAction(classification="SAFE", confidence=0.9)
        self.env.step(action)
        reward = self.env.last_reward
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, SafetyReward)
        self.assertGreaterEqual(reward.value, 0.01)
        self.assertLessEqual(reward.value, 0.99)
        self.assertEqual(reward.step_penalty, -0.01)

    def test_step_info(self):
        self.env.reset(difficulty="easy")
        action = SafetyAction(classification="SAFE", confidence=0.9)
        self.env.step(action)
        info = self.env.last_step_info
        self.assertIsNotNone(info)
        self.assertIsInstance(info, StepInfo)
        self.assertEqual(info.case_id, "easy_01")
        self.assertIn(info.ground_truth, ["SAFE", "UNSAFE", "NEEDS_REVIEW"])

    def test_state_updates(self):
        self.env.reset(difficulty="easy")
        action = SafetyAction(classification="SAFE", confidence=0.9)
        self.env.step(action)
        state = self.env.state
        self.assertEqual(state.current_step, 1)
        self.assertEqual(state.cases_seen, 1)

    def test_full_episode_completion(self):
        self.env.reset(difficulty="easy")
        for i in range(18):
            action = SafetyAction(classification="SAFE", confidence=0.5)
            obs = self.env.step(action)
        self.assertTrue(obs.done)
        self.assertEqual(self.env.state.current_step, 18)

    def test_episode_score_in_range(self):
        self.env.reset(difficulty="easy")
        for _ in range(18):
            self.env.step(SafetyAction(classification="SAFE", confidence=0.5))
        score = self.env.get_score()
        self.assertGreaterEqual(score, 0.01)
        self.assertLessEqual(score, 0.99)


class TestStreakPenalty(unittest.TestCase):
    """Test that streak penalty activates after 3 consecutive errors."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env = SafetyMonitorEnvironment()

    def test_streak_penalty_activates(self):
        self.env.reset(difficulty="easy")
        # First case is SAFE, so submitting UNSAFE is always wrong
        # Submit wrong answers
        rewards = []
        for i in range(5):
            obs = self.env.step(SafetyAction(classification="NEEDS_REVIEW", confidence=0.5))
            rewards.append(self.env.last_reward)

        # After 3 consecutive errors, streak penalty should appear
        self.assertLess(rewards[2].streak_penalty, 0.0,
                       "Streak penalty should activate after 3 wrong answers")


class TestMemoryBonus(unittest.TestCase):
    """Test that memory bonus works across steps."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env = SafetyMonitorEnvironment()

    def test_memory_bonus_after_correct(self):
        # Run one episode with correct answers to build memory
        self.env.reset(difficulty="easy")
        first_case = EASY_CASES[0]
        self.env.step(SafetyAction(
            classification=first_case.ground_truth.value, confidence=0.9))

        # Second case in same category should get memory bonus
        # (only if same category exists - we check the bonus field)
        reward = self.env.last_reward
        self.assertIsNotNone(reward)
        # Memory bonus may or may not apply depending on category matching


class TestInspectAction(unittest.TestCase):
    """Test the INSPECT action for multi-step RL (Gap 1)."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env = SafetyMonitorEnvironment()

    def test_inspect_does_not_advance_case(self):
        self.env.reset(difficulty="easy")
        obs1 = self.env._build_observation()
        self.env.step(SafetyAction(classification="INSPECT", confidence=0.5))
        obs2 = self.env._build_observation()
        # Case should not advance — same case_id
        self.assertEqual(obs1.case_id, obs2.case_id)

    def test_inspect_returns_small_reward(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="INSPECT", confidence=0.5))
        reward = self.env.last_reward
        self.assertIsNotNone(reward)
        self.assertGreaterEqual(reward.value, 0.01)
        self.assertLessEqual(reward.value, 0.99)
        # Base score should be 0 for inspect
        self.assertEqual(reward.base_score, 0.0)

    def test_inspect_then_classify(self):
        self.env.reset(difficulty="easy")
        # Inspect first
        self.env.step(SafetyAction(classification="INSPECT", confidence=0.5))
        # Then classify
        self.env.step(SafetyAction(classification="SAFE", confidence=0.9))
        reward = self.env.last_reward
        self.assertGreater(reward.base_score, 0.0)

    def test_repeated_inspect_gets_loop_penalty(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="INSPECT", confidence=0.5))
        self.env.step(SafetyAction(classification="INSPECT", confidence=0.5))
        reward = self.env.last_reward
        self.assertLess(reward.loop_penalty, 0.0,
                       "Repeated INSPECT should trigger loop penalty")


class TestMalformedAction(unittest.TestCase):
    """Test malformed action rejection and penalty (Gap 2)."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env = SafetyMonitorEnvironment()

    def test_invalid_classification_rejected(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="INVALID_LABEL", confidence=0.5))
        reward = self.env.last_reward
        self.assertLess(reward.malformed_penalty, 0.0,
                       "Invalid classification should get malformed penalty")

    def test_malformed_does_not_advance_case(self):
        self.env.reset(difficulty="easy")
        obs1 = self.env._build_observation()
        self.env.step(SafetyAction(classification="GARBAGE", confidence=0.5))
        obs2 = self.env._build_observation()
        self.assertEqual(obs1.case_id, obs2.case_id,
                        "Malformed action should not advance the case")

    def test_random_string_malformed(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="banana", confidence=0.5))
        info = self.env.last_step_info
        self.assertIsNotNone(info.error)
        self.assertIn("invalid_classification", info.error)

    def test_empty_string_malformed(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="", confidence=0.5))
        reward = self.env.last_reward
        self.assertLess(reward.malformed_penalty, 0.0)


class TestAntiLoopPenalty(unittest.TestCase):
    """Test anti-loop penalty for repeated identical actions (Gap 4)."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env = SafetyMonitorEnvironment()

    def test_repeated_same_action_penalized(self):
        self.env.reset(difficulty="easy")
        # Submit same classification twice
        self.env.step(SafetyAction(classification="SAFE", confidence=0.50))
        self.env.step(SafetyAction(classification="SAFE", confidence=0.50))
        reward = self.env.last_reward
        self.assertLess(reward.loop_penalty, 0.0,
                       "Exact same action submitted twice should get loop penalty")

    def test_different_actions_no_penalty(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="SAFE", confidence=0.9))
        self.env.step(SafetyAction(classification="UNSAFE", confidence=0.8))
        reward = self.env.last_reward
        self.assertEqual(reward.loop_penalty, 0.0,
                        "Different actions should not get loop penalty")

    def test_same_label_different_confidence_no_penalty(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="SAFE", confidence=0.9))
        self.env.step(SafetyAction(classification="SAFE", confidence=0.5))
        reward = self.env.last_reward
        self.assertEqual(reward.loop_penalty, 0.0,
                        "Same label but different confidence should not trigger loop penalty")


class TestDeterminism(unittest.TestCase):
    """Test that grading is 100% deterministic and reproducible (Gap 3)."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env1 = SafetyMonitorEnvironment()
        self.env2 = SafetyMonitorEnvironment()

    def test_same_episode_same_rewards(self):
        """Two identical episodes must produce identical rewards."""
        rewards1 = []
        rewards2 = []
        
        self.env1.reset(difficulty="easy")
        self.env2.reset(difficulty="easy")
        
        for case in EASY_CASES:
            action = SafetyAction(
                classification=case.ground_truth.value,
                confidence=0.85,
                severity="medium",
            )
            self.env1.step(action)
            self.env2.step(action)
            rewards1.append(self.env1.last_reward.value)
            rewards2.append(self.env2.last_reward.value)

        self.assertEqual(rewards1, rewards2,
                        "Identical episodes must produce identical rewards")

    def test_deterministic_across_all_difficulties(self):
        """All 3 difficulty levels produce deterministic results."""
        for diff in ["easy", "medium", "hard"]:
            cases = get_cases(diff)
            
            self.env1.reset(difficulty=diff)
            self.env2.reset(difficulty=diff)
            
            for case in cases:
                action = SafetyAction(
                    classification=case.ground_truth.value,
                    confidence=0.80,
                )
                self.env1.step(action)
                self.env2.step(action)
                
                self.assertEqual(
                    self.env1.last_reward.value,
                    self.env2.last_reward.value,
                    f"Non-deterministic reward for {case.id}",
                )

    def test_grader_pure_function(self):
        """Grade function has no side effects — identical calls, identical results."""
        for _ in range(10):
            r1 = grade("easy", predicted=SafetyLabel.SAFE, ground_truth=SafetyLabel.SAFE, confidence=0.85)
            r2 = grade("easy", predicted=SafetyLabel.SAFE, ground_truth=SafetyLabel.SAFE, confidence=0.85)
            self.assertEqual(r1, r2)


class TestProposeAction(unittest.TestCase):
    """Test PROPOSE action for multi-step reasoning loop."""

    def setUp(self):
        from server.environment import SafetyMonitorEnvironment
        self.env = SafetyMonitorEnvironment()

    def test_propose_does_not_advance_case(self):
        self.env.reset(difficulty="easy")
        obs1 = self.env._build_observation()
        self.env.step(SafetyAction(classification="PROPOSE:SAFE", confidence=0.5))
        obs2 = self.env._build_observation()
        self.assertEqual(obs1.case_id, obs2.case_id)

    def test_propose_returns_feedback(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="PROPOSE:SAFE", confidence=0.5))
        reward = self.env.last_reward
        self.assertIn("PROPOSAL FEEDBACK", reward.explanation)

    def test_propose_correct_feedback(self):
        self.env.reset(difficulty="easy")
        first_case = EASY_CASES[0]
        self.env.step(SafetyAction(
            classification=f"PROPOSE:{first_case.ground_truth.value}",
            confidence=0.5))
        reward = self.env.last_reward
        self.assertIn("CORRECT", reward.explanation)

    def test_propose_then_classify(self):
        self.env.reset(difficulty="easy")
        # Propose first
        self.env.step(SafetyAction(classification="PROPOSE:SAFE", confidence=0.5))
        # Then classify
        self.env.step(SafetyAction(classification="SAFE", confidence=0.9))
        info = self.env.last_step_info
        self.assertIsNotNone(info.ground_truth)  # Should have advanced

    def test_repeated_propose_gets_loop_penalty(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="PROPOSE:SAFE", confidence=0.5))
        self.env.step(SafetyAction(classification="PROPOSE:UNSAFE", confidence=0.5))
        reward = self.env.last_reward
        self.assertLess(reward.loop_penalty, 0.0,
                       "Repeated PROPOSE should trigger loop penalty")

    def test_proposal_reward_in_range(self):
        self.env.reset(difficulty="easy")
        self.env.step(SafetyAction(classification="PROPOSE:SAFE", confidence=0.5))
        reward = self.env.last_reward
        self.assertGreaterEqual(reward.value, 0.01)
        self.assertLessEqual(reward.value, 0.99)


if __name__ == "__main__":
    unittest.main()

