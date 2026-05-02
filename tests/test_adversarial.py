"""Tests for cognicore.adversarial — AdversarialTester."""

from cognicore.adversarial import AdversarialTester


class TestAdversarialTesterInit:
    def test_create_default(self):
        tester = AdversarialTester()
        assert tester.env_id == "SafetyClassification-v1"
        assert tester.difficulty == "hard"

    def test_create_with_env_id(self):
        tester = AdversarialTester("SafetyClassification-v1")
        assert tester.env_id == "SafetyClassification-v1"

    def test_create_with_difficulty(self):
        tester = AdversarialTester("SafetyClassification-v1", difficulty="easy")
        assert tester.difficulty == "easy"


class TestAdversarialAttackTemplates:
    def test_has_prompt_attacks(self):
        assert len(AdversarialTester.PROMPT_ATTACKS) > 0

    def test_prompt_attacks_are_strings(self):
        for attack in AdversarialTester.PROMPT_ATTACKS:
            assert isinstance(attack, str)

    def test_edge_cases_exist(self):
        assert len(AdversarialTester.EDGE_CASES) > 0


class TestAdversarialBreakMyAgent:
    def test_break_my_agent_no_agent(self):
        tester = AdversarialTester("SafetyClassification-v1")
        failures = tester.break_my_agent(None, max_attempts=2)
        assert isinstance(failures, list)

    def test_break_my_agent_returns_list(self):
        tester = AdversarialTester("SafetyClassification-v1")
        failures = tester.break_my_agent(None, max_attempts=3)
        assert isinstance(failures, list)


class TestAdversarialStressTest:
    def test_stress_test_basic(self):
        tester = AdversarialTester("SafetyClassification-v1")
        report = tester.stress_test(None, rounds=2, verbose=False)
        assert report.injection_resistance >= 0
        assert report.edge_case_handling >= 0
        assert report.stress_stability >= 0

    def test_stress_test_vulnerabilities(self):
        tester = AdversarialTester("SafetyClassification-v1")
        report = tester.stress_test(None, rounds=2, verbose=False)
        vulns = report.vulnerabilities()
        assert isinstance(vulns, list)

    def test_stress_test_scores_in_range(self):
        tester = AdversarialTester("SafetyClassification-v1")
        report = tester.stress_test(None, rounds=2, verbose=False)
        assert 0 <= report.injection_resistance <= 100
        assert 0 <= report.edge_case_handling <= 100
        assert 0 <= report.stress_stability <= 100
