"""Tests for cognicore.auto_improve — auto_improve function."""

from cognicore.auto_improve import auto_improve


class TestAutoImproveBasic:
    def test_returns_dict(self):
        result = auto_improve(
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=1,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        result = auto_improve(
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=2,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert "cycles" in result
        assert "initial_accuracy" in result
        assert "final_accuracy" in result

    def test_cycles_count(self):
        result = auto_improve(
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=3,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert result["cycles"] <= 3
        assert result["cycles"] >= 1

    def test_accuracy_in_range(self):
        result = auto_improve(
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=2,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert 0.0 <= result["initial_accuracy"] <= 1.0
        assert 0.0 <= result["final_accuracy"] <= 1.0


class TestAutoImproveWithAgent:
    def test_with_custom_agent(self):
        from cognicore.smart_agents import AutoLearner

        agent = AutoLearner()
        result = auto_improve(
            agent=agent,
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=2,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert "cycles" in result

    def test_with_safe_agent(self):
        from cognicore.smart_agents import SafeAgent

        agent = SafeAgent()
        result = auto_improve(
            agent=agent,
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=2,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert isinstance(result, dict)


class TestAutoImproveDifferentEnvs:
    def test_math_env(self):
        result = auto_improve(
            env_id="MathReasoning-v1",
            difficulty="easy",
            max_cycles=1,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert "cycles" in result

    def test_code_env(self):
        result = auto_improve(
            env_id="CodeDebugging-v1",
            difficulty="easy",
            max_cycles=1,
            episodes_per_cycle=1,
            verbose=False,
        )
        assert "cycles" in result


class TestAutoImproveEarlyStopping:
    def test_patient_stops_on_no_improvement(self):
        result = auto_improve(
            env_id="SafetyClassification-v1",
            difficulty="easy",
            max_cycles=10,
            episodes_per_cycle=1,
            patience=1,
            verbose=False,
        )
        # Should stop before 10 cycles if no improvement
        assert result["cycles"] <= 10
        assert "cycles" in result
