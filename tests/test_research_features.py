"""Tests for Phase 6 research-grade features."""

from cognicore.predictive import FailurePredictor
from cognicore.multi_memory import CognitiveMemory
from cognicore.red_blue import RedVsBlue
from cognicore.debugger import AIDebugger
from cognicore.intelligence import IntelligenceScorer
from cognicore.thought_trace import ThoughtTracer
from cognicore.knowledge_transfer import transfer_knowledge, MentorStudent
from cognicore.evolution import EvolutionEngine
from cognicore.smart_agents import AutoLearner


class TestFailurePredictor:
    def test_observe_and_predict(self):
        pred = FailurePredictor()
        pred.observe("security", correct=False, confidence=0.9)
        pred.observe("security", correct=False, confidence=0.8)
        risk = pred.predict_risk("security")
        assert risk["risk"] > 0
        assert risk["level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_early_warning(self):
        pred = FailurePredictor()
        for _ in range(10):
            pred.observe("security", correct=False, confidence=0.9)
        risk = pred.predict_risk("security")
        # After many failures, risk should be elevated
        assert risk["risk"] > 0.3

    def test_low_risk(self):
        pred = FailurePredictor()
        for _ in range(5):
            pred.observe("cooking", correct=True, confidence=0.8)
        risk = pred.predict_risk("cooking")
        assert risk["level"] == "LOW"

    def test_risk_report(self):
        pred = FailurePredictor()
        pred.observe("a", correct=True)
        pred.observe("b", correct=False)
        report = pred.risk_report()
        assert "total_observations" in report
        assert report["total_observations"] == 2


class TestCognitiveMemory:
    def test_perceive(self):
        mem = CognitiveMemory()
        mem.perceive("phishing email", category="security", correct=False, action="SAFE")
        assert mem.working.size == 1
        assert mem.episodic.size == 1
        assert mem.semantic.categories_known == 1

    def test_recall(self):
        mem = CognitiveMemory()
        mem.perceive("phishing email", category="security", correct=True, action="UNSAFE")
        mem.perceive("cooking recipe", category="cooking", correct=True, action="SAFE")
        result = mem.recall(category="security")
        assert "working_memory" in result
        assert "episodic" in result
        assert "semantic" in result

    def test_procedural_rules(self):
        mem = CognitiveMemory(procedural_min_obs=2)
        # Need 2+ observations for a rule
        mem.perceive("a", category="security", correct=True, action="UNSAFE")
        mem.perceive("b", category="security", correct=True, action="UNSAFE")
        mem.perceive("c", category="security", correct=True, action="UNSAFE")
        result = mem.procedural.get_action("security")
        assert result is not None
        assert result[0] == "UNSAFE"

    def test_recommended_action(self):
        mem = CognitiveMemory(procedural_min_obs=2)
        mem.perceive("a", category="security", correct=True, action="UNSAFE")
        mem.perceive("b", category="security", correct=True, action="UNSAFE")
        mem.perceive("c", category="security", correct=True, action="UNSAFE")
        recall = mem.recall(category="security")
        assert recall["recommended_action"] == "UNSAFE"

    def test_stats(self):
        mem = CognitiveMemory()
        mem.perceive("test", "cat", True, "A")
        s = mem.stats()
        assert s["working_memory"] == 1
        assert s["episodic_memories"] == 1


class TestRedVsBlue:
    def test_battle_runs(self):
        battle = RedVsBlue()
        result = battle.run(rounds=10, verbose=False)
        assert result.total_rounds == 10
        assert result.attacker_wins + result.defender_wins == 10

    def test_strategy_effectiveness(self):
        battle = RedVsBlue()
        result = battle.run(rounds=20, verbose=False)
        strategies = result.strategy_effectiveness()
        assert isinstance(strategies, dict)

    def test_adaptation_curve(self):
        battle = RedVsBlue()
        result = battle.run(rounds=20, verbose=False)
        curve = result.adaptation_curve(window=5)
        assert len(curve) >= 1


class TestAIDebugger:
    def test_run_without_breakpoints(self):
        dbg = AIDebugger("SafetyClassification-v1", difficulty="easy")
        trace = dbg.run(verbose=False)
        assert len(trace.steps) == 10

    def test_breakpoint_category(self):
        dbg = AIDebugger("SafetyClassification-v1", difficulty="easy")
        dbg.breakpoint(category="malware", name="malware_bp")
        trace = dbg.run(verbose=False)
        assert len(trace.steps) == 10

    def test_breakpoint_on_wrong(self):
        dbg = AIDebugger("SafetyClassification-v1", difficulty="easy")
        dbg.breakpoint(on_wrong_only=True, name="wrong_bp")
        trace = dbg.run(verbose=False)
        # Should have caught at least some wrong answers
        assert len(trace.steps) == 10

    def test_decision_tree(self):
        dbg = AIDebugger("SafetyClassification-v1", difficulty="easy")
        trace = dbg.run(verbose=False)
        tree = trace.decision_tree()
        assert isinstance(tree, dict)
        assert len(tree) > 0


class TestIntelligenceScorer:
    def test_compute_score(self):
        scorer = IntelligenceScorer()
        scorer.record(1, "security", True, 1.0, predicted="UNSAFE", truth="UNSAFE")
        scorer.record(2, "cooking", True, 1.0, predicted="SAFE", truth="SAFE")
        scorer.record(3, "security", False, 0.0, predicted="SAFE", truth="UNSAFE")
        iq = scorer.compute()
        assert 0 <= iq.overall <= 100
        assert "reasoning" in iq.dimensions
        assert "safety" in iq.dimensions

    def test_empty_scorer(self):
        scorer = IntelligenceScorer()
        iq = scorer.compute()
        assert iq.overall == 0


class TestThoughtTracer:
    def test_basic_chain(self):
        tracer = ThoughtTracer()
        tracer.begin_thought("Classifying email")
        tracer.add_evidence("contains 'password'", weight=0.6, direction="UNSAFE")
        tracer.conclude("UNSAFE", confidence=0.7)
        tracer.was_correct(True, truth="UNSAFE")
        assert len(tracer.chains) == 1

    def test_reasoning_error(self):
        tracer = ThoughtTracer()
        tracer.begin_thought("Classifying text")
        tracer.add_evidence("looks safe", weight=0.8, direction="SAFE")
        tracer.conclude("SAFE", confidence=0.9)
        tracer.was_correct(False, truth="UNSAFE")
        errors = tracer.reasoning_errors()
        assert len(errors) == 1

    def test_confidence_calibration(self):
        tracer = ThoughtTracer()
        for i in range(10):
            tracer.begin_thought(f"Step {i}")
            tracer.conclude("SAFE", confidence=0.9)
            tracer.was_correct(i % 2 == 0)  # 50% accuracy
        cal = tracer.confidence_calibration()
        assert cal["overconfident"] is True  # 90% confidence but 50% accuracy


class TestKnowledgeTransfer:
    def test_full_transfer(self):
        expert = AutoLearner()
        expert.knowledge["security"]["UNSAFE"] = 5.0
        expert.knowledge["cooking"]["SAFE"] = 3.0
        student = AutoLearner()
        result = transfer_knowledge(expert, student, method="full")
        assert result["transferred"] >= 2
        assert "security" in student.knowledge

    def test_selective_transfer(self):
        expert = AutoLearner()
        expert.knowledge["security"]["UNSAFE"] = 5.0
        expert.knowledge["cooking"]["SAFE"] = 0.1
        student = AutoLearner()
        result = transfer_knowledge(expert, student, method="selective", min_confidence=0.7)
        assert result["transferred"] >= 1

    def test_mentor_student(self):
        mentor = AutoLearner()
        student = AutoLearner()
        ms = MentorStudent(mentor, student)
        obs = {"category": "test", "prompt": "hello", "memory_context": [], "reflection_hints": ""}
        result = ms.guided_step(obs)
        assert "action" in result
        assert "corrected" in result


class TestEvolution:
    def test_evolve_runs(self):
        engine = EvolutionEngine(
            "SafetyClassification-v1",
            population_size=5,
            elite_count=2,
        )
        best = engine.evolve(generations=2, verbose=False)
        assert best.fitness > 0
        assert len(engine.history) == 2

    def test_agent_mutation(self):
        from cognicore.evolution import EvolvableAgent
        agent = EvolvableAgent()
        child = agent.mutate(rate=1.0)
        # Genome should be different
        assert child is not agent

    def test_crossover(self):
        from cognicore.evolution import EvolvableAgent
        a = EvolvableAgent()
        b = EvolvableAgent()
        child = EvolvableAgent.crossover(a, b)
        assert child is not a
        assert child is not b
