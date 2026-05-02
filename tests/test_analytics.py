"""Tests for cognicore.analytics — PerformanceAnalyzer, AnalyticsReport."""

import cognicore
from cognicore.analytics import PerformanceAnalyzer


class TestPerformanceAnalyzerRecord:
    def test_record_step(self):
        analyzer = PerformanceAnalyzer()
        analyzer.record_step(
            step=1,
            episode=1,
            category="security",
            correct=True,
            reward_total=1.0,
        )
        assert len(analyzer._step_data) == 1

    def test_record_multiple_steps(self):
        analyzer = PerformanceAnalyzer()
        for i in range(5):
            analyzer.record_step(
                step=i + 1,
                episode=1,
                category="security",
                correct=(i % 2 == 0),
                reward_total=0.5,
            )
        assert len(analyzer._step_data) == 5

    def test_record_episode(self):
        analyzer = PerformanceAnalyzer()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})
        analyzer.record_episode(env, episode_num=1)
        assert len(analyzer.episodes) == 1

    def test_record_episode_data(self):
        analyzer = PerformanceAnalyzer()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        env.reset()
        env.step({"classification": "SAFE"})
        analyzer.record_episode(env, episode_num=1)
        ep = analyzer.episodes[0]
        assert "accuracy" in ep
        assert "score" in ep
        assert ep["episode"] == 1


class TestAnalyticsReportLearningCurve:
    def test_learning_curve_empty(self):
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze()
        curve = report.learning_curve()
        assert curve == []

    def test_learning_curve_with_data(self):
        analyzer = PerformanceAnalyzer()
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        for ep in range(3):
            env.reset()
            for _ in range(10):
                obs, reward, done, _, info = env.step({"classification": "SAFE"})
                if done:
                    break
            analyzer.record_episode(env, episode_num=ep + 1)

        report = analyzer.analyze()
        curve = report.learning_curve()
        assert len(curve) == 3
        for point in curve:
            assert "episode" in point
            assert "accuracy" in point
            assert "score" in point

    def test_is_improving_insufficient_data(self):
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze()
        assert report.is_improving() is False


class TestAnalyticsReportWeakCategories:
    def test_weak_categories_empty(self):
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze()
        weak = report.weak_categories()
        assert weak == []

    def test_weak_categories_with_data(self):
        analyzer = PerformanceAnalyzer()
        for i in range(5):
            analyzer.record_step(
                step=i + 1,
                episode=1,
                category="security",
                correct=False,
                reward_total=0.0,
            )
        for i in range(5):
            analyzer.record_step(
                step=i + 6,
                episode=1,
                category="cooking",
                correct=True,
                reward_total=1.0,
            )
        report = analyzer.analyze()
        weak = report.weak_categories()
        assert len(weak) >= 1
        # Security should be weaker
        assert weak[0]["category"] == "security"

    def test_strong_categories(self):
        analyzer = PerformanceAnalyzer()
        for i in range(5):
            analyzer.record_step(
                step=i + 1,
                episode=1,
                category="cooking",
                correct=True,
                reward_total=1.0,
            )
        for i in range(5):
            analyzer.record_step(
                step=i + 6,
                episode=1,
                category="security",
                correct=False,
                reward_total=0.0,
            )
        report = analyzer.analyze()
        strong = report.strong_categories()
        assert len(strong) >= 1
        assert strong[0]["category"] == "cooking"


class TestAnalyticsReportMemoryImpact:
    def test_memory_impact_empty(self):
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze()
        impact = report.memory_impact()
        assert "memory_total" in impact
        assert "memory_pct_of_reward" in impact

    def test_memory_impact_with_bonus(self):
        analyzer = PerformanceAnalyzer()
        analyzer.record_step(
            step=1,
            episode=1,
            category="security",
            correct=True,
            reward_total=1.1,
            memory_bonus=0.1,
        )
        report = analyzer.analyze()
        impact = report.memory_impact()
        assert impact["memory_total"] == 0.1

    def test_print_insights_runs(self):
        analyzer = PerformanceAnalyzer()
        analyzer.record_step(1, 1, "security", True, 1.0)
        report = analyzer.analyze()
        # Should not raise
        report.print_insights()
