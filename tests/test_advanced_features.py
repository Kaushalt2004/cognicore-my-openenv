"""Tests for Curriculum, Benchmark, and Pipeline (Phase 3/4 features)."""

from cognicore.curriculum import CurriculumRunner
from cognicore.benchmark import benchmark_agent, BenchmarkResult
from cognicore.compose import Pipeline


class TestCurriculum:
    def test_create_runner(self):
        runner = CurriculumRunner("SafetyClassification-v1")
        assert runner.difficulty == "easy"
        assert runner.current_level == 0

    def test_run_episodes(self):
        runner = CurriculumRunner("SafetyClassification-v1")
        result = runner.run(max_episodes=3, verbose=False)
        assert result["episodes"] == 3
        assert len(result["history"]) == 3
        assert "avg_accuracy" in result
        assert "avg_score" in result

    def test_difficulty_levels(self):
        CurriculumRunner("SafetyClassification-v1")
        assert CurriculumRunner.LEVELS == ["easy", "medium", "hard"]

    def test_callbacks(self):
        events = []
        runner = CurriculumRunner(
            "SafetyClassification-v1",
            promotion_threshold=0.0,  # promote immediately
            window=1,
        )
        runner.on_level_change(lambda t, o, n: events.append((t, o, n)))
        runner.run(max_episodes=3, verbose=False)
        # Should have promoted at least once
        assert len(events) >= 1
        assert events[0][0] == "promote"


class TestBenchmark:
    def test_benchmark_single_env(self):
        result = benchmark_agent(
            envs=["SafetyClassification-v1"],
            difficulties=["easy"],
            episodes=1,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.results) == 1
        assert result.overall_accuracy >= 0
        assert result.overall_score >= 0

    def test_benchmark_by_env(self):
        result = benchmark_agent(
            envs=["SafetyClassification-v1", "MathReasoning-v1"],
            difficulties=["easy"],
            episodes=1,
            verbose=False,
        )
        by_env = result.by_env()
        assert "SafetyClassification-v1" in by_env
        assert "MathReasoning-v1" in by_env

    def test_benchmark_by_difficulty(self):
        result = benchmark_agent(
            envs=["SafetyClassification-v1"],
            difficulties=["easy", "medium"],
            episodes=1,
            verbose=False,
        )
        by_diff = result.by_difficulty()
        assert "easy" in by_diff
        assert "medium" in by_diff

    def test_benchmark_to_dict(self):
        result = benchmark_agent(
            envs=["SafetyClassification-v1"],
            difficulties=["easy"],
            episodes=1,
            verbose=False,
        )
        d = result.to_dict()
        assert "agent_id" in d
        assert "overall_accuracy" in d
        assert "by_env" in d


class TestPipeline:
    def test_create_pipeline(self):
        pipe = Pipeline(
            [
                ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
                ("math", "MathReasoning-v1", {"difficulty": "easy"}),
            ]
        )
        assert len(pipe.stage_defs) == 2
        assert pipe.done is False

    def test_pipeline_reset(self):
        pipe = Pipeline(
            [
                ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
            ]
        )
        obs = pipe.reset()
        assert "_pipeline_stage" in obs
        assert obs["_pipeline_stage"] == "safety"

    def test_pipeline_run_through(self):
        pipe = Pipeline(
            [
                ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
                ("math", "MathReasoning-v1", {"difficulty": "easy"}),
            ]
        )
        obs = pipe.reset()

        steps = 0
        while not pipe.done:
            action = {"classification": "SAFE", "answer": 42}
            obs, reward, done, _, info = pipe.step(action)
            steps += 1
            if steps > 25:  # safety limit
                break

        assert steps == 20  # 10 per env
        assert pipe.done is True

    def test_pipeline_report(self):
        pipe = Pipeline(
            [
                ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
            ]
        )
        obs = pipe.reset()
        for _ in range(10):
            obs, reward, done, _, info = pipe.step({"classification": "SAFE"})

        report = pipe.report()
        assert report["stages_completed"] == 1
        assert "overall_accuracy" in report
        assert len(report["stages"]) == 1

    def test_pipeline_shared_memory(self):
        pipe = Pipeline(
            [
                ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
                ("math", "MathReasoning-v1", {"difficulty": "easy"}),
            ],
            share_memory=True,
        )
        obs = pipe.reset()

        # Run through safety stage
        for _ in range(10):
            obs, reward, done, _, info = pipe.step(
                {"classification": "SAFE", "answer": 42}
            )

        # After stage 1, math env should have memory from safety
        math_env = pipe._stages[1]
        assert len(math_env.memory.entries) > 0

    def test_pipeline_no_shared_memory(self):
        pipe = Pipeline(
            [
                ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
                ("math", "MathReasoning-v1", {"difficulty": "easy"}),
            ],
            share_memory=False,
        )
        obs = pipe.reset()

        for _ in range(10):
            obs, reward, done, _, info = pipe.step(
                {"classification": "SAFE", "answer": 42}
            )

        # Math env should NOT have safety memories
        math_env = pipe._stages[1]
        safety_entries = [
            e
            for e in math_env.memory.entries
            if e.get("category", "").startswith("malw")
        ]
        assert len(safety_entries) == 0
