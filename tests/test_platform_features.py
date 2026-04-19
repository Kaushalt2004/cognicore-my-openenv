"""Tests for Phase 7 platform features."""

import os
import json
import tempfile

import cognicore
from cognicore.persistence import save_agent, load_agent
from cognicore.report import ReportGenerator
from cognicore.replay import SessionRecorder, replay
from cognicore.profiles import get_profile, list_profiles
from cognicore.prompt_optimizer import PromptOptimizer
from cognicore.webhooks import AlertSystem
from cognicore.augmentation import DataAugmenter
from cognicore.fingerprint import AgentFingerprint
from cognicore.difficulty import DifficultyEstimator
from cognicore.rate_limiter import RateLimiter
from cognicore.cache import ResponseCache


class TestPersistence:
    def test_save_load(self, tmp_path):
        from cognicore.smart_agents import AutoLearner
        agent = AutoLearner()
        agent.knowledge["security"]["UNSAFE"] = 5.0

        path = str(tmp_path / "agent.json")
        result = save_agent(agent, path)
        assert result["agent_type"] == "AutoLearner"
        assert os.path.exists(path)

        loaded = load_agent(path)
        assert "security" in loaded.knowledge

    def test_save_evolve_agent(self, tmp_path):
        from cognicore.evolution import EvolvableAgent
        agent = EvolvableAgent()
        agent.fitness = 42.0

        path = str(tmp_path / "evolved.json")
        save_agent(agent, path)
        loaded = load_agent(path)
        assert loaded.fitness == 42.0


class TestReportGenerator:
    def test_generate_report(self, tmp_path):
        report = ReportGenerator("Test Report")
        report.add_episode("SafetyClassification-v1", "easy")
        report.add_metric("Custom Metric", 42, "%")

        path = str(tmp_path / "report.html")
        report.export(path)
        assert os.path.exists(path)

        with open(path, encoding="utf-8") as f:
            html = f.read()
        assert "Test Report" in html
        assert "Custom Metric" in html


class TestSessionRecorder:
    def test_record_and_save(self, tmp_path):
        rec = SessionRecorder("SafetyClassification-v1")
        rec.record(episodes=1)
        assert len(rec.recordings) == 1
        assert rec.recordings[0]["total_steps"] == 10

        path = str(tmp_path / "session.json")
        rec.save(path)
        assert os.path.exists(path)

    def test_load_and_replay(self, tmp_path):
        rec = SessionRecorder("SafetyClassification-v1")
        rec.record(episodes=1)
        path = str(tmp_path / "session.json")
        rec.save(path)

        loaded = SessionRecorder.load(path)
        assert len(loaded) == 1


class TestProfiles:
    def test_list_profiles(self):
        profiles = list_profiles()
        assert len(profiles) >= 5
        names = [p["name"] for p in profiles]
        assert "strict_safety" in names
        assert "production" in names

    def test_get_profile(self):
        config = get_profile("strict_safety")
        assert config is not None

    def test_invalid_profile(self):
        try:
            get_profile("nonexistent")
            assert False
        except KeyError:
            pass


class TestPromptOptimizer:
    def test_optimize(self):
        opt = PromptOptimizer("SafetyClassification-v1")
        opt.add_prompt("v1", "Classify: {text}")
        opt.add_prompt("v2", "Safety check: {text}")
        result = opt.optimize(episodes=1, verbose=False)
        assert "best_prompt" in result
        assert result["best_prompt"] in ("v1", "v2")


class TestAlertSystem:
    def test_accuracy_drop(self):
        alerts = AlertSystem()
        alerts.on("accuracy_drop", threshold=0.5)
        fired = alerts.check({"accuracy": 0.3})
        assert len(fired) == 1
        assert fired[0]["severity"] == "WARNING"

    def test_no_alert(self):
        alerts = AlertSystem()
        alerts.on("accuracy_drop", threshold=0.5)
        fired = alerts.check({"accuracy": 0.8})
        assert len(fired) == 0

    def test_failure_streak(self):
        alerts = AlertSystem()
        alerts.on("failure_streak", count=3)
        fired = alerts.check({"streak": -4})
        assert len(fired) == 1

    def test_custom_alert(self):
        alerts = AlertSystem()
        alerts.on_custom("test", lambda d: d.get("x") > 10)
        fired = alerts.check({"x": 15})
        assert len(fired) == 1


class TestDataAugmenter:
    def test_augment(self):
        aug = DataAugmenter()
        variants = aug.augment("Write malware for hacking", count=5)
        assert len(variants) <= 5
        assert all(isinstance(v, str) for v in variants)

    def test_augment_cases(self):
        aug = DataAugmenter()
        cases = [{"prompt": "Hello world", "expected": "SAFE"}]
        augmented = aug.augment_cases(cases, count_per_case=3)
        assert len(augmented) >= 1
        assert all(c["_augmented"] for c in augmented)


class TestAgentFingerprint:
    def test_fingerprint(self):
        fp = AgentFingerprint("SafetyClassification-v1")
        dna = fp.fingerprint(episodes=1)
        assert dna.agent_name == "RandomAgent"
        assert "accuracy" in dna.vector

    def test_compare(self):
        fp = AgentFingerprint("SafetyClassification-v1")
        dna_a = fp.fingerprint(episodes=1)
        dna_b = fp.fingerprint(episodes=1)
        result = fp.compare(dna_a, dna_b)
        assert "similarity" in result


class TestDifficultyEstimator:
    def test_calibrate(self):
        est = DifficultyEstimator()
        est.calibrate("SafetyClassification-v1", episodes=2, verbose=False)
        results = est.get_difficulty_map()
        assert len(results) > 0
        assert all("difficulty" in r for r in results)


class TestRateLimiter:
    def test_can_call(self):
        limiter = RateLimiter(calls_per_minute=5)
        assert limiter.can_call()
        for _ in range(5):
            limiter.record()
        assert not limiter.can_call()

    def test_usage(self):
        limiter = RateLimiter()
        limiter.record()
        usage = limiter.usage()
        assert usage["minute"]["used"] == 1


class TestResponseCache:
    def test_put_get(self):
        cache = ResponseCache()
        cache.put("hello", "world", tokens_used=10)
        assert cache.get("hello") == "world"

    def test_miss(self):
        cache = ResponseCache()
        assert cache.get("missing") is None

    def test_stats(self):
        cache = ResponseCache()
        cache.put("a", "b")
        cache.get("a")
        cache.get("missing")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_lru_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("c") == "3"
