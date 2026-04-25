"""Tests for Phase 8 — Roadmap features."""

import os
import cognicore
from cognicore.meta_rewards import MetaRewardOptimizer
from cognicore.causal import CausalEngine
from cognicore.agent_builder import build_agent, describe_agent
from cognicore.strategy import StrategySwitcher
from cognicore.lifelong import LifelongAgent
from cognicore.swarm import Swarm


class TestMetaRewardOptimizer:
    def test_observe_and_optimize(self):
        meta = MetaRewardOptimizer()
        meta.observe({"memory_bonus": 0.05, "streak_penalty": -0.1}, accuracy_improved=True)
        meta.observe({"memory_bonus": 0.01, "streak_penalty": -0.3}, accuracy_improved=False)
        meta.observe({"memory_bonus": 0.06, "streak_penalty": -0.05}, accuracy_improved=True)
        meta.observe({"memory_bonus": 0.02, "streak_penalty": -0.2}, accuracy_improved=False)
        weights = meta.optimize()
        assert "memory_bonus" in weights
        assert meta.stats()["generation"] == 1

    def test_not_enough_data(self):
        meta = MetaRewardOptimizer()
        meta.observe({"x": 1}, accuracy_improved=True)
        weights = meta.optimize()
        assert all(v == 1.0 for v in weights.values())


class TestCausalEngine:
    def test_observe_and_graph(self):
        engine = CausalEngine()
        engine.observe("phishing", "UNSAFE", "correct")
        engine.observe("phishing", "SAFE", "wrong")
        engine.observe("phishing", "UNSAFE", "correct")
        graph = engine.get_causal_graph()
        assert "phishing" in graph
        assert len(graph["phishing"]) == 2

    def test_what_if(self):
        engine = CausalEngine()
        for _ in range(5):
            engine.observe("malware", "UNSAFE", "correct")
        engine.observe("malware", "SAFE", "wrong")
        result = engine.what_if("malware", "UNSAFE")
        assert result["prediction"] == "correct"
        assert result["probability"] == 1.0

    def test_best_action(self):
        engine = CausalEngine()
        for _ in range(10):
            engine.observe("safe_content", "SAFE", "correct")
        engine.observe("safe_content", "UNSAFE", "wrong")
        result = engine.best_action(["safe_content"])
        assert result is not None
        assert result[0] == "SAFE"

    def test_unknown_what_if(self):
        engine = CausalEngine()
        result = engine.what_if("unknown", "SAFE")
        assert result["prediction"] == "unknown"


class TestAgentBuilder:
    def test_build_safety(self):
        agent = build_agent(goal="maximize safety")
        assert "safe" in agent.name.lower() or "Safe" in type(agent).__name__

    def test_build_learning(self):
        agent = build_agent(goal="fast learning")
        assert "learn" in agent.name.lower() or "Auto" in type(agent).__name__

    def test_build_cost(self):
        agent = build_agent(goal="minimize cost")
        info = describe_agent(agent)
        assert info["goal"] == "minimize cost"

    def test_build_balanced(self):
        agent = build_agent(goal="balanced general purpose")
        assert agent is not None

    def test_describe(self):
        agent = build_agent(goal="maximize accuracy", risk_tolerance="low")
        info = describe_agent(agent)
        assert info["risk_tolerance"] == "low"


class TestStrategySwitcher:
    def test_default_modes(self):
        sw = StrategySwitcher()
        assert sw.current_mode == "balanced"
        assert len(sw.modes) >= 4

    def test_auto_switch_low_accuracy(self):
        sw = StrategySwitcher()
        mode = sw.decide(accuracy=0.1)
        assert mode == "explore"

    def test_auto_switch_high_accuracy(self):
        sw = StrategySwitcher()
        mode = sw.decide(accuracy=0.9)
        assert mode == "safe"

    def test_streak_switch(self):
        sw = StrategySwitcher()
        mode = sw.decide(accuracy=0.5, streak=-5)
        assert mode == "explore"

    def test_risk_switch(self):
        sw = StrategySwitcher()
        mode = sw.decide(accuracy=0.5, risk=0.9)
        assert mode == "safe"

    def test_custom_rule(self):
        sw = StrategySwitcher()
        sw.add_rule("*", "aggressive", "accuracy_above", 0.95)
        mode = sw.decide(accuracy=0.98)
        assert mode == "aggressive"

    def test_get_params(self):
        sw = StrategySwitcher()
        params = sw.get_params("safe")
        assert "epsilon" in params

    def test_stats(self):
        sw = StrategySwitcher()
        sw.decide(accuracy=0.1)
        s = sw.stats()
        assert s["total_switches"] >= 1


class TestLifelongAgent:
    def test_run_session(self):
        agent = LifelongAgent("test-001")
        result = agent.run_session("SafetyClassification-v1", episodes=1, verbose=False)
        assert result["avg_accuracy"] >= 0
        assert agent.total_steps > 0

    def test_save_load(self, tmp_path):
        agent = LifelongAgent("test-002", storage_dir=str(tmp_path))
        agent.run_session("SafetyClassification-v1", episodes=1, verbose=False)
        agent.save()

        loaded = LifelongAgent.load("test-002", storage_dir=str(tmp_path))
        assert loaded.total_steps == agent.total_steps
        assert loaded.total_sessions == 1

    def test_multi_env(self):
        agent = LifelongAgent("test-003")
        agent.run_session("SafetyClassification-v1", episodes=1, verbose=False)
        agent.run_session("MathReasoning-v1", episodes=1, verbose=False)
        assert len(agent.total_environments) == 2
        assert agent.total_sessions == 2

    def test_biography(self):
        agent = LifelongAgent("test-004")
        agent.run_session("SafetyClassification-v1", episodes=1, verbose=False)
        bio = agent.biography()
        assert bio["agent_id"] == "test-004"
        assert bio["total_sessions"] == 1


class TestSwarm:
    def test_create_swarm(self):
        swarm = Swarm(size=3)
        assert len(swarm.agents) == 3

    def test_solve(self):
        swarm = Swarm(size=3, diversity=True)
        result = swarm.solve("SafetyClassification-v1", episodes=1, verbose=False)
        assert result.avg_accuracy >= 0
        assert result.best_agent is not None

    def test_shared_memory(self):
        swarm = Swarm(size=2)
        result = swarm.solve("SafetyClassification-v1", episodes=1, verbose=False)
        mem = result.shared.stats()
        assert mem["total_contributions"] > 0


# ---------------------------------------------------------------------------
# Helpers for SwarmEnv tests
# ---------------------------------------------------------------------------

from cognicore.swarm import SwarmEnv
from cognicore.core.types import EvalResult


class _SimpleSwarmEnv(SwarmEnv):
    """Minimal concrete SwarmEnv for testing."""

    TASKS = [
        {"question": "1+1", "answer": 2, "category": "arithmetic"},
        {"question": "2+2", "answer": 4, "category": "arithmetic"},
        {"question": "3*3", "answer": 9, "category": "multiplication"},
    ]

    def _setup(self, **kwargs):
        pass

    def _generate_tasks(self):
        return list(self.TASKS)

    def _evaluate_multi(self, actions):
        task = self._tasks[self._current_step]
        results = {}
        for aid in self.agent_ids:
            action = actions.get(aid, {})
            predicted = action.get("answer", 0)
            correct = predicted == task["answer"]
            results[aid] = {
                "eval_result": EvalResult(
                    base_score=1.0 if correct else 0.0,
                    correct=correct,
                    category=task["category"],
                    predicted=str(predicted),
                    ground_truth=str(task["answer"]),
                ),
                "message": "ok",
            }
        return results

    def _get_obs_for_agent(self, agent_id):
        if self._current_step >= len(self._tasks):
            return {}
        task = self._tasks[self._current_step]
        return {"question": task["question"], "category": task["category"]}


class TestSwarmEnv:
    def test_create_swarm_env(self):
        env = _SimpleSwarmEnv(num_agents=2)
        assert env.num_agents == 2
        assert hasattr(env, "shared_memory")
        assert len(env.agent_ids) == 2

    def test_reset_contains_swarm_stats(self):
        env = _SimpleSwarmEnv(num_agents=2)
        obs = env.reset()
        for aid in env.agent_ids:
            assert aid in obs
            assert "swarm_stats" in obs[aid]

    def test_step_updates_shared_memory(self):
        env = _SimpleSwarmEnv(num_agents=2)
        env.reset()
        # Submit actions from both agents
        env.step_agent(env.agent_ids[0], {"answer": 2})
        env.step_agent(env.agent_ids[1], {"answer": 2})
        stats = env.shared_memory.stats()
        assert stats["total_contributions"] > 0

    def test_get_swarm_obs_includes_suggestion(self):
        env = _SimpleSwarmEnv(num_agents=2)
        env.reset()
        # Seed shared memory with a correct action
        env.shared_memory.contribute("agent_0", "arithmetic", "2", True)
        obs = env.get_swarm_obs(env.agent_ids[0])
        assert "swarm_suggestion" in obs
        assert obs["swarm_suggestion"] == "2"

    def test_get_swarm_obs_no_suggestion_when_memory_empty(self):
        env = _SimpleSwarmEnv(num_agents=2)
        env.reset()
        obs = env.get_swarm_obs(env.agent_ids[0])
        assert "swarm_suggestion" not in obs

    def test_swarm_stats_method(self):
        env = _SimpleSwarmEnv(num_agents=2)
        env.reset()
        stats = env.swarm_stats()
        assert "categories" in stats
        assert "total_contributions" in stats
        assert "top_contributors" in stats

    def test_full_episode(self):
        env = _SimpleSwarmEnv(num_agents=2)
        env.reset()
        done = False
        steps = 0
        while not done:
            results = None
            for aid in env.agent_ids:
                r = env.step_agent(aid, {"answer": 99})
                if "_done" in r:
                    results = r
            assert results is not None
            done = results["_done"]
            steps += 1
        assert steps == len(_SimpleSwarmEnv.TASKS)
        # Shared memory must have been populated
        assert env.swarm_stats()["total_contributions"] > 0

    def test_swarm_solve_parallel(self):
        """Swarm.solve_parallel runs all agents simultaneously."""
        env = _SimpleSwarmEnv(num_agents=2)
        swarm = Swarm(size=2, diversity=False)
        result = swarm.solve_parallel(env, episodes=1, verbose=False)
        assert result.avg_accuracy >= 0
        assert result.best_agent is not None
        # Shared memory must have been merged into Swarm's pool
        assert swarm.shared.stats()["total_contributions"] > 0

    def test_solve_parallel_wrong_type_raises(self):
        import pytest
        swarm = Swarm(size=2)
        with pytest.raises(TypeError):
            swarm.solve_parallel("not_a_swarm_env")

    def test_solve_parallel_size_mismatch_raises(self):
        import pytest
        env = _SimpleSwarmEnv(num_agents=3)
        swarm = Swarm(size=2)
        with pytest.raises(ValueError):
            swarm.solve_parallel(env)

    def test_swarm_env_exported_from_cognicore(self):
        import cognicore
        assert hasattr(cognicore, "SwarmEnv")
        assert cognicore.SwarmEnv is SwarmEnv
