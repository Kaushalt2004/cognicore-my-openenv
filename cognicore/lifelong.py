"""
CogniCore Lifelong Learning — Persistent agent identity across tasks.

Agents maintain identity, accumulate knowledge, and grow across
sessions and environments without resetting.

Usage::

    from cognicore.lifelong import LifelongAgent

    agent = LifelongAgent("agent-001")
    agent.run_session("SafetyClassification-v1", episodes=3)
    agent.run_session("CodeDebugging-v1", episodes=3)
    agent.save()  # persists everything

    # Next day...
    agent = LifelongAgent.load("agent-001")
    agent.run_session("MathReasoning-v1", episodes=3)
    agent.print_biography()
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import cognicore
from cognicore.smart_agents import AutoLearner


class LifelongAgent:
    """An agent with persistent identity that learns across sessions.

    Never resets. Knowledge accumulates across environments and tasks.

    Parameters
    ----------
    agent_id : str
        Unique persistent identifier.
    storage_dir : str
        Where to store agent state.
    """

    def __init__(self, agent_id: str, storage_dir: str = "./cognicore_agents"):
        self.agent_id = agent_id
        self.storage_dir = storage_dir
        self.inner = AutoLearner()
        self.inner.name = agent_id

        # Lifelong stats
        self.total_steps = 0
        self.total_correct = 0
        self.total_sessions = 0
        self.total_environments = set()
        self.session_log: List[Dict] = []
        self.created_at = time.time()
        self.last_active = time.time()

        # Cross-environment knowledge
        self.env_performance: Dict[str, Dict] = defaultdict(
            lambda: {"sessions": 0, "accuracy": 0, "best_accuracy": 0}
        )

    def run_session(
        self,
        env_id: str,
        difficulty: str = "easy",
        episodes: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run a learning session on an environment."""
        self.total_sessions += 1
        self.total_environments.add(env_id)
        self.last_active = time.time()

        session_results = []

        for ep in range(episodes):
            env = cognicore.make(env_id, difficulty=difficulty)
            obs = env.reset()
            ep_correct = 0
            ep_steps = 0

            while True:
                action = self.inner.act(obs)
                obs, reward, done, _, info = env.step(action)
                self.inner.learn(reward, info)

                ep_steps += 1
                self.total_steps += 1
                if info.get("eval_result", {}).get("correct"):
                    ep_correct += 1
                    self.total_correct += 1

                if done:
                    break

            stats = env.episode_stats()
            session_results.append(
                {
                    "episode": ep + 1,
                    "accuracy": stats.accuracy,
                    "score": env.get_score(),
                    "correct": stats.correct_count,
                    "steps": stats.steps,
                }
            )

            if verbose:
                print(
                    f"  [{self.agent_id}] {env_id} ep{ep + 1}: "
                    f"accuracy={stats.accuracy:.0%} score={env.get_score():.4f}"
                )

        # Update env performance
        avg_acc = sum(r["accuracy"] for r in session_results) / len(session_results)
        perf = self.env_performance[env_id]
        perf["sessions"] += episodes
        perf["accuracy"] = avg_acc
        perf["best_accuracy"] = max(perf["best_accuracy"], avg_acc)

        self.session_log.append(
            {
                "env_id": env_id,
                "difficulty": difficulty,
                "episodes": episodes,
                "avg_accuracy": avg_acc,
                "timestamp": time.time(),
            }
        )

        return {
            "env_id": env_id,
            "episodes": episodes,
            "avg_accuracy": avg_acc,
            "results": session_results,
        }

    def save(self, path: str = None) -> str:
        """Save agent to disk."""
        if path is None:
            os.makedirs(self.storage_dir, exist_ok=True)
            path = os.path.join(self.storage_dir, f"{self.agent_id}.json")

        data = {
            "_cognicore_lifelong": True,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "total_steps": self.total_steps,
            "total_correct": self.total_correct,
            "total_sessions": self.total_sessions,
            "total_environments": list(self.total_environments),
            "env_performance": dict(self.env_performance),
            "session_log": self.session_log[-100:],
            "knowledge": {
                k: dict(v) if isinstance(v, dict) else v
                for k, v in self.inner.knowledge.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return path

    @classmethod
    def load(
        cls, agent_id: str, storage_dir: str = "./cognicore_agents"
    ) -> "LifelongAgent":
        """Load agent from disk."""
        path = os.path.join(storage_dir, f"{agent_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        agent = cls(agent_id, storage_dir)
        agent.created_at = data.get("created_at", time.time())
        agent.last_active = data.get("last_active", time.time())
        agent.total_steps = data.get("total_steps", 0)
        agent.total_correct = data.get("total_correct", 0)
        agent.total_sessions = data.get("total_sessions", 0)
        agent.total_environments = set(data.get("total_environments", []))
        agent.session_log = data.get("session_log", [])

        for env_id, perf in data.get("env_performance", {}).items():
            agent.env_performance[env_id] = perf

        for k, v in data.get("knowledge", {}).items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    agent.inner.knowledge[k][k2] = v2

        return agent

    @property
    def lifetime_accuracy(self) -> float:
        if self.total_steps == 0:
            return 0
        return self.total_correct / self.total_steps

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def biography(self) -> Dict[str, Any]:
        """Get agent's full biography."""
        return {
            "agent_id": self.agent_id,
            "age_hours": round(self.age_hours, 2),
            "total_sessions": self.total_sessions,
            "total_steps": self.total_steps,
            "lifetime_accuracy": round(self.lifetime_accuracy, 3),
            "environments_experienced": len(self.total_environments),
            "knowledge_categories": len(self.inner.knowledge),
            "env_performance": dict(self.env_performance),
        }

    def print_biography(self):
        """Print formatted agent biography."""
        bio = self.biography()
        print(f"\n{'=' * 55}")
        print(f"  Agent: {bio['agent_id']}")
        print(f"  Age: {bio['age_hours']:.1f} hours")
        print(f"{'=' * 55}")
        print(f"  Sessions: {bio['total_sessions']}")
        print(f"  Steps: {bio['total_steps']}")
        print(f"  Lifetime accuracy: {bio['lifetime_accuracy']:.0%}")
        print(f"  Environments: {bio['environments_experienced']}")
        print(f"  Categories known: {bio['knowledge_categories']}")

        if bio["env_performance"]:
            print("\n  Performance by environment:")
            for env_id, perf in bio["env_performance"].items():
                bar_len = int(perf.get("best_accuracy", 0) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(
                    f"    {env_id:35s} [{bar}] best={perf.get('best_accuracy', 0):.0%}"
                )

        print(f"{'=' * 55}\n")
