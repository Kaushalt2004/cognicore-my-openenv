"""
CogniCore Benchmark — Run agents across all environments and produce reports.

Usage::

    from cognicore.benchmark import benchmark_agent

    results = benchmark_agent(my_agent, envs="all", episodes=3)
    results.print_report()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cognicore
from cognicore.agents.base_agent import BaseAgent, RandomAgent


class BenchmarkResult:
    """Results from a benchmark run."""

    def __init__(self, agent_id: str, results: List[Dict[str, Any]]):
        self.agent_id = agent_id
        self.results = results

    @property
    def overall_accuracy(self) -> float:
        total_correct = sum(r["correct"] for r in self.results)
        total_steps = sum(r["total"] for r in self.results)
        return total_correct / total_steps if total_steps else 0

    @property
    def overall_score(self) -> float:
        return sum(r["score"] for r in self.results) / len(self.results) if self.results else 0

    def by_env(self) -> Dict[str, Dict[str, float]]:
        """Group results by environment."""
        groups = {}
        for r in self.results:
            env = r["env_id"]
            if env not in groups:
                groups[env] = {"scores": [], "accuracies": []}
            groups[env]["scores"].append(r["score"])
            groups[env]["accuracies"].append(r["accuracy"])
        return {
            k: {
                "avg_score": sum(v["scores"]) / len(v["scores"]),
                "avg_accuracy": sum(v["accuracies"]) / len(v["accuracies"]),
                "runs": len(v["scores"]),
            }
            for k, v in groups.items()
        }

    def by_difficulty(self) -> Dict[str, Dict[str, float]]:
        """Group results by difficulty."""
        groups = {}
        for r in self.results:
            d = r["difficulty"]
            if d not in groups:
                groups[d] = {"scores": [], "accuracies": []}
            groups[d]["scores"].append(r["score"])
            groups[d]["accuracies"].append(r["accuracy"])
        return {
            k: {
                "avg_score": sum(v["scores"]) / len(v["scores"]),
                "avg_accuracy": sum(v["accuracies"]) / len(v["accuracies"]),
            }
            for k, v in groups.items()
        }

    def print_report(self):
        """Print a formatted benchmark report."""
        print(f"\n{'=' * 70}")
        print(f"  CogniCore Benchmark Report - Agent: {self.agent_id}")
        print(f"{'=' * 70}")
        print(f"  Overall: accuracy={self.overall_accuracy:.0%}  score={self.overall_score:.4f}")
        print(f"  Environments tested: {len(self.by_env())}")
        print(f"  Total runs: {len(self.results)}")
        print(f"\n  {'Environment':<35} {'Score':<10} {'Accuracy':<10}")
        print(f"  {'-'*55}")
        for env, stats in sorted(self.by_env().items()):
            print(f"  {env:<35} {stats['avg_score']:<10.4f} {stats['avg_accuracy']*100:<9.0f}%")

        print(f"\n  {'Difficulty':<15} {'Score':<10} {'Accuracy':<10}")
        print(f"  {'-'*35}")
        for d, stats in sorted(self.by_difficulty().items()):
            print(f"  {d:<15} {stats['avg_score']:<10.4f} {stats['avg_accuracy']*100:<9.0f}%")
        print(f"{'=' * 70}\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "overall_accuracy": self.overall_accuracy,
            "overall_score": self.overall_score,
            "by_env": self.by_env(),
            "by_difficulty": self.by_difficulty(),
            "results": self.results,
        }


def benchmark_agent(
    agent: Optional[BaseAgent] = None,
    agent_id: str = "benchmark-agent",
    envs: str | List[str] = "all",
    difficulties: List[str] = ["easy", "medium", "hard"],
    episodes: int = 1,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run an agent across multiple environments and produce a report.

    Parameters
    ----------
    agent : BaseAgent or None
        Agent to benchmark. None uses RandomAgent (baseline).
    agent_id : str
        Name for the agent in reports.
    envs : str or list
        "all" for all base environments, or list of env IDs.
    difficulties : list
        Difficulty levels to test.
    episodes : int
        Episodes per env/difficulty combo.
    verbose : bool
        Print progress.

    Returns
    -------
    BenchmarkResult
        Results with report generation.
    """
    if envs == "all":
        all_envs = cognicore.list_envs()
        # Get base envs only (not Easy/Medium/Hard variants)
        base_ids = [
            e["id"] for e in all_envs
            if not any(d in e["id"] for d in ["-Easy-", "-Medium-", "-Hard-"])
        ]
    else:
        base_ids = envs

    results = []
    total = len(base_ids) * len(difficulties) * episodes

    if verbose:
        print(f"\nBenchmarking {agent_id} across {len(base_ids)} envs x {len(difficulties)} difficulties x {episodes} episodes = {total} runs")
        print("-" * 60)

    run = 0
    for env_id in base_ids:
        for diff in difficulties:
            for ep in range(episodes):
                run += 1
                try:
                    env = cognicore.make(env_id, difficulty=diff)
                except Exception:
                    continue

                if agent is None:
                    _agent = RandomAgent(env.action_space)
                else:
                    _agent = agent

                obs = env.reset()
                while True:
                    action = _agent.act(obs)
                    obs, reward, done, _, info = env.step(action)
                    if done:
                        break

                stats = env.episode_stats()
                result = {
                    "env_id": env_id,
                    "difficulty": diff,
                    "episode": ep + 1,
                    "score": env.get_score(),
                    "accuracy": stats.accuracy,
                    "correct": stats.correct_count,
                    "total": stats.steps,
                    "memory_entries": stats.memory_entries_created,
                }
                results.append(result)

                if verbose:
                    print(
                        f"  [{run:3d}/{total}] {env_id:<30s} {diff:<6s} "
                        f"acc={stats.accuracy:.0%} score={env.get_score():.4f}"
                    )

    return BenchmarkResult(agent_id=agent_id, results=results)
