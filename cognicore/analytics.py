"""
CogniCore Analytics — Deep performance analysis across episodes.

Provides pattern detection, weakness identification, learning curves,
and actionable insights from agent performance data.

Usage::

    from cognicore.analytics import PerformanceAnalyzer

    analyzer = PerformanceAnalyzer()
    analyzer.record(env)  # after each episode
    report = analyzer.analyze()
    report.print_insights()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import defaultdict


class PerformanceAnalyzer:
    """Deep analytics engine for CogniCore agent performance.

    Records episode data and produces insights about:
    - Weak categories (what the agent struggles with)
    - Learning curves (is the agent improving?)
    - Memory utilization (is memory helping?)
    - Streak patterns (when does the agent fail?)
    - Reward component analysis
    """

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []
        self._step_data: List[Dict[str, Any]] = []

    def record_step(
        self,
        step: int,
        episode: int,
        category: str,
        correct: bool,
        reward_total: float,
        memory_bonus: float = 0,
        streak_penalty: float = 0,
        novelty_bonus: float = 0,
        predicted: str = "",
        ground_truth: str = "",
    ):
        """Record a single step for analysis."""
        self._step_data.append({
            "step": step, "episode": episode, "category": category,
            "correct": correct, "reward": reward_total,
            "memory_bonus": memory_bonus, "streak_penalty": streak_penalty,
            "novelty_bonus": novelty_bonus,
            "predicted": predicted, "ground_truth": ground_truth,
        })

    def record_episode(self, env, episode_num: int = 0):
        """Record data from a completed environment episode."""
        stats = env.episode_stats()
        self.episodes.append({
            "episode": episode_num or len(self.episodes) + 1,
            "score": env.get_score(),
            "accuracy": stats.accuracy,
            "correct": stats.correct_count,
            "total": stats.steps,
            "memory_entries": stats.memory_entries_created,
        })

    def analyze(self) -> "AnalyticsReport":
        """Generate a comprehensive analytics report."""
        return AnalyticsReport(self.episodes, self._step_data)


class AnalyticsReport:
    """Generated analytics report with insights."""

    def __init__(self, episodes: list, steps: list):
        self.episodes = episodes
        self.steps = steps

    def learning_curve(self) -> List[Dict[str, float]]:
        """Track accuracy and score across episodes."""
        return [
            {"episode": e["episode"], "accuracy": e["accuracy"], "score": e["score"]}
            for e in self.episodes
        ]

    def is_improving(self) -> bool:
        """Check if the agent is trending upward."""
        if len(self.episodes) < 3:
            return False
        scores = [e["score"] for e in self.episodes]
        first_half = sum(scores[:len(scores)//2]) / max(len(scores)//2, 1)
        second_half = sum(scores[len(scores)//2:]) / max(len(scores) - len(scores)//2, 1)
        return second_half > first_half

    def weak_categories(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Identify categories where the agent struggles most."""
        cat_stats = defaultdict(lambda: {"correct": 0, "wrong": 0})
        for s in self.steps:
            key = "correct" if s["correct"] else "wrong"
            cat_stats[s["category"]][key] += 1

        results = []
        for cat, counts in cat_stats.items():
            total = counts["correct"] + counts["wrong"]
            acc = counts["correct"] / total if total else 0
            results.append({
                "category": cat,
                "accuracy": acc,
                "correct": counts["correct"],
                "wrong": counts["wrong"],
                "total": total,
            })

        return sorted(results, key=lambda x: x["accuracy"])[:top_k]

    def strong_categories(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Identify categories where the agent excels."""
        weak = self.weak_categories(top_k=100)
        return sorted(weak, key=lambda x: -x["accuracy"])[:top_k]

    def memory_impact(self) -> Dict[str, float]:
        """Analyze how much memory bonuses contributed to total rewards."""
        total_reward = sum(s["reward"] for s in self.steps) or 1
        total_memory = sum(s["memory_bonus"] for s in self.steps)
        total_streak = sum(s["streak_penalty"] for s in self.steps)
        total_novelty = sum(s["novelty_bonus"] for s in self.steps)

        return {
            "memory_total": total_memory,
            "memory_pct_of_reward": total_memory / abs(total_reward) * 100,
            "streak_total": total_streak,
            "novelty_total": total_novelty,
            "memory_helped_steps": sum(1 for s in self.steps if s["memory_bonus"] > 0),
            "streak_hit_steps": sum(1 for s in self.steps if s["streak_penalty"] < 0),
        }

    def streak_analysis(self) -> Dict[str, Any]:
        """Analyze failure streak patterns."""
        if not self.steps:
            return {"max_streak": 0, "avg_streak": 0, "total_streaks": 0}

        streaks = []
        current = 0
        for s in self.steps:
            if not s["correct"]:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)

        return {
            "max_streak": max(streaks) if streaks else 0,
            "avg_streak": sum(streaks) / len(streaks) if streaks else 0,
            "total_streaks": len(streaks),
            "streak_lengths": streaks,
        }

    def confusion_pairs(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the most common prediction errors (predicted vs truth)."""
        pairs = defaultdict(int)
        for s in self.steps:
            if not s["correct"]:
                pairs[(s["predicted"], s["ground_truth"])] += 1

        return [
            {"predicted": p, "truth": t, "count": c}
            for (p, t), c in sorted(pairs.items(), key=lambda x: -x[1])[:top_k]
        ]

    def print_insights(self):
        """Print a human-readable insights report."""
        print(f"\n{'=' * 60}")
        print(f"  CogniCore Performance Insights")
        print(f"{'=' * 60}")

        # Learning curve
        if self.episodes:
            first = self.episodes[0]
            last = self.episodes[-1]
            trend = "IMPROVING" if self.is_improving() else "FLAT/DECLINING"
            print(f"\n  Learning: {trend}")
            print(f"    First episode: accuracy={first['accuracy']:.0%} score={first['score']:.4f}")
            print(f"    Last episode:  accuracy={last['accuracy']:.0%} score={last['score']:.4f}")

        # Weak categories
        weak = self.weak_categories(3)
        if weak:
            print(f"\n  Weakest categories:")
            for w in weak:
                print(f"    {w['category']:25s} accuracy={w['accuracy']:.0%} ({w['correct']}/{w['total']})")

        # Memory impact
        mem = self.memory_impact()
        print(f"\n  Memory impact:")
        print(f"    Total memory bonus: {mem['memory_total']:+.2f} ({mem['memory_pct_of_reward']:.1f}% of reward)")
        print(f"    Steps where memory helped: {mem['memory_helped_steps']}")
        print(f"    Steps with streak penalty: {mem['streak_hit_steps']}")

        # Streak analysis
        streaks = self.streak_analysis()
        print(f"\n  Streaks:")
        print(f"    Longest failure streak: {streaks['max_streak']}")
        print(f"    Average streak length: {streaks['avg_streak']:.1f}")

        # Confusion pairs
        confused = self.confusion_pairs(3)
        if confused:
            print(f"\n  Most common mistakes:")
            for c in confused:
                print(f"    Predicted '{c['predicted']}' but truth was '{c['truth']}' ({c['count']}x)")

        print(f"{'=' * 60}\n")
