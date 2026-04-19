"""
CogniCore Auto-Improvement — Self-improving agent loop.

Runs the cycle: Run → Analyze → Improve → Repeat until
the agent stops improving or reaches the target accuracy.

Usage::

    from cognicore.auto_improve import auto_improve

    result = auto_improve(
        agent=my_agent,
        env_id="SafetyClassification-v1",
        target_accuracy=0.9,
        max_cycles=10,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import cognicore
from cognicore.analytics import PerformanceAnalyzer
from cognicore.smart_agents import AutoLearner


def auto_improve(
    agent=None,
    env_id: str = "SafetyClassification-v1",
    difficulty: str = "easy",
    target_accuracy: float = 0.9,
    max_cycles: int = 10,
    episodes_per_cycle: int = 3,
    patience: int = 3,
    verbose: bool = True,
    **env_kwargs,
) -> Dict[str, Any]:
    """Self-improving agent loop.

    Runs cycles of: Run episodes → Analyze weaknesses → Focus on weak areas → Repeat

    Parameters
    ----------
    agent : BaseAgent or None
        Agent to improve. None uses AutoLearner.
    env_id : str
        Environment to train on.
    target_accuracy : float
        Stop when this accuracy is achieved.
    max_cycles : int
        Maximum improvement cycles.
    episodes_per_cycle : int
        Episodes per cycle.
    patience : int
        Stop if no improvement for this many cycles.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Improvement history with before/after stats.
    """
    if agent is None:
        agent = AutoLearner()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Auto-Improvement Loop")
        print(f"  Env: {env_id} ({difficulty})")
        print(f"  Target: {target_accuracy:.0%} accuracy")
        print(f"  Max cycles: {max_cycles}")
        print(f"{'=' * 60}")

    history = []
    best_accuracy = 0
    no_improve_count = 0
    analyzer = PerformanceAnalyzer()

    for cycle in range(1, max_cycles + 1):
        cycle_correct = 0
        cycle_total = 0
        cycle_score = 0

        for ep in range(episodes_per_cycle):
            env = cognicore.make(env_id, difficulty=difficulty, **env_kwargs)
            obs = env.reset()

            while True:
                action = agent.act(obs)
                obs, reward, done, _, info = env.step(action)

                # Let agent learn
                if hasattr(agent, 'learn'):
                    agent.learn(reward, info)

                er = info.get("eval_result", {})
                if er.get("correct"):
                    cycle_correct += 1
                cycle_total += 1

                # Record for analytics
                analyzer.record_step(
                    step=cycle_total,
                    episode=cycle,
                    category=er.get("category", "?"),
                    correct=er.get("correct", False),
                    reward_total=reward.total,
                    memory_bonus=reward.memory_bonus,
                    streak_penalty=reward.streak_penalty,
                    novelty_bonus=reward.novelty_bonus,
                    predicted=str(er.get("predicted", "")),
                    ground_truth=str(er.get("ground_truth", "")),
                )

                if done:
                    break

            stats = env.episode_stats()
            cycle_score = env.get_score()
            analyzer.record_episode(env, cycle)

        # Cycle results
        accuracy = cycle_correct / cycle_total if cycle_total > 0 else 0
        improved = accuracy > best_accuracy

        if improved:
            best_accuracy = accuracy
            no_improve_count = 0
        else:
            no_improve_count += 1

        record = {
            "cycle": cycle,
            "accuracy": accuracy,
            "score": cycle_score,
            "correct": cycle_correct,
            "total": cycle_total,
            "improved": improved,
            "best_accuracy": best_accuracy,
        }
        history.append(record)

        if verbose:
            icon = "UP" if improved else "--"
            print(
                f"  Cycle {cycle:2d}: accuracy={accuracy:.0%} "
                f"score={cycle_score:.4f} "
                f"best={best_accuracy:.0%} [{icon}]"
            )

        # Check stopping conditions
        if accuracy >= target_accuracy:
            if verbose:
                print(f"\n  TARGET REACHED! {accuracy:.0%} >= {target_accuracy:.0%}")
            break

        if no_improve_count >= patience:
            if verbose:
                print(f"\n  PLATEAU: No improvement for {patience} cycles. Stopping.")
            break

    # Final report
    report = analyzer.analyze()

    result = {
        "cycles": len(history),
        "initial_accuracy": history[0]["accuracy"] if history else 0,
        "final_accuracy": history[-1]["accuracy"] if history else 0,
        "best_accuracy": best_accuracy,
        "improvement": best_accuracy - (history[0]["accuracy"] if history else 0),
        "target_reached": best_accuracy >= target_accuracy,
        "history": history,
        "weak_categories": report.weak_categories(5),
        "is_improving": report.is_improving(),
    }

    if verbose:
        print(f"\n  Summary:")
        print(f"    {result['initial_accuracy']:.0%} -> {result['final_accuracy']:.0%} "
              f"(improvement: {result['improvement']:+.0%})")
        if result["weak_categories"]:
            print(f"    Still weak on:")
            for w in result["weak_categories"][:3]:
                print(f"      - {w['category']}: {w['accuracy']:.0%}")
        print(f"{'=' * 60}\n")

    return result
