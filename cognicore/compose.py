"""
CogniCore Environment Composition — Chain environments into pipelines.

Compose multiple environments into a single meta-environment where
the agent progresses through stages. Each stage can be a different
environment type.

Usage::

    from cognicore.compose import Pipeline

    pipe = Pipeline([
        ("safety", "SafetyClassification-v1", {"difficulty": "easy"}),
        ("math",   "MathReasoning-v1",        {"difficulty": "medium"}),
        ("code",   "CodeDebugging-v1",         {"difficulty": "hard"}),
    ])

    obs = pipe.reset()
    while not pipe.done:
        action = agent.act(obs)
        obs, reward, done, _, info = pipe.step(action)
        print(f"Stage: {pipe.current_stage_name}, Reward: {reward.total:.2f}")

    print(pipe.report())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cognicore
from cognicore.core.base_env import CogniCoreEnv


class Pipeline:
    """Chain multiple environments into a sequential pipeline.

    Agents progress through stages in order. Memory is optionally
    shared across stages (cross-domain transfer learning).
    """

    def __init__(
        self,
        stages: List[Tuple[str, str, Dict[str, Any]]],
        share_memory: bool = True,
    ):
        """
        Parameters
        ----------
        stages : list of (name, env_id, kwargs)
            Each stage is a tuple of (stage_name, environment_id, kwargs).
        share_memory : bool
            If True, memory carries across stages (transfer learning).
        """
        self.stage_defs = stages
        self.share_memory = share_memory
        self._stages: List[CogniCoreEnv] = []
        self._stage_idx = 0
        self._stage_results: List[Dict[str, Any]] = []
        self.done = False

    @property
    def current_stage_name(self) -> str:
        if self._stage_idx < len(self.stage_defs):
            return self.stage_defs[self._stage_idx][0]
        return "complete"

    @property
    def current_env(self) -> Optional[CogniCoreEnv]:
        if self._stage_idx < len(self._stages):
            return self._stages[self._stage_idx]
        return None

    def reset(self) -> Dict[str, Any]:
        """Reset the pipeline. Creates all environments."""
        self._stages = []
        self._stage_idx = 0
        self._stage_results = []
        self.done = False

        for name, env_id, kwargs in self.stage_defs:
            env = cognicore.make(env_id, **kwargs)
            self._stages.append(env)

        obs = self._stages[0].reset()
        obs["_pipeline_stage"] = self.current_stage_name
        obs["_pipeline_progress"] = f"1/{len(self.stage_defs)}"
        return obs

    def step(self, action: Dict[str, Any]):
        """Step the current stage's environment."""
        if self.done:
            raise RuntimeError("Pipeline is done. Call reset().")

        env = self._stages[self._stage_idx]
        obs, reward, stage_done, truncated, info = env.step(action)

        info["pipeline_stage"] = self.current_stage_name
        info["pipeline_progress"] = f"{self._stage_idx + 1}/{len(self.stage_defs)}"

        if stage_done:
            # Record results for this stage
            stats = env.episode_stats()
            self._stage_results.append(
                {
                    "name": self.current_stage_name,
                    "env_id": self.stage_defs[self._stage_idx][1],
                    "score": env.get_score(),
                    "accuracy": stats.accuracy,
                    "correct": stats.correct_count,
                    "total": stats.steps,
                }
            )

            # Move to next stage
            self._stage_idx += 1
            if self._stage_idx >= len(self._stages):
                self.done = True
                info["pipeline_complete"] = True
            else:
                # Transfer memory if enabled
                if self.share_memory:
                    prev_memory = env.memory.entries
                    next_env = self._stages[self._stage_idx]
                    next_env.memory.entries = list(prev_memory)

                obs = self._stages[self._stage_idx].reset()
                obs["_pipeline_stage"] = self.current_stage_name
                obs["_pipeline_progress"] = (
                    f"{self._stage_idx + 1}/{len(self.stage_defs)}"
                )
                stage_done = False  # pipeline continues

        return obs, reward, self.done, truncated, info

    def report(self) -> Dict[str, Any]:
        """Generate pipeline completion report."""
        if not self._stage_results:
            return {"status": "no results"}

        total_correct = sum(r["correct"] for r in self._stage_results)
        total_steps = sum(r["total"] for r in self._stage_results)

        return {
            "stages_completed": len(self._stage_results),
            "total_stages": len(self.stage_defs),
            "overall_accuracy": total_correct / total_steps if total_steps else 0,
            "overall_score": sum(r["score"] for r in self._stage_results)
            / len(self._stage_results),
            "stages": self._stage_results,
        }

    def print_report(self):
        """Print a formatted report."""
        r = self.report()
        print(f"\n{'=' * 50}")
        print(f"  Pipeline Report ({r['stages_completed']}/{r['total_stages']} stages)")
        print(f"{'=' * 50}")
        print(
            f"  Overall: accuracy={r['overall_accuracy']:.0%} score={r['overall_score']:.4f}"
        )
        for s in r["stages"]:
            print(
                f"  [{s['name']:15s}] {s['env_id']:30s} acc={s['accuracy']:.0%} score={s['score']:.4f}"
            )
        print(f"{'=' * 50}\n")
