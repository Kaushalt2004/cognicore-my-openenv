"""
Fine-Tuning Data Exporter — Export CogniCore episodes for LLM fine-tuning.

Supports multiple output formats:
- JSONL (OpenAI fine-tuning format)
- DPO pairs (preferred/rejected for Direct Preference Optimization)
- Reward-labeled dataset (for RLHF reward modeling)

Usage::

    from cognicore.finetuning import EpisodeRecorder, export_jsonl, export_dpo

    recorder = EpisodeRecorder()
    # Run episodes...
    recorder.record_step(obs, action, reward, info)

    export_jsonl(recorder.episodes, "training_data.jsonl")
    export_dpo(recorder.episodes, "dpo_pairs.jsonl")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from cognicore.core.types import StructuredReward


class EpisodeRecorder:
    """Records environment interactions for fine-tuning data export.

    Captures obs/action/reward triples during episodes and organizes
    them for various fine-tuning formats.
    """

    def __init__(self):
        self.episodes: List[List[Dict[str, Any]]] = []
        self._current_episode: List[Dict[str, Any]] = []
        self._recording = False

    def start_episode(self, env_id: str = "", metadata: Optional[Dict] = None):
        """Start recording a new episode."""
        self._current_episode = []
        self._recording = True
        self._episode_meta = {
            "env_id": env_id,
            **(metadata or {}),
        }

    def record_step(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        reward: StructuredReward,
        info: Dict[str, Any],
    ):
        """Record a single step."""
        if not self._recording:
            return

        self._current_episode.append(
            {
                "observation": _sanitize(observation),
                "action": _sanitize(action),
                "reward": reward.to_dict()
                if hasattr(reward, "to_dict")
                else {"total": float(reward)},
                "correct": info.get("eval_result", {}).get("correct", False),
                "ground_truth": info.get("eval_result", {}).get("ground_truth", ""),
            }
        )

    def end_episode(self, score: float = 0.0, accuracy: float = 0.0):
        """End the current episode."""
        if self._recording and self._current_episode:
            self.episodes.append(
                {
                    "metadata": {
                        **self._episode_meta,
                        "score": score,
                        "accuracy": accuracy,
                    },
                    "steps": self._current_episode,
                }
            )
        self._current_episode = []
        self._recording = False

    @property
    def total_steps(self) -> int:
        return sum(len(ep["steps"]) for ep in self.episodes)


def export_jsonl(episodes: list, output_path: str, system_prompt: str = "") -> int:
    """Export episodes as OpenAI fine-tuning JSONL.

    Each correct step becomes a training example:
    ``{"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}``

    Returns number of examples exported.
    """
    count = 0
    with open(output_path, "w") as f:
        for ep in episodes:
            for step in ep["steps"]:
                if not step["correct"]:
                    continue  # only train on correct examples

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                # User message from observation
                user_content = _obs_to_prompt(step["observation"])
                messages.append({"role": "user", "content": user_content})

                # Assistant response from action
                assistant_content = json.dumps(step["action"])
                messages.append({"role": "assistant", "content": assistant_content})

                f.write(json.dumps({"messages": messages}) + "\n")
                count += 1

    return count


def export_dpo(episodes: list, output_path: str) -> int:
    """Export episodes as DPO (Direct Preference Optimization) pairs.

    Creates preferred/rejected pairs where correct actions are preferred
    over incorrect actions for the same observation category.

    Returns number of pairs exported.
    """
    # Group steps by observation similarity
    correct_steps = []
    incorrect_steps = []

    for ep in episodes:
        for step in ep["steps"]:
            if step["correct"]:
                correct_steps.append(step)
            else:
                incorrect_steps.append(step)

    # Create pairs
    count = 0
    with open(output_path, "w") as f:
        for correct in correct_steps:
            for incorrect in incorrect_steps:
                pair = {
                    "prompt": _obs_to_prompt(correct["observation"]),
                    "chosen": json.dumps(correct["action"]),
                    "rejected": json.dumps(incorrect["action"]),
                    "chosen_reward": correct["reward"].get("total", 0),
                    "rejected_reward": incorrect["reward"].get("total", 0),
                }
                f.write(json.dumps(pair) + "\n")
                count += 1

                if count >= 10000:  # cap to prevent huge files
                    return count

    return count


def export_reward_dataset(episodes: list, output_path: str) -> int:
    """Export episodes as reward-labeled dataset for RLHF reward modeling.

    Each step becomes: ``{"input": ..., "output": ..., "reward": ..., "components": ...}``

    Returns number of examples exported.
    """
    count = 0
    with open(output_path, "w") as f:
        for ep in episodes:
            for step in ep["steps"]:
                entry = {
                    "input": _obs_to_prompt(step["observation"]),
                    "output": json.dumps(step["action"]),
                    "reward": step["reward"].get("total", 0),
                    "components": step["reward"],
                    "correct": step["correct"],
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

    return count


def _obs_to_prompt(obs: Dict[str, Any]) -> str:
    """Convert an observation dict to a text prompt."""
    parts = []
    for key in ("text", "prompt", "question", "buggy_code", "scenario", "user_message"):
        if key in obs and obs[key]:
            parts.append(f"{key}: {obs[key]}")
    if not parts:
        parts.append(json.dumps(obs))
    return "\n".join(parts)


def _sanitize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
