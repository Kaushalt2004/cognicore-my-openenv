"""
CogniCore Agent Fingerprinting — Behavioral DNA for agents.

Creates a vector representation of agent behavior for comparison,
clustering, and similarity analysis.

Usage::

    from cognicore.fingerprint import AgentFingerprint

    fp = AgentFingerprint("SafetyClassification-v1")
    dna_a = fp.fingerprint(agent_a)
    dna_b = fp.fingerprint(agent_b)
    similarity = fp.compare(dna_a, dna_b)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import cognicore
from cognicore.agents.base_agent import RandomAgent


class AgentDNA:
    """Vectorized representation of agent behavior."""

    def __init__(self, agent_name: str, vector: Dict[str, float]):
        self.agent_name = agent_name
        self.vector = vector

    def to_list(self) -> List[float]:
        return list(self.vector.values())

    def distance(self, other: "AgentDNA") -> float:
        """Euclidean distance to another DNA."""
        keys = set(self.vector) | set(other.vector)
        total = 0
        for k in keys:
            a = self.vector.get(k, 0)
            b = other.vector.get(k, 0)
            total += (a - b) ** 2
        return math.sqrt(total)

    def similarity(self, other: "AgentDNA") -> float:
        """Cosine similarity to another DNA (0-1)."""
        keys = list(set(self.vector) | set(other.vector))
        dot = sum(self.vector.get(k, 0) * other.vector.get(k, 0) for k in keys)
        mag_a = math.sqrt(sum(self.vector.get(k, 0) ** 2 for k in keys))
        mag_b = math.sqrt(sum(other.vector.get(k, 0) ** 2 for k in keys))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def print_dna(self):
        print(f"\n  Agent DNA: {self.agent_name}")
        for k, v in sorted(self.vector.items()):
            bar_len = int(abs(v) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {k:25s} [{bar}] {v:.3f}")


class AgentFingerprint:
    """Generate behavioral fingerprints for agents.

    Runs the agent through an environment and extracts a
    multi-dimensional behavior vector.
    """

    def __init__(
        self, env_id: str = "SafetyClassification-v1", difficulty: str = "easy"
    ):
        self.env_id = env_id
        self.difficulty = difficulty

    def fingerprint(self, agent=None, episodes: int = 2) -> AgentDNA:
        """Create a DNA fingerprint from agent behavior."""
        if agent is None:
            agent = RandomAgent(cognicore.make(self.env_id).action_space)

        name = getattr(agent, "name", type(agent).__name__)
        metrics = {
            "accuracy": [],
            "reward_avg": [],
            "memory_bonus_avg": [],
            "streak_penalty_avg": [],
            "novelty_avg": [],
            "action_diversity": [],
            "confidence_avg": [],
        }
        category_accuracy = {}
        action_counts = {}

        for _ in range(episodes):
            env = cognicore.make(self.env_id, difficulty=self.difficulty)
            obs = env.reset()
            ep_rewards = []
            ep_memory = []
            ep_streak = []
            ep_novelty = []
            ep_conf = []

            while True:
                action = agent.act(obs)
                action_str = str(action.get("classification", action))
                action_counts[action_str] = action_counts.get(action_str, 0) + 1

                obs, reward, done, _, info = env.step(action)
                er = info.get("eval_result", {})

                cat = er.get("category", "?")
                if cat not in category_accuracy:
                    category_accuracy[cat] = {"c": 0, "t": 0}
                category_accuracy[cat]["t"] += 1
                if er.get("correct"):
                    category_accuracy[cat]["c"] += 1

                ep_rewards.append(reward.total)
                ep_memory.append(reward.memory_bonus)
                ep_streak.append(reward.streak_penalty)
                ep_novelty.append(reward.novelty_bonus)
                ep_conf.append(er.get("confidence", 0))

                if hasattr(agent, "learn"):
                    agent.learn(reward, info)

                if done:
                    break

            stats = env.episode_stats()
            metrics["accuracy"].append(stats.accuracy)
            metrics["reward_avg"].append(sum(ep_rewards) / max(len(ep_rewards), 1))
            metrics["memory_bonus_avg"].append(sum(ep_memory) / max(len(ep_memory), 1))
            metrics["streak_penalty_avg"].append(
                sum(ep_streak) / max(len(ep_streak), 1)
            )
            metrics["novelty_avg"].append(sum(ep_novelty) / max(len(ep_novelty), 1))
            metrics["confidence_avg"].append(sum(ep_conf) / max(len(ep_conf), 1))

            unique_actions = len(set(action_counts.keys()))
            total_actions_taken = sum(action_counts.values())
            metrics["action_diversity"].append(
                unique_actions / max(total_actions_taken, 1)
            )

        # Build DNA vector
        vector = {}
        for k, vals in metrics.items():
            vector[k] = sum(vals) / max(len(vals), 1)

        # Category specialization
        for cat, data in category_accuracy.items():
            acc = data["c"] / data["t"] if data["t"] else 0
            vector[f"cat_{cat}"] = acc

        # Action preference
        total = sum(action_counts.values())
        for action, count in action_counts.items():
            vector[f"action_{action}"] = count / total if total else 0

        return AgentDNA(name, vector)

    def compare(self, dna_a: AgentDNA, dna_b: AgentDNA) -> Dict[str, Any]:
        """Compare two agent fingerprints."""
        sim = dna_a.similarity(dna_b)
        dist = dna_a.distance(dna_b)

        # Which dimensions differ most
        all_keys = set(dna_a.vector) | set(dna_b.vector)
        diffs = []
        for k in all_keys:
            va = dna_a.vector.get(k, 0)
            vb = dna_b.vector.get(k, 0)
            diffs.append({"dimension": k, "a": va, "b": vb, "diff": abs(va - vb)})

        diffs.sort(key=lambda x: -x["diff"])

        return {
            "similarity": sim,
            "distance": dist,
            "agents": [dna_a.agent_name, dna_b.agent_name],
            "biggest_differences": diffs[:5],
        }
