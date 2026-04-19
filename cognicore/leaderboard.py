"""
CogniCore Leaderboard — Track and compare agent performance.

Usage::

    from cognicore.leaderboard import Leaderboard

    lb = Leaderboard(storage_dir="./cognicore_data")
    lb.submit("my-agent", env_id="SafetyClassification-v1", score=0.95, accuracy=0.90)
    lb.get_rankings("SafetyClassification-v1")
    lb.get_rankings()  # all environments
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class Leaderboard:
    """Local leaderboard for comparing agent performance.

    Tracks scores, accuracy, and metadata for agents across environments.
    Data is persisted to a JSON file.
    """

    def __init__(self, storage_dir: str = "./cognicore_data"):
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self._file = os.path.join(self.storage_dir, "leaderboard.json")
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self._file):
            with open(self._file) as f:
                return json.load(f)
        return {"entries": []}

    def _save(self):
        with open(self._file, "w") as f:
            json.dump(self._data, f, indent=2)

    def submit(
        self,
        agent_id: str,
        env_id: str,
        score: float,
        accuracy: float,
        difficulty: str = "easy",
        episodes: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a score to the leaderboard.

        Returns the entry with rank information.
        """
        entry = {
            "agent_id": agent_id,
            "env_id": env_id,
            "score": round(score, 4),
            "accuracy": round(accuracy, 4),
            "difficulty": difficulty,
            "episodes": episodes,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        self._data["entries"].append(entry)
        self._save()

        # Calculate rank
        rankings = self.get_rankings(env_id, difficulty)
        rank = next(
            (i + 1 for i, r in enumerate(rankings) if r["agent_id"] == agent_id),
            len(rankings),
        )
        entry["rank"] = rank
        entry["total_entries"] = len(rankings)

        return entry

    def get_rankings(
        self,
        env_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get leaderboard rankings.

        Parameters
        ----------
        env_id : str or None
            Filter by environment. None returns all.
        difficulty : str or None
            Filter by difficulty level.
        top_k : int
            Maximum number of entries to return.
        """
        entries = self._data["entries"]

        if env_id:
            entries = [e for e in entries if e["env_id"] == env_id]
        if difficulty:
            entries = [e for e in entries if e.get("difficulty") == difficulty]

        # Keep best score per agent (per env)
        best_scores: Dict[str, Dict] = {}
        for e in entries:
            key = f"{e['agent_id']}:{e['env_id']}:{e.get('difficulty', '')}"
            if key not in best_scores or e["score"] > best_scores[key]["score"]:
                best_scores[key] = e

        # Sort by score descending
        rankings = sorted(best_scores.values(), key=lambda x: x["score"], reverse=True)

        # Add rank
        for i, r in enumerate(rankings[:top_k]):
            r["rank"] = i + 1

        return rankings[:top_k]

    def get_agent_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all submissions for a specific agent."""
        return [
            e for e in self._data["entries"]
            if e["agent_id"] == agent_id
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall leaderboard statistics."""
        entries = self._data["entries"]
        agents = set(e["agent_id"] for e in entries)
        envs = set(e["env_id"] for e in entries)

        return {
            "total_submissions": len(entries),
            "unique_agents": len(agents),
            "unique_environments": len(envs),
            "agents": sorted(agents),
            "environments": sorted(envs),
        }

    def clear(self):
        """Clear all leaderboard data."""
        self._data = {"entries": []}
        self._save()
