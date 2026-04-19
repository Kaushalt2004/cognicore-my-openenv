"""
Persistent Memory Manager — Save/load agent memory across sessions.

Usage::

    from cognicore.memory_manager import MemoryManager

    mgr = MemoryManager(storage_dir="./cognicore_data")
    mgr.save_session("agent-1", env)
    mgr.load_session("agent-1", env)
    mgr.list_sessions()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from cognicore.core.base_env import CogniCoreEnv


class MemoryManager:
    """Persistent memory across CogniCore sessions.

    Saves and loads agent memory, episode history, and leaderboard
    data to/from disk. Each agent gets a directory with:
    - ``memory.json`` — full memory entries
    - ``history.json`` — episode history (scores, accuracy, etc.)
    - ``metadata.json`` — agent metadata (created, last run, etc.)
    """

    def __init__(self, storage_dir: str = "./cognicore_data"):
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

    def _agent_dir(self, agent_id: str) -> str:
        d = os.path.join(self.storage_dir, agent_id)
        os.makedirs(d, exist_ok=True)
        return d

    def save_session(
        self,
        agent_id: str,
        env: CogniCoreEnv,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save environment memory and episode stats for an agent.

        Returns the path to the saved session directory.
        """
        d = self._agent_dir(agent_id)

        # Save memory entries
        memory_path = os.path.join(d, "memory.json")
        with open(memory_path, "w") as f:
            json.dump(env.memory.entries, f, indent=2, default=str)

        # Append to episode history
        history_path = os.path.join(d, "history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)

        stats = env.episode_stats()
        history.append({
            "episode": stats.episode_number,
            "steps": stats.steps,
            "total_reward": stats.total_reward,
            "accuracy": stats.accuracy,
            "correct_count": stats.correct_count,
            "score": env.get_score(),
            "memory_entries": stats.memory_entries_created,
            "timestamp": time.time(),
        })

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save metadata
        meta_path = os.path.join(d, "metadata.json")
        meta = {
            "agent_id": agent_id,
            "last_saved": time.time(),
            "total_episodes": len(history),
            "env_class": env.__class__.__name__,
            **(metadata or {}),
        }

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                existing = json.load(f)
            meta["created"] = existing.get("created", time.time())
        else:
            meta["created"] = time.time()

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return d

    def load_session(self, agent_id: str, env: CogniCoreEnv) -> bool:
        """Load saved memory into an environment.

        Returns True if memory was loaded, False if no saved data exists.
        """
        d = self._agent_dir(agent_id)
        memory_path = os.path.join(d, "memory.json")

        if not os.path.exists(memory_path):
            return False

        with open(memory_path) as f:
            entries = json.load(f)

        env.memory.entries = entries
        return True

    def get_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get episode history for an agent."""
        history_path = os.path.join(self._agent_dir(agent_id), "history.json")
        if not os.path.exists(history_path):
            return []
        with open(history_path) as f:
            return json.load(f)

    def get_metadata(self, agent_id: str) -> Dict[str, Any]:
        """Get agent metadata."""
        meta_path = os.path.join(self._agent_dir(agent_id), "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved agent sessions."""
        sessions = []
        if not os.path.exists(self.storage_dir):
            return sessions

        for name in sorted(os.listdir(self.storage_dir)):
            agent_dir = os.path.join(self.storage_dir, name)
            if os.path.isdir(agent_dir):
                meta = self.get_metadata(name)
                history = self.get_history(name)
                sessions.append({
                    "agent_id": name,
                    "total_episodes": len(history),
                    "best_score": max((h["score"] for h in history), default=0),
                    "best_accuracy": max((h["accuracy"] for h in history), default=0),
                    "last_saved": meta.get("last_saved"),
                    "env_class": meta.get("env_class", "unknown"),
                })
        return sessions

    def delete_session(self, agent_id: str) -> bool:
        """Delete all saved data for an agent."""
        import shutil
        d = os.path.join(self.storage_dir, agent_id)
        if os.path.exists(d):
            shutil.rmtree(d)
            return True
        return False
