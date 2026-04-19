"""
CogniCore Agent Persistence — Save and load trained agents.

Preserves learned knowledge, genome, and history so agents survive
across sessions.

Usage::

    from cognicore.persistence import save_agent, load_agent

    save_agent(agent, "my_expert.json")
    agent = load_agent("my_expert.json")
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional


def save_agent(agent, path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Save a trained agent to disk.

    Serializes the agent's knowledge, genome, history, and configuration
    to a JSON file that can be loaded later.

    Parameters
    ----------
    agent : any agent with knowledge/genome attributes
    path : str
        File path to save to (JSON format).
    metadata : dict, optional
        Extra metadata (name, description, etc.)

    Returns
    -------
    dict with save info
    """
    data = {
        "_cognicore_agent": True,
        "_version": "1.0",
        "_saved_at": time.time(),
        "_agent_type": type(agent).__name__,
        "metadata": metadata or {},
    }

    # Save knowledge (AutoLearner, SafeAgent, AdaptiveAgent, EvolvableAgent)
    knowledge = getattr(agent, "knowledge", None)
    if knowledge is not None:
        if isinstance(knowledge, defaultdict):
            # Convert nested defaultdicts to regular dicts
            data["knowledge"] = {
                k: dict(v) if isinstance(v, dict) else v
                for k, v in knowledge.items()
            }
        elif isinstance(knowledge, dict):
            data["knowledge"] = dict(knowledge)

    # Save genome (EvolvableAgent)
    genome = getattr(agent, "genome", None)
    if genome is not None:
        data["genome"] = genome

    # Save history
    history = getattr(agent, "history", None)
    if history is not None:
        data["history"] = list(history)[-1000:]  # cap at 1000

    # Save config
    for attr in ("epsilon", "name", "threshold", "fitness", "generation",
                 "_correct", "_total", "actions"):
        val = getattr(agent, attr, None)
        if val is not None:
            data[attr] = val

    # Write
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    size = os.path.getsize(path)
    return {
        "path": path,
        "size_bytes": size,
        "agent_type": data["_agent_type"],
        "knowledge_categories": len(data.get("knowledge", {})),
        "history_entries": len(data.get("history", [])),
    }


def load_agent(path: str, agent_class=None):
    """Load a saved agent from disk.

    Parameters
    ----------
    path : str
        Path to the saved agent JSON file.
    agent_class : class, optional
        Agent class to instantiate. If None, auto-detects.

    Returns
    -------
    Instantiated agent with restored knowledge.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("_cognicore_agent"):
        raise ValueError("Not a CogniCore agent file")

    agent_type = data.get("_agent_type", "AutoLearner")

    # Auto-detect class
    if agent_class is None:
        from cognicore.smart_agents import AutoLearner, SafeAgent, AdaptiveAgent
        class_map = {
            "AutoLearner": AutoLearner,
            "SafeAgent": SafeAgent,
            "AdaptiveAgent": AdaptiveAgent,
        }
        try:
            from cognicore.evolution import EvolvableAgent
            class_map["EvolvableAgent"] = EvolvableAgent
        except ImportError:
            pass

        agent_class = class_map.get(agent_type)
        if agent_class is None:
            from cognicore.smart_agents import AutoLearner
            agent_class = AutoLearner

    # Instantiate
    if agent_type == "EvolvableAgent" and "genome" in data:
        agent = agent_class(genome=data["genome"])
    else:
        agent = agent_class()

    # Restore knowledge
    if "knowledge" in data and hasattr(agent, "knowledge"):
        for k, v in data["knowledge"].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    agent.knowledge[k][k2] = v2
            else:
                agent.knowledge[k] = v

    # Restore attributes
    for attr in ("epsilon", "name", "threshold", "fitness", "generation",
                 "_correct", "_total"):
        if attr in data and hasattr(agent, attr):
            setattr(agent, attr, data[attr])

    if "history" in data and hasattr(agent, "history"):
        agent.history = data["history"]

    return agent
