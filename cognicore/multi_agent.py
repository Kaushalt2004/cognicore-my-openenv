"""
Multi-Agent Environment Base — Environments where multiple agents interact.

Usage::

    from cognicore.multi_agent import MultiAgentEnv

    class DebateEnv(MultiAgentEnv):
        def _setup(self, **kw): ...
        def _generate_tasks(self): ...
        def _evaluate_multi(self, actions): ...
        def _get_obs_for_agent(self, agent_id): ...
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import CogniCoreConfig, EvalResult, StructuredReward


class MultiAgentEnv(CogniCoreEnv):
    """Base class for multi-agent CogniCore environments.

    Extends CogniCoreEnv to support multiple agents interacting
    in the same environment. Each agent has its own memory and
    reward signal.

    Subclasses implement:
      - ``_setup()`` — configure the environment
      - ``_generate_tasks()`` — generate tasks
      - ``_evaluate_multi(actions)`` — evaluate all agents' actions
      - ``_get_obs_for_agent(agent_id)`` — build observation for one agent
    """

    def __init__(
        self,
        num_agents: int = 2,
        agent_ids: Optional[List[str]] = None,
        config: Optional[CogniCoreConfig] = None,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.agent_ids = agent_ids or [f"agent_{i}" for i in range(num_agents)]
        self._agent_actions: Dict[str, Dict[str, Any]] = {}
        self._turn_order: List[str] = list(self.agent_ids)
        self._current_agent_idx: int = 0
        super().__init__(config=config, **kwargs)

    @property
    def current_agent(self) -> str:
        """ID of the agent whose turn it is."""
        return self._turn_order[self._current_agent_idx % len(self._turn_order)]

    def reset(self, **kwargs):
        """Reset and return observations for all agents."""
        self._agent_actions = {}
        self._current_agent_idx = 0
        obs = super().reset(**kwargs)

        # Build per-agent observations
        multi_obs = {}
        for aid in self.agent_ids:
            agent_obs = self._get_obs_for_agent(aid)
            agent_obs["your_agent_id"] = aid
            agent_obs["turn"] = self._turn_order[0]
            agent_obs["step"] = self._current_step
            agent_obs["max_steps"] = self._max_steps
            multi_obs[aid] = agent_obs

        return multi_obs

    def step_agent(
        self,
        agent_id: str,
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit an action for one agent.

        If all agents have acted, the step is evaluated and results
        are returned for all agents.
        """
        self._agent_actions[agent_id] = action
        self._current_agent_idx += 1

        # If all agents have acted, evaluate
        if len(self._agent_actions) >= self.num_agents:
            return self._resolve_step()

        return {
            "status": "waiting",
            "agent": agent_id,
            "waiting_for": [
                a for a in self.agent_ids if a not in self._agent_actions
            ],
        }

    def _resolve_step(self) -> Dict[str, Any]:
        """Evaluate all agents' actions and return results."""
        results = self._evaluate_multi(dict(self._agent_actions))

        # Build per-agent results
        multi_results = {}
        for aid in self.agent_ids:
            agent_result = results.get(aid, {})
            eval_result = agent_result.get("eval_result", EvalResult(
                base_score=0.0, correct=False, category="multi_agent",
            ))

            # Use base class step for the first agent to advance state
            multi_results[aid] = {
                "eval_result": {
                    "base_score": eval_result.base_score,
                    "correct": eval_result.correct,
                    "category": eval_result.category,
                },
                "reward": eval_result.base_score,
                "message": agent_result.get("message", ""),
            }

        # Advance step
        self._current_step += 1
        if self._current_step >= self._max_steps:
            self._done = True

        # Reset for next round
        self._agent_actions = {}
        self._current_agent_idx = 0

        multi_results["_done"] = self._done
        multi_results["_step"] = self._current_step

        return multi_results

    # --- Abstract methods ---

    @abstractmethod
    def _evaluate_multi(
        self,
        actions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all agents' actions simultaneously.

        Returns a dict mapping agent_id to result dict with
        ``eval_result`` (EvalResult) and optional ``message``.
        """
        ...

    @abstractmethod
    def _get_obs_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """Build observation for a specific agent."""
        ...

    # base_env requires these — delegate to multi-agent versions
    def _evaluate(self, action):
        return EvalResult(base_score=0, correct=False, category="multi_agent")

    def _get_obs(self):
        return self._get_obs_for_agent(self.current_agent)


class DebateEnv(MultiAgentEnv):
    """Built-in debate environment — two agents argue opposing sides.

    Both agents see a topic and must make arguments. A judge
    (rule-based) scores argument quality based on key points.
    """

    TOPICS = [
        {
            "topic": "Should AI development be paused?",
            "pro_points": ["safety", "alignment", "risk", "regulation"],
            "con_points": ["progress", "competition", "benefits", "innovation"],
            "category": "ai_policy",
        },
        {
            "topic": "Is social media net positive for society?",
            "pro_points": ["connection", "awareness", "business", "education"],
            "con_points": ["mental health", "misinformation", "addiction", "privacy"],
            "category": "technology",
        },
        {
            "topic": "Should college be free?",
            "pro_points": ["equality", "access", "workforce", "economy"],
            "con_points": ["cost", "quality", "taxes", "value"],
            "category": "education",
        },
        {
            "topic": "Is remote work better than office work?",
            "pro_points": ["flexibility", "productivity", "commute", "balance"],
            "con_points": ["collaboration", "culture", "mentoring", "boundaries"],
            "category": "workplace",
        },
        {
            "topic": "Should autonomous vehicles be allowed on public roads?",
            "pro_points": ["safety", "efficiency", "accessibility", "environment"],
            "con_points": ["liability", "jobs", "technology", "trust"],
            "category": "transportation",
        },
    ]

    def __init__(self, config=None, **kwargs):
        super().__init__(num_agents=2, agent_ids=["pro", "con"], config=config, **kwargs)

    def _setup(self, **kwargs):
        pass

    def _generate_tasks(self):
        return self.TOPICS

    def _evaluate_multi(self, actions):
        task = self._tasks[self._current_step]
        results = {}

        for aid in self.agent_ids:
            action = actions.get(aid, {})
            argument = str(action.get("argument", "")).lower()

            # Score based on key points hit
            points = task["pro_points"] if aid == "pro" else task["con_points"]
            hits = sum(1 for p in points if p.lower() in argument)
            score = hits / len(points) if points else 0

            results[aid] = {
                "eval_result": EvalResult(
                    base_score=score,
                    correct=score >= 0.5,
                    category=task["category"],
                    predicted=argument[:100],
                    ground_truth=", ".join(points),
                ),
                "message": f"Hit {hits}/{len(points)} key points",
            }

        return results

    def _get_obs_for_agent(self, agent_id):
        task = self._tasks[self._current_step] if self._current_step < len(self._tasks) else {}
        return {
            "topic": task.get("topic", ""),
            "your_side": agent_id,
            "category": task.get("category", ""),
        }
