"""
CogniCore Environment Registry — Gymnasium-style ``make()`` factory.

Provides ``make()``, ``register()``, and ``list_envs()`` for discovering
and instantiating CogniCore environments.

Usage::

    import cognicore

    # List available environments
    cognicore.list_envs()

    # Create an environment
    env = cognicore.make("SafetyClassification-v1", difficulty="hard")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from cognicore.core.base_env import CogniCoreEnv


# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register(
    env_id: str,
    entry_point: Type[CogniCoreEnv] | str,
    *,
    description: str = "",
    default_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Register an environment.

    Parameters
    ----------
    env_id : str
        Unique identifier (e.g. ``"SafetyClassification-v1"``).
    entry_point : type or str
        The environment class, or a dotted import path
        (e.g. ``"cognicore.envs.safety_classification:SafetyClassificationEnv"``).
    description : str
        Human-readable description.
    default_kwargs : dict or None
        Default keyword arguments passed to the constructor.
    """
    _REGISTRY[env_id] = {
        "entry_point": entry_point,
        "description": description,
        "default_kwargs": default_kwargs or {},
    }


def make(env_id: str, **kwargs: Any) -> CogniCoreEnv:
    """Create an environment by ID.

    Parameters
    ----------
    env_id : str
        Registered environment identifier.
    **kwargs
        Passed to the environment constructor (override defaults).

    Returns
    -------
    CogniCoreEnv
        An initialized environment instance.

    Raises
    ------
    InvalidEnvironmentError
        If ``env_id`` is not registered.
    """
    if env_id not in _REGISTRY:
        from cognicore.core.errors import InvalidEnvironmentError
        raise InvalidEnvironmentError(env_id, list(_REGISTRY.keys()))

    spec = _REGISTRY[env_id]
    entry_point = spec["entry_point"]

    # Resolve string entry points
    if isinstance(entry_point, str):
        module_path, class_name = entry_point.rsplit(":", 1)
        import importlib

        module = importlib.import_module(module_path)
        entry_point = getattr(module, class_name)

    # Merge default kwargs with user kwargs
    merged_kwargs = {**spec["default_kwargs"], **kwargs}

    return entry_point(**merged_kwargs)


def list_envs() -> List[Dict[str, str]]:
    """List all registered environments.

    Returns
    -------
    list of dict
        Each dict has ``"id"`` and ``"description"`` keys.
    """
    return [
        {"id": env_id, "description": spec["description"]}
        for env_id, spec in sorted(_REGISTRY.items())
    ]


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

# Safety Classification — three difficulty presets
register(
    "SafetyClassification-v1",
    entry_point="cognicore.envs.safety_classification:SafetyClassificationEnv",
    description="AI Safety classification: classify AI responses as SAFE/UNSAFE/NEEDS_REVIEW. 30 cases across 3 difficulty levels.",
    default_kwargs={"difficulty": "easy"},
)

register(
    "SafetyClassification-Easy-v1",
    entry_point="cognicore.envs.safety_classification:SafetyClassificationEnv",
    description="AI Safety classification — Easy (10 binary cases, expected 90%+).",
    default_kwargs={"difficulty": "easy"},
)

register(
    "SafetyClassification-Medium-v1",
    entry_point="cognicore.envs.safety_classification:SafetyClassificationEnv",
    description="AI Safety classification — Medium (10 nuanced cases, expected 70%+).",
    default_kwargs={"difficulty": "medium"},
)

register(
    "SafetyClassification-Hard-v1",
    entry_point="cognicore.envs.safety_classification:SafetyClassificationEnv",
    description="AI Safety classification — Hard (10 adversarial cases, expected 50%+).",
    default_kwargs={"difficulty": "hard"},
)

# Math Reasoning
register(
    "MathReasoning-v1",
    entry_point="cognicore.envs.math_reasoning:MathReasoningEnv",
    description="Math reasoning: solve arithmetic, algebra, and logic problems. 30 cases across 3 difficulty levels.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "MathReasoning-Easy-v1",
    entry_point="cognicore.envs.math_reasoning:MathReasoningEnv",
    description="Math reasoning — Easy (10 arithmetic problems).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "MathReasoning-Medium-v1",
    entry_point="cognicore.envs.math_reasoning:MathReasoningEnv",
    description="Math reasoning — Medium (10 algebra/logic problems).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "MathReasoning-Hard-v1",
    entry_point="cognicore.envs.math_reasoning:MathReasoningEnv",
    description="Math reasoning — Hard (10 advanced math problems).",
    default_kwargs={"difficulty": "hard"},
)

# Code Debugging
register(
    "CodeDebugging-v1",
    entry_point="cognicore.envs.code_debugging:CodeDebuggingEnv",
    description="Code debugging: find and fix bugs in Python code. 30 cases across 3 difficulty levels.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "CodeDebugging-Easy-v1",
    entry_point="cognicore.envs.code_debugging:CodeDebuggingEnv",
    description="Code debugging — Easy (10 syntax/obvious bugs).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "CodeDebugging-Medium-v1",
    entry_point="cognicore.envs.code_debugging:CodeDebuggingEnv",
    description="Code debugging — Medium (10 logic/algorithm bugs).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "CodeDebugging-Hard-v1",
    entry_point="cognicore.envs.code_debugging:CodeDebuggingEnv",
    description="Code debugging — Hard (10 subtle/security bugs).",
    default_kwargs={"difficulty": "hard"},
)

# Conversation / Negotiation
register(
    "Conversation-v1",
    entry_point="cognicore.envs.conversation:ConversationEnv",
    description="Conversation: choose best responses in dialogue scenarios. 30 cases across 3 difficulty levels.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "Conversation-Easy-v1",
    entry_point="cognicore.envs.conversation:ConversationEnv",
    description="Conversation — Easy (10 clear-cut dialogue scenarios).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "Conversation-Medium-v1",
    entry_point="cognicore.envs.conversation:ConversationEnv",
    description="Conversation — Medium (10 nuanced negotiation scenarios).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "Conversation-Hard-v1",
    entry_point="cognicore.envs.conversation:ConversationEnv",
    description="Conversation — Hard (10 complex negotiation/ethical scenarios).",
    default_kwargs={"difficulty": "hard"},
)

# Multi-Step Planning
register(
    "Planning-v1",
    entry_point="cognicore.envs.multi_step_planning:MultiStepPlanningEnv",
    description="Multi-step planning: order steps correctly to solve planning problems. 30 cases across 3 difficulty levels.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "Planning-Easy-v1",
    entry_point="cognicore.envs.multi_step_planning:MultiStepPlanningEnv",
    description="Planning — Easy (10 sequential task ordering problems).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "Planning-Medium-v1",
    entry_point="cognicore.envs.multi_step_planning:MultiStepPlanningEnv",
    description="Planning — Medium (10 dependency/constraint problems).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "Planning-Hard-v1",
    entry_point="cognicore.envs.multi_step_planning:MultiStepPlanningEnv",
    description="Planning — Hard (10 optimization/conflict resolution problems).",
    default_kwargs={"difficulty": "hard"},
)

# Text Summarization
register(
    "Summarization-v1",
    entry_point="cognicore.envs.text_summarization:TextSummarizationEnv",
    description="Text summarization: summarize passages with key point coverage. 30 cases across 3 difficulty levels.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "Summarization-Easy-v1",
    entry_point="cognicore.envs.text_summarization:TextSummarizationEnv",
    description="Summarization — Easy (10 news article summaries).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "Summarization-Medium-v1",
    entry_point="cognicore.envs.text_summarization:TextSummarizationEnv",
    description="Summarization — Medium (10 technical/academic summaries).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "Summarization-Hard-v1",
    entry_point="cognicore.envs.text_summarization:TextSummarizationEnv",
    description="Summarization — Hard (10 long, nuanced multi-viewpoint summaries).",
    default_kwargs={"difficulty": "hard"},
)

# ── Real-World Environments ──────────────────────────────────────────

register(
    "RealWorldSafety-v1",
    entry_point="cognicore.envs.real_safety:RealWorldSafetyEnv",
    description="Real-world AI safety: jailbreaks, PII leaks, prompt injection, and edge cases from production.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "RealWorldSafety-Easy-v1",
    entry_point="cognicore.envs.real_safety:RealWorldSafetyEnv",
    description="Real-world safety — Easy (clear-cut SAFE/UNSAFE, no ambiguous cases).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "RealWorldSafety-Medium-v1",
    entry_point="cognicore.envs.real_safety:RealWorldSafetyEnv",
    description="Real-world safety — Medium (mix of clear and ambiguous cases).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "RealWorldSafety-Hard-v1",
    entry_point="cognicore.envs.real_safety:RealWorldSafetyEnv",
    description="Real-world safety — Hard (includes NEEDS_REVIEW edge cases).",
    default_kwargs={"difficulty": "hard"},
)

register(
    "RealWorldCodeBugs-v1",
    entry_point="cognicore.envs.real_code_bugs:RealWorldCodeBugsEnv",
    description="Real-world Python bugs: SQL injection, race conditions, resource leaks, mutable defaults, and more.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "RealWorldCodeBugs-Hard-v1",
    entry_point="cognicore.envs.real_code_bugs:RealWorldCodeBugsEnv",
    description="Real-world code bugs — Hard (no hints, 12 bug categories).",
    default_kwargs={"difficulty": "hard"},
)

# ── RL Environments (actual reinforcement learning) ─────────────────

register(
    "GridWorld-v1",
    entry_point="cognicore.envs.gridworld:GridWorldEnv",
    description="GridWorld navigation: agent navigates NxN grid, avoids traps, reaches goal. Real RL with reward shaping.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "GridWorld-Easy-v1",
    entry_point="cognicore.envs.gridworld:GridWorldEnv",
    description="GridWorld — Easy (5x5 grid, 3 traps).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "GridWorld-Medium-v1",
    entry_point="cognicore.envs.gridworld:GridWorldEnv",
    description="GridWorld — Medium (7x7 grid, 7 traps).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "GridWorld-Hard-v1",
    entry_point="cognicore.envs.gridworld:GridWorldEnv",
    description="GridWorld — Hard (10x10 grid, 15 traps).",
    default_kwargs={"difficulty": "hard"},
)

register(
    "ResourceGathering-v1",
    entry_point="cognicore.envs.resource_gathering:ResourceGatheringEnv",
    description="Multi-objective resource gathering: food/wood/stone collection, building, energy management.",
    default_kwargs={"difficulty": "easy"},
)
register(
    "ResourceGathering-Easy-v1",
    entry_point="cognicore.envs.resource_gathering:ResourceGatheringEnv",
    description="ResourceGathering — Easy (high energy, cheap buildings).",
    default_kwargs={"difficulty": "easy"},
)
register(
    "ResourceGathering-Medium-v1",
    entry_point="cognicore.envs.resource_gathering:ResourceGatheringEnv",
    description="ResourceGathering — Medium (moderate energy, balanced costs).",
    default_kwargs={"difficulty": "medium"},
)
register(
    "ResourceGathering-Hard-v1",
    entry_point="cognicore.envs.resource_gathering:ResourceGatheringEnv",
    description="ResourceGathering — Hard (low energy, expensive buildings, scarce resources).",
    default_kwargs={"difficulty": "hard"},
)
