"""
CogniCore Types — Core data structures for the framework.

Defines the typed contracts between environments, middleware, and agents.
All types use dataclasses for zero-dependency lightweight serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Structured Reward — CogniCore's signature 8-component reward
# ---------------------------------------------------------------------------

@dataclass
class StructuredReward:
    """8-component reward signal — richer than a single scalar.

    Every CogniCore environment returns this instead of a plain float.
    Each component is independently interpretable, letting agents learn
    *why* they were rewarded or penalized.
    """

    base_score: float = 0.0
    """Raw correctness score from the environment grader (0.0–1.0)."""

    memory_bonus: float = 0.0
    """Bonus for consistency with successful past actions in memory."""

    reflection_bonus: float = 0.0
    """Bonus for following a reflection hint that was provided."""

    streak_penalty: float = 0.0
    """Penalty for consecutive incorrect actions (negative value)."""

    propose_bonus: float = 0.0
    """Bonus for improving from a PROPOSE draft to the final REVISE action."""

    novelty_bonus: float = 0.0
    """Bonus for correctly handling a category/type never seen before."""

    confidence_cal: float = 0.0
    """Calibration score: rewarded for high confidence + correct, penalized for overconfidence + wrong."""

    time_decay: float = 0.0
    """Small decay for very slow responses (encourages efficiency)."""

    @property
    def total(self) -> float:
        """Sum all 8 components into a single scalar."""
        return (
            self.base_score
            + self.memory_bonus
            + self.reflection_bonus
            + self.streak_penalty
            + self.propose_bonus
            + self.novelty_bonus
            + self.confidence_cal
            + self.time_decay
        )

    def as_float(self) -> float:
        """Single float for Gymnasium / scalar-reward compatibility."""
        return self.total

    def to_dict(self) -> Dict[str, float]:
        """Serialize all components to a dict."""
        d = asdict(self)
        d["total"] = self.total
        return d

    def __repr__(self) -> str:
        return (
            f"StructuredReward(total={self.total:.4f}, "
            f"base={self.base_score:.2f}, mem={self.memory_bonus:.2f}, "
            f"refl={self.reflection_bonus:.2f}, streak={self.streak_penalty:.2f}, "
            f"propose={self.propose_bonus:.2f}, novelty={self.novelty_bonus:.2f}, "
            f"conf={self.confidence_cal:.2f}, time={self.time_decay:.2f})"
        )


# ---------------------------------------------------------------------------
# Evaluation Result — returned by subclass _evaluate()
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of grading an agent's action.

    Returned by ``CogniCoreEnv._evaluate()`` — the environment-specific
    grading logic.  The base class then wraps this into a full
    ``StructuredReward``.
    """

    base_score: float
    """Raw score from the grader (0.0–1.0 typical, but not enforced)."""

    correct: bool
    """Whether the action was fully correct."""

    ground_truth: Any = None
    """The correct answer (for logging/display)."""

    predicted: Any = None
    """The agent's answer (for logging/display)."""

    category: str = ""
    """Task category / grouping key for memory and reflection."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra info from the environment grader."""


# ---------------------------------------------------------------------------
# Step Result — the 5-tuple returned by env.step()
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """The return value of ``CogniCoreEnv.step()``.

    Follows the Gymnasium 5-tuple convention:
      (observation, reward, terminated, truncated, info)
    """

    observation: Dict[str, Any]
    reward: StructuredReward
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self):
        """Unpack into a Gymnasium-compatible 5-tuple."""
        return (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
        )


# ---------------------------------------------------------------------------
# Episode Stats — summary at end of an episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Summary statistics for a completed episode."""

    episode_number: int = 0
    steps: int = 0
    total_reward: float = 0.0
    accuracy: float = 0.0
    correct_count: int = 0
    memory_entries_created: int = 0
    reflection_hints_given: int = 0
    proposals_made: int = 0
    proposal_improvements: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Proposal Feedback — returned by env.propose()
# ---------------------------------------------------------------------------

@dataclass
class ProposalFeedback:
    """Feedback from a PROPOSE action (before the real step).

    The agent gets this without the step counting — it's free exploration.
    """

    memory_context: List[Dict[str, Any]] = field(default_factory=list)
    """Relevant past experiences from memory."""

    reflection_hint: Optional[str] = None
    """Hint from the reflection engine, if available."""

    confidence_estimate: float = 0.5
    """Estimated confidence based on past performance in this category."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Extra info the environment wants to share."""


# ---------------------------------------------------------------------------
# Configuration — controls which middleware to enable
# ---------------------------------------------------------------------------

@dataclass
class CogniCoreConfig:
    """Configuration for CogniCore middleware.

    Pass to ``CogniCoreEnv.__init__()`` to toggle features.
    """

    # Memory
    enable_memory: bool = True
    memory_max_size: int = 10_000
    memory_similarity_key: str = "category"
    memory_retrieve_top_k: int = 3

    # Reflection
    enable_reflection: bool = True
    reflection_min_samples: int = 2
    reflection_failure_threshold: int = 2

    # Safety monitor
    enable_safety_monitor: bool = True
    streak_threshold: int = 3
    streak_penalty: float = -0.1

    # PROPOSE → Revise
    enable_propose_revise: bool = True
    max_proposals_per_step: int = 1
    propose_improvement_bonus: float = 0.05

    # Reward tuning
    memory_bonus_value: float = 0.05
    reflection_bonus_value: float = 0.03
    novelty_bonus_value: float = 0.04
    confidence_bonus_scale: float = 0.02
    time_decay_rate: float = 0.001
    time_decay_threshold_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
