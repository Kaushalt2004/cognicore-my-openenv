"""
Data models for the CogniCore AI Safety Monitor Environment.

Uses the official OpenEnv SDK base types (Action, Observation, State)
to ensure full spec compliance with the OpenEnv framework.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


# ─── Enums ──────────────────────────────────────────────────

class SafetyLabel(str, Enum):
    """Classification labels for AI response safety."""
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Severity(str, Enum):
    """Severity level of unsafe content."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─── OpenEnv Action / Observation / State ───────────────────

class SafetyAction(Action):
    """Agent's classification action for a safety case.
    
    The agent must provide a classification label, and optionally 
    confidence, severity assessment, and reasoning for richer scoring.
    """
    classification: str = Field(
        ..., description="Safety label: SAFE, UNSAFE, or NEEDS_REVIEW"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Agent's confidence in classification (0.0-1.0)"
    )
    severity: str = Field(
        default="medium",
        description="Agent's severity assessment: low, medium, high, critical"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for the classification"
    )


class SafetyObservation(Observation):
    """Observation returned by the environment at each step.
    
    Contains the AI prompt/response to classify, plus CogniCore 
    memory context and reflection hints from past episodes.
    """
    case_id: str = Field(default="", description="Unique case identifier")
    prompt: str = Field(default="", description="User prompt sent to the AI")
    response: str = Field(default="", description="AI-generated response to classify")
    difficulty: str = Field(default="easy", description="Difficulty level")
    category: str = Field(default="", description="Content category")
    content_type: str = Field(default="text", description="Type of content")
    tags: List[str] = Field(default_factory=list, description="Safety-relevant tags")
    memory_context: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Past classifications in similar categories from CogniCore memory"
    )
    reflection_hint: Optional[str] = Field(
        default=None,
        description="Metacognitive hint from CogniCore reflection engine"
    )
    step: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=10, description="Max steps in episode")
    episode_accuracy: float = Field(
        default=0.0,
        description="Agent's running accuracy in this episode"
    )


class SafetyReward(BaseModel):
    """Structured reward with full penalty breakdown.
    
    Provides transparent scoring so agents can learn from each component.
    """
    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., ge=-1.0, le=1.0,
        description="Net reward after all bonuses and penalties")
    base_score: float = Field(..., ge=0.0, le=1.0,
        description="Raw grader score before bonuses/penalties")
    delta: float = Field(default=0.0, ge=-1.0, le=1.0,
        description="Improvement over previous best score")
    memory_bonus: float = Field(default=0.0, ge=0.0, le=0.1,
        description="Bonus for consistency with past correct answers")
    streak_penalty: float = Field(default=0.0, ge=-1.0, le=0.0,
        description="Penalty for consecutive wrong answers")
    confidence_penalty: float = Field(default=0.0, ge=-1.0, le=0.0,
        description="Penalty for miscalibrated confidence")
    step_penalty: float = Field(default=-0.01,
        description="Small fixed cost per step to encourage efficiency")
    explanation: str = Field(default="",
        description="Human-readable explanation of the score")


class StepInfo(BaseModel):
    """Detailed info returned with each step for agent debugging."""
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(default="", description="Case that was evaluated")
    ground_truth: Optional[str] = Field(default=None, description="Correct label")
    predicted: Optional[str] = Field(default=None, description="Agent's prediction")
    correct: bool = Field(default=False, description="Whether prediction was correct")
    current_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Score for this step")
    best_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Best score achieved so far in episode")
    terminated_by: Optional[str] = Field(default=None,
        description="What ended the episode: max_steps, perfect_score, or None")
    error: Optional[str] = Field(default=None, description="Error message if any")


class SafetyState(State):
    """Extended state for the safety monitor environment."""
    task_name: str = Field(default="", description="Name of current task")
    difficulty: str = Field(default="", description="Current difficulty")
    current_step: int = Field(default=0, description="Current step")
    max_steps: int = Field(default=10, description="Max steps in task")
    total_reward: float = Field(default=0.0, description="Accumulated reward")
    best_score: float = Field(default=0.0, description="Best score in episode")
    cases_seen: int = Field(default=0, description="Cases processed")
    correct_count: int = Field(default=0, description="Correct classifications")
    accuracy: float = Field(default=0.0, description="Running accuracy")
    wrong_streak: int = Field(default=0, description="Current consecutive errors")
    category_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-category accuracy statistics"
    )


# ─── Internal models (not part of OpenEnv interface) ────────

class SafetyCase:
    """A single AI safety evaluation case with rich metadata."""
    def __init__(
        self,
        id: str,
        prompt: str,
        response: str,
        ground_truth: SafetyLabel,
        category: str,
        difficulty: str,
        explanation: str,
        severity: Severity = Severity.MEDIUM,
        content_type: str = "text",
        tags: List[str] = None,
        manipulation_technique: Optional[str] = None,
    ):
        self.id = id
        self.prompt = prompt
        self.response = response
        self.ground_truth = ground_truth
        self.category = category
        self.difficulty = difficulty
        self.explanation = explanation
        self.severity = severity
        self.content_type = content_type
        self.tags = tags or []
        self.manipulation_technique = manipulation_technique


__all__ = [
    "SafetyLabel",
    "Severity",
    "SafetyAction",
    "SafetyObservation",
    "SafetyReward",
    "StepInfo",
    "SafetyState",
    "SafetyCase",
]
