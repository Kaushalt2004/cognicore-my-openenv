"""Pydantic models for the AI Safety Monitor environment.

Enhanced with multi-dimensional observation/action spaces for richer
RL signal: confidence calibration, severity assessment, manipulation detection.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


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


class ContentType(str, Enum):
    """Type of content being evaluated."""
    TEXT = "text"
    CODE = "code"
    CONVERSATION = "conversation"
    MULTI_TURN = "multi_turn"


class SafetyCase(BaseModel):
    """A single AI safety evaluation case — expanded with rich metadata."""
    id: str
    prompt: str
    response: str
    ground_truth: SafetyLabel
    category: str
    difficulty: str
    explanation: str
    severity: Severity = Severity.MEDIUM
    content_type: ContentType = ContentType.TEXT
    tags: List[str] = Field(default_factory=list)
    manipulation_technique: Optional[str] = None  # For hard cases


class Observation(BaseModel):
    """What the agent sees at each step — enhanced with episode context."""
    case_id: str
    prompt: str
    response: str
    difficulty: str
    category: str
    content_type: str = "text"
    tags: List[str] = Field(default_factory=list)
    memory_context: List[Dict[str, Any]] = Field(default_factory=list)
    reflection_hint: Optional[str] = None
    step: int = 0
    max_steps: int = 10
    episode_stats: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """What the agent outputs at each step — multi-dimensional."""
    classification: SafetyLabel
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    severity: str = "medium"
    reasoning: Optional[str] = None
    manipulation_type: Optional[str] = None


class RewardInfo(BaseModel):
    """Detailed reward breakdown returned after each step."""
    reward: float
    correct: bool
    ground_truth: SafetyLabel
    predicted: SafetyLabel
    base_reward: float = 0.0
    confidence_bonus: float = 0.0
    severity_bonus: float = 0.0
    reasoning_bonus: float = 0.0
    memory_bonus: float = 0.0
    streak_penalty: float = 0.0
    details: str = ""


class EnvState(BaseModel):
    """Full environment state for the /state endpoint."""
    current_step: int
    max_steps: int
    total_reward: float
    done: bool
    task_name: str
    difficulty: str
    cases_seen: int
    correct_count: int
    wrong_streak: int
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    category_accuracy: Dict[str, float] = Field(default_factory=dict)
