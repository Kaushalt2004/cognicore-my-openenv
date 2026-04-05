"""Pydantic models for the AI Safety Monitor environment."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class SafetyLabel(str, Enum):
    """Classification labels for AI response safety."""
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class SafetyCase(BaseModel):
    """A single AI safety evaluation case."""
    id: str
    prompt: str
    response: str
    ground_truth: SafetyLabel
    category: str
    difficulty: str
    explanation: str


class Observation(BaseModel):
    """What the agent sees at each step."""
    case_id: str
    prompt: str
    response: str
    difficulty: str
    category: str
    memory_context: List[Dict[str, Any]] = Field(default_factory=list)
    reflection_hint: Optional[str] = None
    step: int = 0
    max_steps: int = 30


class Action(BaseModel):
    """What the agent outputs at each step."""
    classification: SafetyLabel
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: Optional[str] = None


class RewardInfo(BaseModel):
    """Reward information returned after each step."""
    reward: float
    correct: bool
    ground_truth: SafetyLabel
    predicted: SafetyLabel
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
