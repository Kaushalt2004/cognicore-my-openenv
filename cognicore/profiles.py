"""
CogniCore Config Profiles — Preset configurations for different use cases.

Usage::

    from cognicore.profiles import get_profile, list_profiles

    config = get_profile("strict_safety")
    env = cognicore.make("SafetyClassification-v1", config=config)
"""

from __future__ import annotations

from typing import Dict, List
from cognicore.core.types import CogniCoreConfig


PROFILES = {
    "default": {
        "description": "Balanced defaults for general use",
        "config": {},
    },
    "strict_safety": {
        "description": "Maximum safety — harsh penalties, conservative memory",
        "config": {
            "streak_penalty": -0.20,
            "memory_bonus_value": 0.03,
            "confidence_bonus_scale": 0.05,
            "memory_max_size": 5000,
        },
    },
    "fast_explore": {
        "description": "Rapid exploration — high bonuses, low penalties",
        "config": {
            "novelty_bonus_value": 0.08,
            "propose_improvement_bonus": 0.10,
            "streak_penalty": -0.03,
            "memory_bonus_value": 0.08,
        },
    },
    "production": {
        "description": "Production deployment — balanced with cost awareness",
        "config": {
            "memory_bonus_value": 0.05,
            "streak_penalty": -0.10,
            "memory_max_size": 10000,
            "time_decay_threshold_seconds": 60.0,
        },
    },
    "research": {
        "description": "Research mode — detailed tracking, no time penalties",
        "config": {
            "time_decay_threshold_seconds": 999999.0,
            "novelty_bonus_value": 0.06,
            "reflection_bonus_value": 0.05,
        },
    },
    "competitive": {
        "description": "Competition mode — maximize score, strict grading",
        "config": {
            "streak_penalty": -0.15,
            "memory_bonus_value": 0.07,
            "novelty_bonus_value": 0.05,
            "confidence_bonus_scale": 0.04,
        },
    },
    "beginner": {
        "description": "Gentle mode for new agents — small penalties, big bonuses",
        "config": {
            "streak_penalty": -0.02,
            "memory_bonus_value": 0.10,
            "novelty_bonus_value": 0.08,
            "reflection_bonus_value": 0.06,
        },
    },
}


def get_profile(name: str) -> CogniCoreConfig:
    """Get a preset configuration profile.

    Available profiles: default, strict_safety, fast_explore,
    production, research, competitive, beginner
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")

    profile = PROFILES[name]
    return CogniCoreConfig(**profile["config"])


def list_profiles() -> List[Dict[str, str]]:
    """List all available profiles."""
    return [
        {"name": name, "description": p["description"]} for name, p in PROFILES.items()
    ]
