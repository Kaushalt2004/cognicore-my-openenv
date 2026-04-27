"""
CogniCore Spaces — Lightweight action/observation space definitions.

These are dependency-free alternatives to ``gymnasium.spaces``.
They describe what an agent can see and do, enabling introspection
without requiring NumPy or Gymnasium as dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiscreteSpace:
    """A discrete set of valid actions/observations.

    Examples:
        >>> s = DiscreteSpace(3, labels=["SAFE", "UNSAFE", "NEEDS_REVIEW"])
        >>> s.contains("SAFE")
        True
        >>> s.sample()  # returns a random label
    """

    n: int
    """Number of discrete values."""

    labels: Optional[List[str]] = None
    """Human-readable labels (one per value). If None, values are 0..n-1."""

    def contains(self, value: Any) -> bool:
        """Check if *value* is valid in this space."""
        if self.labels is not None:
            return value in self.labels
        if isinstance(value, int):
            return 0 <= value < self.n
        return False

    def sample(self) -> Any:
        """Return a random valid value."""
        import random

        if self.labels:
            return random.choice(self.labels)
        return random.randint(0, self.n - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "discrete", "n": self.n, "labels": self.labels}

    def __repr__(self) -> str:
        if self.labels:
            return f"DiscreteSpace(n={self.n}, labels={self.labels})"
        return f"DiscreteSpace(n={self.n})"


@dataclass
class TextSpace:
    """An open-ended text value (for NLP-style environments).

    Examples:
        >>> s = TextSpace(max_length=512)
        >>> s.contains("hello world")
        True
    """

    max_length: Optional[int] = None
    """Maximum character length. None = unlimited."""

    min_length: int = 0
    """Minimum character length."""

    def contains(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        if len(value) < self.min_length:
            return False
        if self.max_length is not None and len(value) > self.max_length:
            return False
        return True

    def sample(self) -> str:
        """Return a placeholder string."""
        return "<sample_text>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "min_length": self.min_length,
            "max_length": self.max_length,
        }

    def __repr__(self) -> str:
        parts = []
        if self.min_length > 0:
            parts.append(f"min={self.min_length}")
        if self.max_length is not None:
            parts.append(f"max={self.max_length}")
        inner = ", ".join(parts)
        return f"TextSpace({inner})" if inner else "TextSpace()"


@dataclass
class DictSpace:
    """A composite space made of named sub-spaces.

    Examples:
        >>> s = DictSpace(fields={
        ...     "classification": DiscreteSpace(3, ["SAFE","UNSAFE","NEEDS_REVIEW"]),
        ...     "reasoning": TextSpace(max_length=1024),
        ... })
        >>> s.contains({"classification": "SAFE", "reasoning": "looks fine"})
        True
    """

    fields: Dict[str, Any] = field(default_factory=dict)
    """Mapping of field name → sub-space."""

    def contains(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        for key, space in self.fields.items():
            if key not in value:
                return False
            if hasattr(space, "contains") and not space.contains(value[key]):
                return False
        return True

    def sample(self) -> Dict[str, Any]:
        result = {}
        for key, space in self.fields.items():
            if hasattr(space, "sample"):
                result[key] = space.sample()
            else:
                result[key] = None
        return result

    def to_dict(self) -> Dict[str, Any]:
        fields_dict = {}
        for key, space in self.fields.items():
            if hasattr(space, "to_dict"):
                fields_dict[key] = space.to_dict()
            else:
                fields_dict[key] = str(space)
        return {"type": "dict", "fields": fields_dict}

    def __repr__(self) -> str:
        field_strs = [f"{k}: {v!r}" for k, v in self.fields.items()]
        return "DictSpace({" + ", ".join(field_strs) + "})"
