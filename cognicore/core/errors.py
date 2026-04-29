"""
CogniCore Exceptions — Custom error hierarchy for clear, actionable error messages.

All CogniCore-specific exceptions inherit from ``CogniCoreError``.
This makes it trivial to catch framework errors separately from
Python built-in errors.

Usage::

    from cognicore.core.errors import CogniCoreError, InvalidEnvironmentError

    try:
        env = cognicore.make("NonExistent-v1")
    except InvalidEnvironmentError as e:
        print(e)  # "Environment 'NonExistent-v1' not found. Available: ..."
"""

from __future__ import annotations


class CogniCoreError(Exception):
    """Base exception for all CogniCore errors."""
    pass


class InvalidEnvironmentError(CogniCoreError):
    """Raised when an environment ID is not found in the registry."""

    def __init__(self, env_id: str, available: list[str] | None = None) -> None:
        msg = f"Environment '{env_id}' not found."
        if available:
            suggestions = [e for e in available if env_id.lower() in e.lower()]
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions[:3])}?"
            else:
                msg += f" Available: {', '.join(available[:10])}"
                if available and len(available) > 10:
                    msg += f" ... ({len(available)} total)"
        super().__init__(msg)
        self.env_id = env_id


class InvalidActionError(CogniCoreError):
    """Raised when an agent passes an invalid action to step()."""

    def __init__(self, action: object, expected: str = "") -> None:
        msg = f"Invalid action: {action!r}."
        if expected:
            msg += f" Expected: {expected}"
        super().__init__(msg)
        self.action = action


class InvalidConfigError(CogniCoreError):
    """Raised when a CogniCoreConfig has invalid field values."""

    def __init__(self, field: str, value: object, reason: str) -> None:
        super().__init__(f"Invalid config: {field}={value!r}. {reason}")
        self.field = field
        self.value = value


class EnvironmentNotResetError(CogniCoreError):
    """Raised when step() is called before reset()."""

    def __init__(self) -> None:
        super().__init__(
            "Cannot call step() before reset(). Call env.reset() first."
        )


class EpisodeFinishedError(CogniCoreError):
    """Raised when step() is called after the episode has ended."""

    def __init__(self) -> None:
        super().__init__(
            "Episode is finished. Call env.reset() to start a new episode."
        )


class AgentInterfaceError(CogniCoreError):
    """Raised when an agent doesn't implement the required interface."""

    def __init__(self, agent: object, missing_method: str) -> None:
        cls_name = type(agent).__name__
        super().__init__(
            f"Agent '{cls_name}' is missing required method '{missing_method}()'. "
            f"Ensure your agent implements the BaseAgent interface."
        )
        self.agent = agent
        self.missing_method = missing_method
