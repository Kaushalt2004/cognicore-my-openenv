"""
CogniCore Logging — Structured logging utilities.

Provides a consistent logging format for environments, middleware,
and agents with colorized output and structured fields.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a CogniCore logger with structured formatting.

    Parameters
    ----------
    name : str
        Logger name (e.g. ``"cognicore.env"``).
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s │ %(name)-24s │ %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def log_step(
    logger: logging.Logger,
    step: int,
    action: str,
    reward: float,
    correct: bool,
    done: bool,
    **extra: Any,
) -> None:
    """Log a single environment step in a structured format."""
    status = "✅" if correct else "❌"
    logger.info(
        f"[STEP] step={step} action={action} reward={reward:.2f} {status} done={done}"
    )


def log_episode_end(
    logger: logging.Logger,
    episode: int,
    steps: int,
    score: float,
    accuracy: float,
    **extra: Any,
) -> None:
    """Log episode completion summary."""
    logger.info(
        f"[END] episode={episode} steps={steps} "
        f"score={score:.4f} accuracy={accuracy:.1%}"
    )
