"""
CogniCore Rate Limiter — Protect LLM API calls with configurable limits.

Usage::

    from cognicore.rate_limiter import RateLimiter

    limiter = RateLimiter(calls_per_minute=60, calls_per_hour=1000)
    if limiter.can_call():
        limiter.record()
        result = llm.generate(...)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict
import logging

logger = logging.getLogger("cognicore.rate_limiter")


class RateLimiter:
    """Token-bucket-style rate limiter for API calls.

    Parameters
    ----------
    calls_per_minute : int
        Max calls per minute.
    calls_per_hour : int
        Max calls per hour.
    calls_per_day : int
        Max calls per day.
    """

    def __init__(
        self,
        calls_per_minute: int = 60,
        calls_per_hour: int = 1000,
        calls_per_day: int = 10000,
    ):
        self.limits = {
            "minute": calls_per_minute,
            "hour": calls_per_hour,
            "day": calls_per_day,
        }
        self._timestamps: deque = deque(maxlen=calls_per_day)
        self._total_calls = 0
        self._total_waited = 0.0
        self._rejected = 0

    def can_call(self) -> bool:
        """Check if a call is allowed right now."""
        now = time.time()
        self._clean_old(now)

        minute_count = sum(1 for t in self._timestamps if now - t < 60)
        hour_count = sum(1 for t in self._timestamps if now - t < 3600)
        day_count = len(self._timestamps)

        return (
            minute_count < self.limits["minute"]
            and hour_count < self.limits["hour"]
            and day_count < self.limits["day"]
        )

    def record(self) -> None:
        """Record a call."""
        self._timestamps.append(time.time())
        self._total_calls += 1

    def wait_if_needed(self) -> float:
        """Wait until a call is allowed.

        Returns seconds waited.
        """
        waited = 0.0
        while not self.can_call():
            time.sleep(0.1)
            waited += 0.1
            if waited > 120:  # max wait 2 minutes
                self._rejected += 1
                raise RuntimeError("Rate limit exceeded — max wait time reached")

        self._total_waited += waited
        return waited

    def call_and_wait(self) -> float:
        """Wait if needed, then record the call."""
        waited = self.wait_if_needed()
        self.record()
        return waited

    def _clean_old(self, now: float):
        """Remove timestamps older than 1 day."""
        while self._timestamps and now - self._timestamps[0] > 86400:
            self._timestamps.popleft()

    def usage(self) -> Dict[str, Any]:
        """Current usage stats."""
        now = time.time()
        self._clean_old(now)

        minute_count = sum(1 for t in self._timestamps if now - t < 60)
        hour_count = sum(1 for t in self._timestamps if now - t < 3600)
        day_count = len(self._timestamps)

        return {
            "minute": {
                "used": minute_count,
                "limit": self.limits["minute"],
                "remaining": self.limits["minute"] - minute_count,
            },
            "hour": {
                "used": hour_count,
                "limit": self.limits["hour"],
                "remaining": self.limits["hour"] - hour_count,
            },
            "day": {
                "used": day_count,
                "limit": self.limits["day"],
                "remaining": self.limits["day"] - day_count,
            },
            "total_calls": self._total_calls,
            "total_waited_seconds": round(self._total_waited, 2),
            "rejected": self._rejected,
        }

    def print_usage(self):
        """Print usage stats."""
        u = self.usage()
        logger.info("\n  Rate Limiter Status:")
        for period in ("minute", "hour", "day"):
            data = u[period]
            pct = data["used"] / data["limit"] if data["limit"] else 0
            bar_len = int(pct * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(
                f"    {period:8s} [{bar}] {data['used']}/{data['limit']} ({data['remaining']} remaining)"
            )
        logger.info(f"    Total calls: {u['total_calls']}")
        if u["total_waited_seconds"] > 0:
            logger.info(f"    Time waited: {u['total_waited_seconds']}s")
