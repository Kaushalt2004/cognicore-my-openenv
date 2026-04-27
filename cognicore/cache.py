"""
CogniCore Caching Layer — Cache identical prompts to save tokens.

Caches LLM responses so identical inputs don't burn extra tokens.

Usage::

    from cognicore.cache import ResponseCache

    cache = ResponseCache(max_size=1000)
    result = cache.get("classify: hello world")
    if result is None:
        result = llm.generate("classify: hello world")
        cache.put("classify: hello world", result)
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, Optional


class ResponseCache:
    """LRU cache for LLM responses.

    Saves tokens by returning cached responses for identical inputs.

    Parameters
    ----------
    max_size : int
        Maximum number of cached entries.
    ttl : float
        Time-to-live in seconds (0 = no expiry).
    """

    def __init__(self, max_size: int = 1000, ttl: float = 0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    def _key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.md5(prompt.encode("utf-8"), usedforsecurity=False).hexdigest()

    def get(self, prompt: str) -> Optional[Any]:
        """Get cached response for a prompt."""
        key = self._key(prompt)

        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if self.ttl > 0 and time.time() - entry["time"] > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recent)
            self._cache.move_to_end(key)
            self._hits += 1
            self._tokens_saved += entry.get("tokens", 0)
            return entry["response"]

        self._misses += 1
        return None

    def put(self, prompt: str, response: Any, tokens_used: int = 0) -> None:
        """Cache a response."""
        key = self._key(prompt)

        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest

        self._cache[key] = {
            "response": response,
            "time": time.time(),
            "tokens": tokens_used,
            "prompt_preview": prompt[:100],
        }

    def invalidate(self, prompt: str) -> bool:
        """Remove a specific cache entry."""
        key = self._key(prompt)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """Clear all cached entries."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "tokens_saved": self._tokens_saved,
            "total_requests": total,
        }

    def print_stats(self):
        """Print cache statistics."""
        s = self.stats()
        print("\n  Response Cache:")
        print(f"    Size: {s['size']}/{s['max_size']}")
        print(f"    Hits: {s['hits']} | Misses: {s['misses']}")
        print(f"    Hit rate: {s['hit_rate']:.0%}")
        print(f"    Tokens saved: {s['tokens_saved']}")
