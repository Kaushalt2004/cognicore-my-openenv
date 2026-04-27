"""
CogniCore Cost Tracker — Token usage estimation and cost monitoring.

Tracks API costs, latency, and token usage across episodes.

Usage::

    from cognicore.cost_tracker import CostTracker

    tracker = CostTracker(cost_per_1k_input=0.15, cost_per_1k_output=0.60)
    tracker.record_call(input_tokens=500, output_tokens=100, latency_ms=230)
    tracker.print_summary()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class CostTracker:
    """Track LLM API costs, token usage, and latency.

    Parameters
    ----------
    cost_per_1k_input : float
        Cost per 1000 input tokens (USD). Default: Gemini Flash pricing.
    cost_per_1k_output : float
        Cost per 1000 output tokens (USD).
    budget_limit : float or None
        Maximum budget (USD). Raises warning when exceeded.
    """

    # Default pricing (Gemini 1.5 Flash)
    PRICING = {
        "gemini-flash": {"input": 0.075, "output": 0.30},
        "gemini-pro": {"input": 1.25, "output": 5.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-sonnet": {"input": 3.00, "output": 15.00},
    }

    def __init__(
        self,
        cost_per_1k_input: float = 0.075,
        cost_per_1k_output: float = 0.30,
        budget_limit: Optional[float] = None,
        model_name: str = "gemini-flash",
    ):
        if model_name in self.PRICING:
            cost_per_1k_input = self.PRICING[model_name]["input"]
            cost_per_1k_output = self.PRICING[model_name]["output"]

        self.cost_per_input = cost_per_1k_input / 1000
        self.cost_per_output = cost_per_1k_output / 1000
        self.budget_limit = budget_limit
        self.model_name = model_name

        self.calls: List[Dict[str, Any]] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._total_latency_ms = 0
        self._budget_warnings = 0

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0,
        episode: int = 0,
        step: int = 0,
    ) -> Dict[str, Any]:
        """Record an API call.

        Returns cost and budget info for this call.
        """
        cost = (input_tokens * self.cost_per_input +
                output_tokens * self.cost_per_output)

        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost
        self._total_latency_ms += latency_ms

        self.calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
            "episode": episode,
            "step": step,
            "timestamp": time.time(),
        })

        result = {
            "call_cost": cost,
            "total_cost": self._total_cost,
            "budget_remaining": (self.budget_limit - self._total_cost) if self.budget_limit else None,
            "over_budget": self.budget_limit and self._total_cost > self.budget_limit,
        }

        if result["over_budget"]:
            self._budget_warnings += 1

        return result

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token for English)."""
        return max(1, len(text) // 4)

    def record_text(
        self,
        input_text: str,
        output_text: str,
        latency_ms: float = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Record a call using text (auto-estimates tokens)."""
        return self.record_call(
            input_tokens=self.estimate_tokens(input_text),
            output_tokens=self.estimate_tokens(output_text),
            latency_ms=latency_ms,
            **kwargs,
        )

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def avg_latency_ms(self) -> float:
        return self._total_latency_ms / len(self.calls) if self.calls else 0

    def summary(self) -> Dict[str, Any]:
        """Return cost tracking summary."""
        return {
            "model": self.model_name,
            "total_calls": len(self.calls),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "avg_cost_per_call": round(self._total_cost / len(self.calls), 6) if self.calls else 0,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "budget_limit": self.budget_limit,
            "budget_remaining": round(self.budget_limit - self._total_cost, 6) if self.budget_limit else None,
            "budget_warnings": self._budget_warnings,
        }

    def cost_by_episode(self) -> Dict[int, float]:
        """Group costs by episode."""
        episodes = {}
        for c in self.calls:
            ep = c.get("episode", 0)
            episodes[ep] = episodes.get(ep, 0) + c["cost"]
        return episodes

    def compare_models(self, text_sample: str = "A" * 2000) -> Dict[str, float]:
        """Estimate cost of current usage across different models.

        Returns estimated cost per model for the same token usage.
        """
        total_in = self._total_input_tokens
        total_out = self._total_output_tokens

        comparison = {}
        for model, pricing in self.PRICING.items():
            cost = (total_in * pricing["input"] / 1000 +
                    total_out * pricing["output"] / 1000)
            comparison[model] = round(cost, 6)

        return comparison

    def print_summary(self):
        """Print formatted cost summary."""
        s = self.summary()
        print(f"\n{'=' * 55}")
        print(f"  Cost Tracker — {s['model']}")
        print(f"{'=' * 55}")
        print(f"  Total calls:     {s['total_calls']}")
        print(f"  Input tokens:    {s['total_input_tokens']:,}")
        print(f"  Output tokens:   {s['total_output_tokens']:,}")
        print(f"  Total tokens:    {s['total_tokens']:,}")
        print(f"  Total cost:      ${s['total_cost_usd']:.4f}")
        print(f"  Avg per call:    ${s['avg_cost_per_call']:.6f}")
        print(f"  Avg latency:     {s['avg_latency_ms']:.0f}ms")

        if s['budget_limit']:
            remaining = s['budget_remaining']
            print(f"  Budget:          ${s['budget_limit']:.2f}")
            print(f"  Remaining:       ${remaining:.4f}")

        # Model comparison
        comp = self.compare_models()
        if comp and self.total_tokens > 0:
            print("\n  Same usage across models:")
            for model, cost in sorted(comp.items(), key=lambda x: x[1]):
                marker = " <-- current" if model == self.model_name else ""
                print(f"    {model:20s} ${cost:.4f}{marker}")

        print(f"{'=' * 55}\n")
