"""
CogniCore Strategy Switcher — Real-time strategy switching mid-task.

Agents change behavior mode (safe/explore/aggressive) based on
performance, risk, and environment signals.

Usage::

    from cognicore.strategy import StrategySwitcher

    switcher = StrategySwitcher()
    switcher.add_mode("safe", epsilon=0.05, threshold=0.8)
    switcher.add_mode("explore", epsilon=0.4, threshold=0.3)
    mode = switcher.decide(accuracy=0.3, streak=-3)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class StrategyMode:
    """A named strategy configuration."""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    def to_dict(self) -> Dict:
        return {"name": self.name, **self.params}


class StrategySwitcher:
    """Real-time strategy switching based on performance signals.

    Monitors agent performance and automatically switches between
    predefined strategy modes.
    """

    def __init__(self):
        self.modes: Dict[str, StrategyMode] = {}
        self.current_mode: Optional[str] = None
        self.switch_history: List[Dict] = []
        self._rules: List[Dict] = []

        # Add default modes
        self.add_mode(
            "safe", epsilon=0.05, threshold=0.8, description="Conservative, low-risk"
        )
        self.add_mode(
            "explore", epsilon=0.4, threshold=0.3, description="High exploration"
        )
        self.add_mode(
            "balanced", epsilon=0.15, threshold=0.5, description="Default balanced"
        )
        self.add_mode(
            "aggressive",
            epsilon=0.3,
            threshold=0.2,
            description="High-risk, high-reward",
        )

        self.current_mode = "balanced"

    def add_mode(self, name: str, **params) -> "StrategySwitcher":
        """Add a strategy mode."""
        self.modes[name] = StrategyMode(name, **params)
        return self

    def add_rule(
        self,
        from_mode: str,
        to_mode: str,
        condition: str,
        threshold: float = 0,
    ) -> "StrategySwitcher":
        """Add a switching rule.

        Built-in conditions:
        - 'accuracy_below': switch when accuracy < threshold
        - 'accuracy_above': switch when accuracy > threshold
        - 'streak_below': switch when streak < threshold
        - 'risk_above': switch when risk > threshold
        """
        self._rules.append(
            {
                "from": from_mode,
                "to": to_mode,
                "condition": condition,
                "threshold": threshold,
            }
        )
        return self

    def decide(
        self,
        accuracy: float = 0.5,
        streak: int = 0,
        risk: float = 0.0,
        step: int = 0,
        **signals,
    ) -> str:
        """Decide which mode to use based on current signals.

        Returns the mode name and applies it.
        """
        # Check custom rules first
        for rule in self._rules:
            if rule["from"] != self.current_mode and rule["from"] != "*":
                continue

            triggered = False
            if rule["condition"] == "accuracy_below" and accuracy < rule["threshold"]:
                triggered = True
            elif rule["condition"] == "accuracy_above" and accuracy > rule["threshold"]:
                triggered = True
            elif rule["condition"] == "streak_below" and streak < rule["threshold"]:
                triggered = True
            elif rule["condition"] == "risk_above" and risk > rule["threshold"]:
                triggered = True

            if triggered:
                return self._switch(rule["to"], step, reason=rule["condition"])

        # Default auto-switching logic
        if accuracy < 0.3 and self.current_mode != "explore":
            return self._switch("explore", step, reason="very low accuracy → exploring")

        if accuracy > 0.8 and self.current_mode != "safe":
            return self._switch("safe", step, reason="high accuracy → playing safe")

        if streak <= -3 and self.current_mode != "explore":
            return self._switch(
                "explore", step, reason="failure streak → trying new things"
            )

        if risk > 0.7 and self.current_mode != "safe":
            return self._switch(
                "safe", step, reason="high risk detected → switching to safe"
            )

        return self.current_mode

    def _switch(self, new_mode: str, step: int = 0, reason: str = "") -> str:
        """Switch to a new mode."""
        if new_mode not in self.modes:
            return self.current_mode

        old = self.current_mode
        self.current_mode = new_mode
        self.switch_history.append(
            {
                "from": old,
                "to": new_mode,
                "step": step,
                "reason": reason,
            }
        )
        return new_mode

    def get_params(self, mode: str = None) -> Dict:
        """Get parameters for a mode."""
        mode = mode or self.current_mode
        if mode in self.modes:
            return self.modes[mode].params
        return {}

    def apply_to_agent(self, agent, mode: str = None):
        """Apply current mode's parameters to an agent."""
        params = self.get_params(mode)
        for key, value in params.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

    def stats(self) -> Dict[str, Any]:
        return {
            "current_mode": self.current_mode,
            "total_switches": len(self.switch_history),
            "available_modes": list(self.modes.keys()),
            "recent_switches": self.switch_history[-5:],
        }

    def print_status(self):
        print(f"\n  Strategy: {self.current_mode}")
        print(f"  Switches: {len(self.switch_history)}")
        if self.switch_history:
            last = self.switch_history[-1]
            print(f"  Last switch: {last['from']} → {last['to']} ({last['reason']})")
        print(f"  Modes: {', '.join(self.modes.keys())}")
