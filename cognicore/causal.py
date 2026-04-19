"""
CogniCore Causal Engine — Learn cause→effect, not just correlations.

Builds causal graphs from agent experience to understand WHY actions
lead to outcomes, and enables counterfactual analysis.

Usage::

    from cognicore.causal import CausalEngine

    engine = CausalEngine()
    engine.observe(cause="phishing_keywords", action="SAFE", outcome="wrong")
    engine.observe(cause="phishing_keywords", action="UNSAFE", outcome="correct")
    graph = engine.get_causal_graph()
    counterfactual = engine.what_if("phishing_keywords", "UNSAFE")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class CausalLink:
    """A cause→effect relationship with strength."""

    def __init__(self, cause: str, effect: str):
        self.cause = cause
        self.effect = effect
        self.observations = 0
        self.positive = 0  # led to correct
        self.negative = 0  # led to wrong

    @property
    def strength(self) -> float:
        if self.observations == 0:
            return 0
        return self.positive / self.observations

    @property
    def confidence(self) -> float:
        """Confidence grows with observations."""
        return min(1.0, self.observations / 10)

    def to_dict(self) -> Dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": round(self.strength, 3),
            "confidence": round(self.confidence, 3),
            "observations": self.observations,
        }


class CausalEngine:
    """Causal reasoning engine — learns cause→effect relationships.

    Instead of just correlations ("phishing appears with UNSAFE"),
    this tracks directed causality:
      "phishing_keywords → action=UNSAFE → correct"
      "phishing_keywords → action=SAFE → wrong"
    """

    def __init__(self):
        # cause → {action → CausalLink}
        self._links: Dict[str, Dict[str, CausalLink]] = defaultdict(dict)
        self._observations: List[Dict] = []

    def observe(
        self,
        cause: str,
        action: str,
        outcome: str,
        category: str = "",
        context: Dict = None,
    ):
        """Record a causal observation.

        Parameters
        ----------
        cause : str
            The causal factor (e.g., "phishing_keywords", "authority_claim")
        action : str
            What the agent did (e.g., "SAFE", "UNSAFE")
        outcome : str
            What happened: "correct" or "wrong"
        """
        if action not in self._links[cause]:
            self._links[cause][action] = CausalLink(cause, f"{action}→{outcome}")

        link = self._links[cause][action]
        link.observations += 1
        link.effect = f"{action}→{outcome}"
        if outcome == "correct":
            link.positive += 1
        else:
            link.negative += 1

        self._observations.append({
            "cause": cause,
            "action": action,
            "outcome": outcome,
            "category": category,
        })

    def observe_step(self, obs: Dict, action: Dict, correct: bool, category: str = ""):
        """Convenience: observe from an env step."""
        # Extract cause signals from observation
        prompt = str(obs.get("prompt", "")).lower()
        causes = self._extract_causes(prompt)
        action_str = str(action.get("classification", action))
        outcome = "correct" if correct else "wrong"

        for cause in causes:
            self.observe(cause, action_str, outcome, category)

    def _extract_causes(self, text: str) -> List[str]:
        """Extract causal signals from text."""
        signals = {
            "unsafe_keywords": ["hack", "malware", "exploit", "weapon", "bomb", "attack"],
            "safe_keywords": ["cook", "recipe", "garden", "travel", "education"],
            "authority_claim": ["researcher", "expert", "authorized", "certified"],
            "emotional_appeal": ["urgent", "desperate", "emergency", "please help"],
            "negation": ["not", "don't", "never", "no way"],
            "technical_jargon": ["sql", "injection", "buffer", "overflow", "xss"],
            "ambiguity": ["might", "maybe", "could", "perhaps", "sometimes"],
        }

        found = []
        for signal, keywords in signals.items():
            if any(kw in text for kw in keywords):
                found.append(signal)

        return found if found else ["unknown_signal"]

    def what_if(self, cause: str, alternative_action: str) -> Dict[str, Any]:
        """Counterfactual analysis: what if the agent had done X instead?

        Parameters
        ----------
        cause : str
            The causal factor.
        alternative_action : str
            What would happen if the agent did this instead?
        """
        if cause not in self._links:
            return {"prediction": "unknown", "confidence": 0, "reason": "No data for this cause"}

        actions = self._links[cause]
        if alternative_action in actions:
            link = actions[alternative_action]
            return {
                "prediction": "correct" if link.strength > 0.5 else "wrong",
                "probability": link.strength,
                "confidence": link.confidence,
                "observations": link.observations,
                "reason": f"Based on {link.observations} past observations",
            }

        # No data for this specific action
        return {
            "prediction": "unknown",
            "confidence": 0,
            "reason": f"No data for action '{alternative_action}' with cause '{cause}'",
        }

    def get_causal_graph(self) -> Dict[str, List[Dict]]:
        """Get the full causal graph."""
        graph = {}
        for cause, actions in self._links.items():
            graph[cause] = [link.to_dict() for link in actions.values()]
        return graph

    def best_action(self, causes: List[str]) -> Optional[Tuple[str, float]]:
        """Given causal factors, what's the best action?"""
        action_scores: Dict[str, List[float]] = defaultdict(list)

        for cause in causes:
            if cause in self._links:
                for action, link in self._links[cause].items():
                    if link.confidence > 0.3:
                        action_scores[action].append(link.strength)

        if not action_scores:
            return None

        best = max(action_scores, key=lambda a: sum(action_scores[a]) / len(action_scores[a]))
        score = sum(action_scores[best]) / len(action_scores[best])
        return best, score

    def print_graph(self):
        """Print the causal graph."""
        print(f"\n{'=' * 60}")
        print(f"  Causal Graph ({len(self._links)} causes, {len(self._observations)} observations)")
        print(f"{'=' * 60}")
        for cause in sorted(self._links.keys()):
            print(f"\n  {cause}:")
            for action, link in self._links[cause].items():
                bar_len = int(link.strength * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(
                    f"    → {action:15s} [{bar}] {link.strength:.0%} correct "
                    f"({link.observations} obs, conf={link.confidence:.0%})"
                )
        print(f"{'=' * 60}\n")
