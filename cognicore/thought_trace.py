"""
CogniCore Thought Tracer — Track internal reasoning chains.

Traces decision-making paths, showing which inputs led to which
decisions and where reasoning went wrong.

Usage::

    from cognicore.thought_trace import ThoughtTracer

    tracer = ThoughtTracer()
    tracer.begin_thought("Classifying email about password reset")
    tracer.add_evidence("contains 'password'", weight=0.6, direction="UNSAFE")
    tracer.add_evidence("educational context", weight=0.3, direction="SAFE")
    tracer.conclude("UNSAFE", confidence=0.7)
    tracer.was_correct(False, truth="SAFE")
    tracer.print_chain()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ThoughtNode:
    """Single node in a reasoning chain."""

    def __init__(
        self,
        evidence: str,
        weight: float,
        direction: str,
        step: int = 0,
    ):
        self.evidence = evidence
        self.weight = weight
        self.direction = direction
        self.step = step


class ThoughtChain:
    """Complete reasoning chain for one decision."""

    def __init__(self, context: str, step: int = 0):
        self.context = context
        self.step = step
        self.nodes: List[ThoughtNode] = []
        self.conclusion: Optional[str] = None
        self.confidence: float = 0
        self.correct: Optional[bool] = None
        self.truth: Optional[str] = None

    def add_evidence(self, evidence: str, weight: float, direction: str):
        self.nodes.append(ThoughtNode(evidence, weight, direction, self.step))

    def conclude(self, decision: str, confidence: float = 0.5):
        self.conclusion = decision
        self.confidence = confidence

    def was_correct(self, correct: bool, truth: str = ""):
        self.correct = correct
        self.truth = truth

    def dominant_direction(self) -> str:
        """What direction did evidence mostly point?"""
        if not self.nodes:
            return "unknown"
        scores = {}
        for n in self.nodes:
            scores[n.direction] = scores.get(n.direction, 0) + n.weight
        return max(scores, key=scores.get) if scores else "unknown"

    def reasoning_error(self) -> Optional[str]:
        """Identify where reasoning went wrong (if it did)."""
        if self.correct is None or self.correct:
            return None

        dominant = self.dominant_direction()
        if dominant == self.conclusion:
            return f"Evidence correctly pointed to '{dominant}' but that was wrong. The evidence itself was misleading."
        return (
            f"Evidence pointed to '{dominant}' but agent concluded '{self.conclusion}'. "
            f"The agent ignored the evidence."
        )


class ThoughtTracer:
    """Traces reasoning chains across decisions.

    Records evidence, weights, and conclusions for each decision,
    then analyzes where reasoning succeeded or failed.
    """

    def __init__(self):
        self.chains: List[ThoughtChain] = []
        self._current: Optional[ThoughtChain] = None
        self._step = 0

    def begin_thought(self, context: str) -> ThoughtChain:
        """Start a new reasoning chain."""
        self._step += 1
        chain = ThoughtChain(context, self._step)
        self._current = chain
        self.chains.append(chain)
        return chain

    def add_evidence(self, evidence: str, weight: float = 0.5, direction: str = ""):
        """Add evidence to the current reasoning chain."""
        if self._current:
            self._current.add_evidence(evidence, weight, direction)

    def conclude(self, decision: str, confidence: float = 0.5):
        """Record the final decision."""
        if self._current:
            self._current.conclude(decision, confidence)

    def was_correct(self, correct: bool, truth: str = ""):
        """Record whether the decision was correct."""
        if self._current:
            self._current.was_correct(correct, truth)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def reasoning_errors(self) -> List[Dict[str, Any]]:
        """Find all chains where reasoning went wrong."""
        errors = []
        for chain in self.chains:
            error = chain.reasoning_error()
            if error:
                errors.append({
                    "step": chain.step,
                    "context": chain.context[:60],
                    "concluded": chain.conclusion,
                    "truth": chain.truth,
                    "error": error,
                    "evidence_count": len(chain.nodes),
                })
        return errors

    def evidence_accuracy(self) -> Dict[str, Dict[str, float]]:
        """How accurate was each type of evidence?"""
        evidence_stats = {}
        for chain in self.chains:
            for node in chain.nodes:
                key = node.evidence[:40]
                if key not in evidence_stats:
                    evidence_stats[key] = {"helpful": 0, "misleading": 0, "total": 0}
                evidence_stats[key]["total"] += 1

                if chain.correct is not None:
                    if (node.direction == chain.truth) if chain.truth else chain.correct:
                        evidence_stats[key]["helpful"] += 1
                    else:
                        evidence_stats[key]["misleading"] += 1

        return evidence_stats

    def confidence_calibration(self) -> Dict[str, float]:
        """How well-calibrated are confidence scores?"""
        decided = [c for c in self.chains if c.correct is not None]
        if not decided:
            return {"avg_confidence": 0, "accuracy": 0, "calibration_error": 0}

        avg_conf = sum(c.confidence for c in decided) / len(decided)
        accuracy = sum(1 for c in decided if c.correct) / len(decided)
        cal_error = abs(avg_conf - accuracy)

        return {
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "calibration_error": cal_error,
            "overconfident": avg_conf > accuracy + 0.1,
            "underconfident": avg_conf < accuracy - 0.1,
        }

    def print_chain(self, chain_idx: int = -1):
        """Print a specific reasoning chain."""
        if not self.chains:
            print("  No reasoning chains recorded.")
            return

        chain = self.chains[chain_idx]
        print(f"\n  Thought Chain #{chain.step}:")
        print(f"  Context: {chain.context}")

        for i, node in enumerate(chain.nodes):
            arrow = "→" if node.direction else "?"
            print(f"    {i+1}. {node.evidence} (weight={node.weight:.1f}) {arrow} {node.direction}")

        if chain.conclusion:
            icon = "✓" if chain.correct else "✗" if chain.correct is False else "?"
            print(f"  Conclusion: {chain.conclusion} (confidence={chain.confidence:.0%}) [{icon}]")

        if chain.truth and not chain.correct:
            error = chain.reasoning_error()
            print(f"  Truth: {chain.truth}")
            if error:
                print(f"  Error: {error}")

    def print_analysis(self):
        """Print full reasoning analysis."""
        print(f"\n{'=' * 60}")
        print(f"  Thought Trace Analysis")
        print(f"  {len(self.chains)} decisions traced")
        print(f"{'=' * 60}")

        # Calibration
        cal = self.confidence_calibration()
        print(f"\n  Confidence Calibration:")
        print(f"    Avg confidence: {cal['avg_confidence']:.0%}")
        print(f"    Actual accuracy: {cal['accuracy']:.0%}")
        print(f"    Calibration error: {cal['calibration_error']:.0%}")
        if cal.get("overconfident"):
            print(f"    WARNING: Agent is overconfident")

        # Errors
        errors = self.reasoning_errors()
        if errors:
            print(f"\n  Reasoning Errors ({len(errors)}):")
            for e in errors[:5]:
                print(f"    Step {e['step']}: {e['error']}")

        print(f"{'=' * 60}\n")
