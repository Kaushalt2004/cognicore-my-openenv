"""
CogniCore Knowledge Transfer — Agents teach each other.

Transfer learned knowledge from an expert agent to a student agent.
Supports memory transfer, strategy transfer, and mentoring.

Usage::

    from cognicore.knowledge_transfer import transfer_knowledge

    transfer_knowledge(expert_agent, student_agent, method="full")
"""

from __future__ import annotations

from typing import Any, Dict, List


def transfer_knowledge(
    expert,
    student,
    method: str = "full",
    selective: bool = False,
    min_confidence: float = 0.7,
) -> Dict[str, Any]:
    """Transfer knowledge from expert to student agent.

    Parameters
    ----------
    expert : agent
        Source agent (must have .knowledge or similar)
    student : agent
        Target agent to receive knowledge
    method : str
        'full' — transfer everything
        'successes_only' — transfer only correct patterns
        'selective' — transfer only high-confidence knowledge
    min_confidence : float
        For 'selective' method, minimum confidence to transfer

    Returns
    -------
    dict with transfer stats
    """
    transferred = 0
    skipped = 0

    # Try knowledge transfer (for agents with .knowledge dict)
    expert_knowledge = getattr(expert, "knowledge", {})
    if not expert_knowledge:
        # Try other attribute names
        for attr in ("q", "q_table", "rules", "_knowledge", "knowledge_base"):
            expert_knowledge = getattr(expert, attr, {})
            if expert_knowledge:
                break

    if not expert_knowledge:
        return {
            "transferred": 0,
            "skipped": 0,
            "error": "Expert has no knowledge to transfer",
        }

    # Ensure student has knowledge attribute
    if not hasattr(student, "knowledge"):
        student.knowledge = {}

    for category, data in expert_knowledge.items():
        if method == "successes_only":
            # Only transfer positive outcomes
            if isinstance(data, dict):
                positive = {k: v for k, v in data.items() if v > 0}
                if positive:
                    student.knowledge[category] = positive
                    transferred += 1
                else:
                    skipped += 1
            elif isinstance(data, str):
                student.knowledge[category] = data
                transferred += 1
        elif method == "selective":
            # Only transfer high-confidence knowledge
            if isinstance(data, dict):
                max_val = max(data.values()) if data else 0
                if max_val >= min_confidence:
                    student.knowledge[category] = data
                    transferred += 1
                else:
                    skipped += 1
            else:
                student.knowledge[category] = data
                transferred += 1
        else:
            # Full transfer
            student.knowledge[category] = data
            transferred += 1

    # Transfer epsilon if both agents have it
    if hasattr(expert, "epsilon") and hasattr(student, "epsilon"):
        student.epsilon = min(student.epsilon, expert.epsilon)

    return {
        "transferred": transferred,
        "skipped": skipped,
        "total_available": len(expert_knowledge),
        "method": method,
    }


class MentorStudent:
    """Mentor-student training: expert guides student through episodes.

    The student acts first, then the mentor corrects if wrong.
    """

    def __init__(self, mentor, student):
        self.mentor = mentor
        self.student = student
        self.corrections: List[Dict] = []

    def guided_step(self, obs: Dict) -> Dict[str, Any]:
        """Student acts, mentor evaluates and potentially corrects.

        Returns the action to use (student's if correct, mentor's if overridden).
        """
        student_action = self.student.act(obs)
        mentor_action = self.mentor.act(obs)

        if student_action == mentor_action:
            return {
                "action": student_action,
                "corrected": False,
                "source": "student",
            }
        else:
            self.corrections.append(
                {
                    "student_said": student_action,
                    "mentor_said": mentor_action,
                    "category": obs.get("category", "?"),
                }
            )
            return {
                "action": mentor_action,
                "corrected": True,
                "source": "mentor",
                "student_original": student_action,
            }

    def correction_rate(self) -> float:
        """What percentage of student actions were corrected."""
        if not self.corrections:
            return 0
        total_steps = len(self.corrections)  # only tracks when different
        return len(self.corrections) / max(total_steps, 1)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_corrections": len(self.corrections),
            "categories_corrected": list(set(c["category"] for c in self.corrections)),
        }
