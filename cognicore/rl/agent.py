"""
CogniCore Agent — Wraps the LLM to act as a safety classifier agent.

In the Colab notebook, the Agent wraps a PPO model for FrozenLake.
Here, it wraps the LLM client to classify AI responses for safety.
"""

from typing import Tuple, Optional
from cognicore.llm.gemini import classify_safety
from cognicore.memory.vector_memory import VectorMemory
from cognicore.reflection.reflection import Reflection


class Agent:
    """AI Safety classification agent.

    Combines LLM inference with CogniCore's memory and reflection
    layers to classify AI responses.
    """

    def __init__(self, memory: VectorMemory, reflection: Reflection):
        self.memory = memory
        self.reflection = reflection
        self._total_actions = 0

    def act(
        self,
        prompt: str,
        response: str,
        category: str,
        use_memory: bool = True,
        use_reflection: bool = True,
    ) -> Tuple[str, dict]:
        """Classify an AI response using the full CogniCore pipeline.

        Pipeline:
          1. Retrieve memory context from similar past cases
          2. Get reflection hint if available
          3. Ask LLM to classify with augmented context
          4. Apply reflection override if appropriate

        Args:
            prompt: The original user prompt.
            response: The AI response to classify.
            category: Category of the safety case.
            use_memory: Whether to use memory context.
            use_reflection: Whether to use reflection hints.

        Returns:
            Tuple of (classification, metadata_dict).
        """
        self._total_actions += 1
        metadata = {"source": "model_action", "memory_context": "", "reflection_hint": ""}

        # Step 1: Memory context
        memory_context = ""
        if use_memory:
            context_entries = self.memory.get_context_for_observation(category)
            if context_entries:
                memory_context = str(context_entries)
                metadata["memory_context"] = memory_context

        # Step 2: Reflection hint
        reflection_hint = ""
        if use_reflection:
            hint = self.reflection.get_reflection_hint(category)
            if hint:
                reflection_hint = hint
                metadata["reflection_hint"] = reflection_hint

        # Step 3: LLM classification
        classification = classify_safety(
            prompt=prompt,
            response=response,
            memory_context=memory_context,
            reflection_hint=reflection_hint,
        )

        # Step 4: Reflection override
        if use_reflection:
            classification, source = self.reflection.suggest_action(
                category, classification
            )
            metadata["source"] = source

        return classification, metadata

    def stats(self) -> dict:
        """Return agent statistics."""
        return {
            "total_actions": self._total_actions,
            "memory_stats": self.memory.stats(),
            "reflection_stats": self.reflection.stats(),
        }
