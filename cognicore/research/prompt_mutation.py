"""
Sophisticated Prompt Mutation Engine — dynamically rewrites LLM prompts
based on failure history, reflection analysis, and strategy evolution.
"""
from typing import List, Dict, Optional, Tuple


class PromptMutationEngine:
    """Generates dynamically mutated prompts based on CogniCore cognition state.

    Mutations are NOT simple string appends. The engine:
    1. Analyzes failure patterns to identify root causes
    2. Generates specific anti-patterns (things to avoid)
    3. Suggests structural alternatives
    4. Adjusts reasoning constraints
    5. Modifies exploration strategy
    """

    # Failure pattern → mutation rules
    FAILURE_PATTERNS = {
        "NoneType": {
            "anti_patterns": [
                "Do NOT use simple .get() chains on potentially None objects",
                "Do NOT assume nested objects exist",
            ],
            "guidance": [
                "Use 'or {}' after .get() to handle None values: data.get('key') or {}",
                "Consider restructuring to extract nested objects first, then access fields",
                "Add explicit None checks before accessing nested attributes",
            ],
            "strategy": "restructure",
        },
        "IndexError": {
            "anti_patterns": [
                "Do NOT use hardcoded index offsets without boundary checks",
                "Do NOT assume array length matches expected size",
            ],
            "guidance": [
                "Use min/max to clamp indices within bounds",
                "Consider using itertools or slice notation for safe access",
                "Validate array length before indexed access",
            ],
            "strategy": "boundary_safe",
        },
        "KeyError": {
            "anti_patterns": [
                "Do NOT use direct dict[key] access without checking existence",
            ],
            "guidance": [
                "Use dict.get(key, default) for safe access",
                "Use 'if key in dict' guards before access",
                "Consider using collections.defaultdict",
            ],
            "strategy": "defensive",
        },
        "RecursionError": {
            "anti_patterns": [
                "Do NOT use recursive approaches for this problem",
                "Recursive solutions are banned for this task",
            ],
            "guidance": [
                "Convert to iterative approach using a stack or queue",
                "Use memoization if recursion is necessary",
                "Consider dynamic programming bottom-up approach",
            ],
            "strategy": "iterative",
        },
        "Too slow": {
            "anti_patterns": [
                "Do NOT use naive recursive implementation",
                "Exponential time complexity is unacceptable",
            ],
            "guidance": [
                "Use memoization with functools.lru_cache or manual cache dict",
                "Convert to iterative approach for O(n) time",
                "Consider dynamic programming",
            ],
            "strategy": "optimized",
        },
        "AssertionError": {
            "anti_patterns": [
                "The previous fix produced wrong output values",
                "Do NOT repeat the same logical approach",
            ],
            "guidance": [
                "Carefully trace through the algorithm step by step",
                "Verify edge cases: empty input, single element, boundary values",
                "Consider rewriting the function from scratch",
            ],
            "strategy": "rewrite",
        },
        "Race": {
            "anti_patterns": [
                "Do NOT use non-atomic read-modify-write patterns",
            ],
            "guidance": [
                "Use threading.Lock() for all shared state mutations",
                "Use atomic operations where possible",
                "Consider using queue.Queue for thread-safe communication",
            ],
            "strategy": "thread_safe",
        },
    }

    def __init__(self):
        self.mutation_history = []

    def analyze_failures(self, errors: List[str]) -> Dict:
        """Analyze failure patterns to determine mutation strategy."""
        analysis = {"patterns": [], "anti_patterns": [], "guidance": [],
                     "strategy": None, "severity": "low"}

        for error in errors:
            for pattern, rules in self.FAILURE_PATTERNS.items():
                if pattern.lower() in error.lower():
                    analysis["patterns"].append(pattern)
                    analysis["anti_patterns"].extend(rules["anti_patterns"])
                    analysis["guidance"].extend(rules["guidance"])
                    if analysis["strategy"] is None:
                        analysis["strategy"] = rules["strategy"]

        # Dedup
        analysis["anti_patterns"] = list(set(analysis["anti_patterns"]))
        analysis["guidance"] = list(set(analysis["guidance"]))

        # Severity based on repeat count
        if len(errors) >= 3:
            analysis["severity"] = "critical"
        elif len(errors) >= 2:
            analysis["severity"] = "high"

        return analysis

    def mutate_prompt(self, base_prompt: str, failed_patches: List[Tuple[str, str]],
                      context: Dict, attempt: int) -> Tuple[str, Dict]:
        """Generate a mutated prompt based on failure analysis.

        Returns: (mutated_prompt, mutation_metadata)
        """
        errors = [err for _, err in failed_patches if err]
        analysis = self.analyze_failures(errors)
        mutations_applied = []

        parts = [base_prompt]

        # 1. Anti-patterns section
        if analysis["anti_patterns"]:
            parts.append("\n\n⚠️ CRITICAL CONSTRAINTS (violations will fail):")
            for ap in analysis["anti_patterns"][:4]:
                parts.append(f"  - {ap}")
            mutations_applied.append("anti_patterns")

        # 2. Failed approaches section
        if failed_patches:
            parts.append(f"\n\n❌ FAILED APPROACHES ({len(failed_patches)} previous failures):")
            for i, (patch, err) in enumerate(failed_patches[-2:], 1):
                parts.append(f"\n  Failed Fix {i}:")
                parts.append(f"  {patch[:200]}")
                parts.append(f"  Error: {err[:100]}")
            parts.append("\n  Generate a FUNDAMENTALLY DIFFERENT fix.")
            mutations_applied.append("failed_history")

        # 3. Guided strategy section
        if analysis["guidance"]:
            parts.append("\n\n✅ RECOMMENDED APPROACH:")
            for g in analysis["guidance"][:3]:
                parts.append(f"  - {g}")
            mutations_applied.append("strategy_guidance")

        # 4. Reflection context
        if context.get("reflection_hint"):
            parts.append(f"\n\n🔍 RUNTIME REFLECTION:")
            parts.append(f"  {context['reflection_hint'][:200]}")
            mutations_applied.append("reflection")

        # 5. Cross-task memory
        if context.get("failures_to_avoid"):
            parts.append(f"\n\n🧠 MEMORY (from similar tasks):")
            parts.append(f"  Known failed approaches: {', '.join(context['failures_to_avoid'][:3])}")
            mutations_applied.append("cross_task_memory")

        if context.get("successful_patterns"):
            parts.append(f"  Successful approaches: {', '.join(context['successful_patterns'][:2])}")
            mutations_applied.append("successful_patterns")

        # 6. Escalation
        if attempt > 2 and analysis["severity"] in ("high", "critical"):
            parts.append("\n\n🔴 ESCALATION: Previous approaches exhausted.")
            parts.append("  REWRITE the entire function from scratch.")
            parts.append("  Use a completely different algorithm or data structure.")
            mutations_applied.append("escalation")

        mutated = "\n".join(parts) + "\n\nReturn ONLY the fixed Python code:"

        metadata = {
            "mutations": mutations_applied,
            "severity": analysis["severity"],
            "strategy": analysis["strategy"],
            "patterns_detected": analysis["patterns"],
            "attempt": attempt,
        }
        self.mutation_history.append(metadata)

        return mutated, metadata
