"""
CogniCore Runtime — Universal cognition layer for any AI agent.

This is the CORE module. It wraps any callable agent function and adds:
  - Persistent episodic memory
  - Reflection-based failure avoidance
  - Runtime adaptation scoring
  - Repeated failure reduction

Works with: OpenAI, LangChain, CrewAI, AutoGen, SB3, custom agents.

Usage:
    from cognicore.runtime import CogniCoreRuntime

    runtime = CogniCoreRuntime()

    # Wrap any function
    @runtime.wrap
    def my_agent(task, context):
        return do_something(task)

    # Or use directly
    result = runtime.execute(
        agent_fn=my_agent,
        task="Fix the login bug",
        category="code_fix",
    )
"""

from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from cognicore.middleware.memory import Memory
from cognicore.middleware.reflection import ReflectionEngine

logger = logging.getLogger("cognicore.runtime")


# ---------------------------------------------------------------------------
# Runtime Config
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """Configuration for CogniCore Runtime."""
    enable_memory: bool = True
    enable_reflection: bool = True
    memory_max_size: int = 10_000
    memory_similarity_key: str = "category"
    memory_top_k: int = 5
    reflection_min_samples: int = 2
    reflection_failure_threshold: int = 2
    persistence_path: Optional[str] = None
    auto_save: bool = True
    verbose: bool = False


# ---------------------------------------------------------------------------
# Execution Result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of a runtime-wrapped execution."""
    output: Any = None
    success: bool = False
    category: str = ""
    duration_ms: float = 0.0
    memory_context: List[Dict] = field(default_factory=list)
    reflection_hint: Optional[str] = None
    reflection_override: bool = False
    original_output: Any = None
    attempt: int = 1
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output"] = str(self.output)[:200]
        d["original_output"] = str(self.original_output)[:200] if self.original_output else None
        return d


# ---------------------------------------------------------------------------
# Runtime Stats
# ---------------------------------------------------------------------------

@dataclass
class RuntimeStats:
    """Aggregate statistics for the runtime."""
    total_executions: int = 0
    successes: int = 0
    failures: int = 0
    repeated_failures_avoided: int = 0
    memory_retrievals: int = 0
    reflection_overrides: int = 0
    total_duration_ms: float = 0.0
    categories_seen: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.total_executions)

    @property
    def failure_reduction_rate(self) -> float:
        return self.repeated_failures_avoided / max(1, self.failures + self.repeated_failures_avoided)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = f"{self.success_rate:.1%}"
        d["failure_reduction_rate"] = f"{self.failure_reduction_rate:.1%}"
        d["avg_duration_ms"] = f"{self.total_duration_ms / max(1, self.total_executions):.1f}"
        return d


# ---------------------------------------------------------------------------
# CogniCore Runtime — The universal wrapper
# ---------------------------------------------------------------------------

class CogniCoreRuntime:
    """Universal cognition layer that wraps any AI agent.

    This is NOT tied to CogniCoreEnv. It works with any Python callable:
    OpenAI API calls, LangChain chains, CrewAI crews, SB3 policies, etc.

    Parameters
    ----------
    config : RuntimeConfig or None
        Runtime configuration. Uses defaults if None.
    name : str
        Name for this runtime instance (used in persistence).

    Example
    -------
    >>> runtime = CogniCoreRuntime()
    >>> result = runtime.execute(
    ...     agent_fn=lambda task, ctx: "fixed",
    ...     task="Fix bug #123",
    ...     category="code_fix",
    ...     evaluator=lambda out, task: out == "fixed",
    ... )
    >>> result.success
    True
    """

    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        name: str = "cognicore-runtime",
    ) -> None:
        self.config = config or RuntimeConfig()
        self.name = name

        # Initialize middleware
        self.memory = Memory(
            max_size=self.config.memory_max_size,
            similarity_key=self.config.memory_similarity_key,
        )
        self.reflection = ReflectionEngine(
            memory=self.memory,
            min_samples=self.config.reflection_min_samples,
            failure_threshold=self.config.reflection_failure_threshold,
        )
        self.stats = RuntimeStats()
        self._categories: set = set()
        self._execution_log: List[Dict] = []

        # Load persisted state if available
        if self.config.persistence_path:
            self._load_state()

        logger.info(f"CogniCoreRuntime '{name}' initialized. "
                    f"Memory={self.config.enable_memory}, "
                    f"Reflection={self.config.enable_reflection}")

    # ------------------------------------------------------------------
    # Core: Execute with cognition
    # ------------------------------------------------------------------

    def execute(
        self,
        agent_fn: Callable,
        task: Any,
        category: str = "default",
        evaluator: Optional[Callable[[Any, Any], bool]] = None,
        max_retries: int = 0,
        **kwargs,
    ) -> ExecutionResult:
        """Execute an agent function with CogniCore cognition.

        Parameters
        ----------
        agent_fn : callable
            The agent function. Called as ``agent_fn(task, context)``
            where ``context`` is a dict with memory and reflection data.
        task : any
            The task/input to pass to the agent.
        category : str
            Category for memory grouping (e.g. "code_fix", "navigation").
        evaluator : callable or None
            Function ``(output, task) -> bool`` to judge success.
            If None, any non-exception result is considered success.
        max_retries : int
            Number of retry attempts if the first fails.
        **kwargs
            Extra args passed to ``agent_fn``.

        Returns
        -------
        ExecutionResult
            Full result with cognition metadata.
        """
        self.stats.total_executions += 1
        self._categories.add(category)
        self.stats.categories_seen = len(self._categories)

        # Build context from memory + reflection
        context = self._build_context(category)

        # Try execution with retries
        for attempt in range(1, max_retries + 2):
            result = self._single_execute(
                agent_fn, task, category, context, evaluator, attempt, **kwargs
            )
            if result.success or attempt > max_retries:
                break
            # On failure, enrich context with failure info for retry
            context["last_failure"] = result.error or str(result.output)
            context["attempt"] = attempt + 1
            context["reflection_hint"] = self.reflection.get_hint(category)

        # Persist
        if self.config.auto_save and self.config.persistence_path:
            self._save_state()

        return result

    def _single_execute(
        self,
        agent_fn: Callable,
        task: Any,
        category: str,
        context: Dict,
        evaluator: Optional[Callable],
        attempt: int,
        **kwargs,
    ) -> ExecutionResult:
        """Execute once and record the result."""
        result = ExecutionResult(category=category, attempt=attempt)
        result.memory_context = context.get("memory", [])
        result.reflection_hint = context.get("reflection_hint")

        t0 = time.perf_counter()
        try:
            output = agent_fn(task, context, **kwargs)
            result.output = output
            result.duration_ms = (time.perf_counter() - t0) * 1000
            self.stats.total_duration_ms += result.duration_ms

            # Evaluate success
            if evaluator:
                result.success = evaluator(output, task)
            else:
                result.success = True  # No crash = success

        except Exception as e:
            result.duration_ms = (time.perf_counter() - t0) * 1000
            self.stats.total_duration_ms += result.duration_ms
            result.error = str(e)
            result.success = False
            logger.warning(f"Execution failed: {e}")

        # Update stats
        if result.success:
            self.stats.successes += 1
        else:
            self.stats.failures += 1

        # Store in memory
        if self.config.enable_memory:
            self.memory.store({
                self.config.memory_similarity_key: category,
                "predicted": str(result.output)[:500],
                "correct": result.success,
                "task": str(task)[:200],
                "duration_ms": result.duration_ms,
                "attempt": attempt,
                "error": result.error,
            })

        # Log execution
        self._execution_log.append(result.to_dict())

        if self.config.verbose:
            status = "OK" if result.success else "FAIL"
            logger.info(f"[{status}] {category} | {result.duration_ms:.0f}ms | "
                       f"attempt {attempt}")

        return result

    def _build_context(self, category: str) -> Dict[str, Any]:
        """Build cognition context from memory + reflection."""
        context: Dict[str, Any] = {
            "memory": [],
            "reflection_hint": None,
            "failures_to_avoid": [],
            "successful_patterns": [],
            "category": category,
        }

        if self.config.enable_memory:
            context["memory"] = self.memory.get_context(
                category, top_k=self.config.memory_top_k
            )
            context["failures_to_avoid"] = [
                str(e.get("predicted", ""))
                for e in self.memory.retrieve_failures(category, top_k=5)
            ]
            context["successful_patterns"] = [
                str(e.get("predicted", ""))
                for e in self.memory.retrieve_successes(category, top_k=5)
            ]
            self.stats.memory_retrievals += 1

            # Check if we're about to repeat a known failure
            if context["failures_to_avoid"]:
                self.stats.repeated_failures_avoided += 1

        if self.config.enable_reflection:
            hint = self.reflection.get_hint(category)
            if hint:
                context["reflection_hint"] = hint

        return context

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def wrap(
        self,
        category: str = "default",
        evaluator: Optional[Callable] = None,
        max_retries: int = 0,
    ):
        """Decorator to wrap any function with CogniCore cognition.

        Usage:
            @runtime.wrap(category="code_fix")
            def fix_code(task, context):
                return llm.generate(task + str(context))
        """
        def decorator(fn):
            def wrapper(task, **kwargs):
                return self.execute(
                    agent_fn=fn, task=task, category=category,
                    evaluator=evaluator, max_retries=max_retries, **kwargs
                )
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            wrapper._cognicore_wrapped = True
            return wrapper
        return decorator

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics."""
        return {
            "runtime": self.stats.to_dict(),
            "memory": self.memory.stats(),
            "reflection": self.reflection.stats(),
        }

    def get_failure_report(self) -> Dict[str, Any]:
        """Get detailed failure analysis per category."""
        report = {}
        for cat in self._categories:
            failures = self.memory.retrieve_failures(cat, top_k=50)
            successes = self.memory.retrieve_successes(cat, top_k=50)
            analysis = self.reflection.analyze(cat)
            report[cat] = {
                "total_failures": len(failures),
                "total_successes": len(successes),
                "bad_patterns": analysis["bad_predictions"],
                "good_patterns": analysis["good_predictions"],
                "recommendation": analysis["recommendation"],
                "hint": self.reflection.get_hint(cat),
            }
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        path = Path(self.config.persistence_path)
        path.mkdir(parents=True, exist_ok=True)
        self.memory.save(path / f"{self.name}_memory.json")
        # Save stats
        stats_path = path / f"{self.name}_stats.json"
        stats_path.write_text(json.dumps(self.stats.to_dict(), indent=2))

    def _load_state(self):
        path = Path(self.config.persistence_path)
        mem_path = path / f"{self.name}_memory.json"
        if mem_path.exists():
            self.memory.load(mem_path)
            logger.info(f"Loaded {len(self.memory.entries)} memory entries")

    def save(self, path: Optional[str] = None):
        """Manually save runtime state."""
        if path:
            self.config.persistence_path = path
        if self.config.persistence_path:
            self._save_state()

    def reset(self):
        """Reset all runtime state."""
        self.memory.clear()
        self.stats = RuntimeStats()
        self._categories.clear()
        self._execution_log.clear()
