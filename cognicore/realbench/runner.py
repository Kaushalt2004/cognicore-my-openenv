"""
CogniCore RealBench Runner — A/B benchmark engine.

Runs any agent function WITH vs WITHOUT CogniCore cognition
on the same task set, then compares results statistically.
"""
from __future__ import annotations
import time, json, logging, statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from cognicore.runtime import CogniCoreRuntime, RuntimeConfig

logger = logging.getLogger("cognicore.realbench")


@dataclass
class TaskResult:
    task_id: str
    task_desc: str
    output: Any = None
    success: bool = False
    duration_ms: float = 0.0
    attempts: int = 1
    memory_used: bool = False
    reflection_used: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of a full benchmark run (baseline vs cognicore)."""
    benchmark_name: str = ""
    total_tasks: int = 0
    # Baseline
    baseline_successes: int = 0
    baseline_failures: int = 0
    baseline_avg_duration_ms: float = 0.0
    baseline_repeated_failures: int = 0
    # CogniCore
    cogni_successes: int = 0
    cogni_failures: int = 0
    cogni_avg_duration_ms: float = 0.0
    cogni_repeated_failures: int = 0
    cogni_memory_hits: int = 0
    cogni_reflection_hits: int = 0
    # Deltas
    success_improvement: float = 0.0
    failure_reduction: float = 0.0
    retry_reduction: float = 0.0
    adaptation_score: float = 0.0
    # Raw data
    baseline_results: List[Dict] = field(default_factory=list)
    cogni_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("baseline_results", None)
        d.pop("cogni_results", None)
        return d

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  BENCHMARK: {self.benchmark_name}",
            f"{'='*60}",
            f"  Tasks: {self.total_tasks}",
            f"",
            f"  {'Metric':<30} {'Baseline':>10} {'CogniCore':>10} {'Delta':>10}",
            f"  {'-'*60}",
            f"  {'Success Rate':<30} {self.baseline_successes}/{self.total_tasks:>7} {self.cogni_successes}/{self.total_tasks:>7} {self.success_improvement:>+9.1f}%",
            f"  {'Failures':<30} {self.baseline_failures:>10} {self.cogni_failures:>10} {self.failure_reduction:>+9.1f}%",
            f"  {'Avg Duration (ms)':<30} {self.baseline_avg_duration_ms:>10.1f} {self.cogni_avg_duration_ms:>10.1f}",
            f"  {'Repeated Failures':<30} {self.baseline_repeated_failures:>10} {self.cogni_repeated_failures:>10} {self.retry_reduction:>+9.1f}%",
            f"  {'Memory Hits':<30} {'--':>10} {self.cogni_memory_hits:>10}",
            f"  {'Reflection Hits':<30} {'--':>10} {self.cogni_reflection_hits:>10}",
            f"  {'Adaptation Score':<30} {'--':>10} {self.adaptation_score:>10.2f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class BenchmarkRunner:
    """Runs A/B comparison: baseline agent vs CogniCore-enhanced agent.

    Usage:
        runner = BenchmarkRunner("CodingBench")
        runner.add_task("fix_bug_1", "Fix off-by-one error", buggy_code, test_fn)
        result = runner.run(agent_fn=my_agent)
        print(result.summary())
    """

    def __init__(self, name: str = "RealBench",
                 runtime_config: Optional[RuntimeConfig] = None):
        self.name = name
        self.tasks: List[Dict[str, Any]] = []
        self.runtime_config = runtime_config or RuntimeConfig(
            enable_memory=True,
            enable_reflection=True,
            reflection_min_samples=2,
            reflection_failure_threshold=2,
        )

    def add_task(self, task_id: str, description: str,
                 task_data: Any, evaluator: Callable[[Any, Any], bool],
                 category: str = "default"):
        """Add a real task to the benchmark."""
        self.tasks.append({
            "task_id": task_id,
            "description": description,
            "data": task_data,
            "evaluator": evaluator,
            "category": category,
        })

    def run(self, agent_fn: Callable, max_retries: int = 2,
            verbose: bool = True) -> BenchmarkResult:
        """Run the full A/B benchmark.

        Executes every task TWICE:
          1. Baseline: agent_fn called directly, no memory/reflection
          2. CogniCore: agent_fn wrapped with runtime cognition

        Returns comparison results.
        """
        if verbose:
            print(f"\n  Running {self.name}: {len(self.tasks)} tasks")
            print(f"  Mode: A/B comparison (Baseline vs CogniCore)")

        # --- BASELINE RUN (no cognition) ---
        if verbose:
            print(f"\n  [A] BASELINE (no memory, no reflection)")
        baseline_results = self._run_baseline(agent_fn, max_retries, verbose)

        # --- COGNICORE RUN (with cognition) ---
        if verbose:
            print(f"\n  [B] COGNICORE (memory + reflection enabled)")
        runtime = CogniCoreRuntime(config=self.runtime_config, name=self.name)
        cogni_results = self._run_cognicore(agent_fn, runtime, max_retries, verbose)

        # --- COMPUTE METRICS ---
        result = self._compute_metrics(baseline_results, cogni_results, runtime)

        if verbose:
            print(result.summary())

        return result

    def _run_baseline(self, agent_fn: Callable, max_retries: int,
                      verbose: bool) -> List[TaskResult]:
        results = []
        for task in self.tasks:
            tr = TaskResult(task_id=task["task_id"], task_desc=task["description"])
            t0 = time.perf_counter()

            for attempt in range(1, max_retries + 2):
                try:
                    # Baseline: no context at all
                    output = agent_fn(task["data"], {})
                    tr.output = output
                    tr.success = task["evaluator"](output, task["data"])
                    if tr.success:
                        break
                except Exception as e:
                    tr.error = str(e)
                tr.attempts = attempt

            tr.duration_ms = (time.perf_counter() - t0) * 1000
            results.append(tr)
            if verbose:
                s = "PASS" if tr.success else "FAIL"
                print(f"    [{s}] {task['task_id']} ({tr.attempts} attempts, {tr.duration_ms:.0f}ms)")

        return results

    def _run_cognicore(self, agent_fn: Callable, runtime: CogniCoreRuntime,
                       max_retries: int, verbose: bool) -> List[TaskResult]:
        results = []
        for task in self.tasks:
            tr = TaskResult(task_id=task["task_id"], task_desc=task["description"])
            t0 = time.perf_counter()

            exec_result = runtime.execute(
                agent_fn=agent_fn,
                task=task["data"],
                category=task["category"],
                evaluator=task["evaluator"],
                max_retries=max_retries,
            )

            tr.output = exec_result.output
            tr.success = exec_result.success
            tr.attempts = exec_result.attempt
            tr.duration_ms = (time.perf_counter() - t0) * 1000
            tr.error = exec_result.error
            tr.memory_used = len(exec_result.memory_context) > 0
            tr.reflection_used = exec_result.reflection_hint is not None

            results.append(tr)
            if verbose:
                s = "PASS" if tr.success else "FAIL"
                mem = " [MEM]" if tr.memory_used else ""
                refl = " [REFL]" if tr.reflection_used else ""
                print(f"    [{s}] {task['task_id']} ({tr.attempts} attempts, {tr.duration_ms:.0f}ms){mem}{refl}")

        return results

    def _compute_metrics(self, baseline: List[TaskResult],
                         cogni: List[TaskResult],
                         runtime: CogniCoreRuntime) -> BenchmarkResult:
        n = len(self.tasks)
        b_succ = sum(1 for r in baseline if r.success)
        c_succ = sum(1 for r in cogni if r.success)
        b_fail = n - b_succ
        c_fail = n - c_succ
        b_dur = statistics.mean([r.duration_ms for r in baseline]) if baseline else 0
        c_dur = statistics.mean([r.duration_ms for r in cogni]) if cogni else 0
        b_retries = sum(r.attempts - 1 for r in baseline)
        c_retries = sum(r.attempts - 1 for r in cogni)
        mem_hits = sum(1 for r in cogni if r.memory_used)
        refl_hits = sum(1 for r in cogni if r.reflection_used)

        success_imp = ((c_succ - b_succ) / max(1, b_succ)) * 100 if b_succ else 0
        fail_red = ((b_fail - c_fail) / max(1, b_fail)) * 100 if b_fail else 0
        retry_red = ((b_retries - c_retries) / max(1, b_retries)) * 100 if b_retries else 0
        adapt = (c_succ / max(1, n)) * (1 + mem_hits / max(1, n)) * (1 + refl_hits / max(1, n))

        return BenchmarkResult(
            benchmark_name=self.name,
            total_tasks=n,
            baseline_successes=b_succ, baseline_failures=b_fail,
            baseline_avg_duration_ms=b_dur, baseline_repeated_failures=b_retries,
            cogni_successes=c_succ, cogni_failures=c_fail,
            cogni_avg_duration_ms=c_dur, cogni_repeated_failures=c_retries,
            cogni_memory_hits=mem_hits, cogni_reflection_hits=refl_hits,
            success_improvement=success_imp, failure_reduction=fail_red,
            retry_reduction=retry_red, adaptation_score=adapt,
            baseline_results=[asdict(r) for r in baseline],
            cogni_results=[asdict(r) for r in cogni],
        )

    def save_report(self, result: BenchmarkResult, path: str):
        """Save benchmark results to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result.to_dict(), indent=2, default=str))
