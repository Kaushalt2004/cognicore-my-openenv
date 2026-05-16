#!/usr/bin/env python3
"""
CogniCore RealBench — Run All Benchmarks

Executes REAL benchmarks comparing baseline vs CogniCore-enhanced agents.
No synthetic data. No fake environments. Actual code execution.

Run:
    python examples/run_realbench.py
    python examples/run_realbench.py --coding
    python examples/run_realbench.py --workflow
    python examples/run_realbench.py --all
"""
import sys, os, io, argparse, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognicore.realbench.coding_bench import CodingBenchmark, execute_code_safely
from cognicore.realbench.workflow_bench import WorkflowBenchmark


def banner(text):
    print(f"\n{'#'*64}")
    print(f"#  {text:<58} #")
    print(f"{'#'*64}")


def run_coding_benchmark():
    banner("CODING AGENT BENCHMARK (Real Python Bugs)")
    print("  10 real buggy Python programs")
    print("  Each fix is EXECUTED and tested against unit tests")
    print("  Comparing: Baseline (no learning) vs CogniCore (memory+reflection)")
    print()

    bench = CodingBenchmark()
    result = bench.run(verbose=True)
    return result


def run_workflow_benchmark():
    banner("WORKFLOW AGENT BENCHMARK (Real Pipelines)")
    print("  5 multi-step workflow pipelines")
    print("  Real API failure simulation (rate limits, timeouts, 500s)")
    print("  Comparing: Baseline vs CogniCore (retry adaptation)")
    print()

    bench = WorkflowBenchmark()
    result = bench.run(verbose=True)
    return result


def run_unit_verification():
    """Verify that the coding benchmark tasks are real by executing them."""
    banner("VERIFICATION: Executing All Bug Fixes")
    from cognicore.realbench.coding_bench import CODING_TASKS, simple_code_fixer

    passed = 0
    failed = 0
    for task in CODING_TASKS:
        fixed = simple_code_fixer(task, {})
        success, error = execute_code_safely(fixed, task["test_code"])
        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
        else:
            failed += 1
        detail = "" if success else f" ({error})"
        print(f"  [{status}] {task['id']}: {task['description']}{detail}")

    print(f"\n  {passed}/{passed+failed} fixes verified (actual code execution)")
    return passed, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogniCore RealBench")
    parser.add_argument("--coding", action="store_true", help="Run coding benchmark")
    parser.add_argument("--workflow", action="store_true", help="Run workflow benchmark")
    parser.add_argument("--verify", action="store_true", help="Verify all bug fixes execute")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--report", type=str, help="Save JSON report to path")
    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.coding, args.workflow, args.verify, args.all]):
        args.all = True

    print("\n" + "="*64)
    print("  CogniCore RealBench — Real-World AI Agent Benchmarking")
    print("  No synthetic data. No fake environments. Real execution.")
    print("="*64)

    t0 = time.perf_counter()
    results = {}

    if args.verify or args.all:
        passed, failed = run_unit_verification()

    if args.coding or args.all:
        results["coding"] = run_coding_benchmark()

    if args.workflow or args.all:
        results["workflow"] = run_workflow_benchmark()

    elapsed = time.perf_counter() - t0

    # Summary
    banner("FINAL SUMMARY")
    print(f"  Total runtime: {elapsed:.1f}s\n")

    for name, result in results.items():
        imp = result.success_improvement
        red = result.failure_reduction
        print(f"  {result.benchmark_name}:")
        print(f"    Baseline:  {result.baseline_successes}/{result.total_tasks} tasks passed")
        print(f"    CogniCore: {result.cogni_successes}/{result.total_tasks} tasks passed")
        print(f"    Improvement:    {imp:+.1f}%")
        print(f"    Failure reduction: {red:+.1f}%")
        print(f"    Memory hits:    {result.cogni_memory_hits}")
        print(f"    Reflection hits: {result.cogni_reflection_hits}")
        print()

    if args.report:
        report = {
            name: r.to_dict() for name, r in results.items()
        }
        report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["total_runtime_seconds"] = round(elapsed, 2)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved to: {args.report}")

    print("  CogniCore RealBench complete.")
    print("="*64)
