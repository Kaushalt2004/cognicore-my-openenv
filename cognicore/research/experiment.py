"""
Experiment Tracker — reproducible experiment configuration and logging.
Every run stores seed, model, temperature, thresholds, timestamps.
"""
import json, time, os, random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Reproducible experiment configuration."""
    seed: int = 42
    model: str = "rule-based"
    temperature: float = 0.3
    max_attempts: int = 5
    similarity_threshold: float = 0.85
    benchmark_version: str = "v3.1"
    experiment_name: str = "coderepair"
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def seed_random(self):
        random.seed(self.seed)

    def to_dict(self):
        return asdict(self)

    def print_config(self):
        print(f"\n  ┌─ Experiment Config ─────────────────────┐")
        for k, v in self.to_dict().items():
            print(f"  │ {k:<25} {str(v):>15} │")
        print(f"  └───────────────────────────────────────────┘")


@dataclass
class ExperimentResult:
    """Results from a single bug repair experiment."""
    bug_id: str
    category: str
    title: str
    # Baseline
    baseline_solved: bool = False
    baseline_attempts: int = 0
    baseline_repeated: int = 0
    baseline_unique_patches: int = 0
    # CogniCore
    cogni_solved: bool = False
    cogni_attempts: int = 0
    cogni_repeated: int = 0
    cogni_unique_patches: int = 0
    cogni_memory_hits: int = 0
    cogni_reflections: int = 0
    cogni_mutations: int = 0
    cogni_rejections: int = 0
    cogni_strategy_changes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class ExperimentTracker:
    """Tracks and persists experiment data."""

    def __init__(self, config: ExperimentConfig, output_dir: str = "experiments"):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.cognition_events: List[Dict] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_result(self, result: ExperimentResult):
        self.results.append(result)

    def log_event(self, event_type: str, bug_id: str, detail: str = "",
                  metadata: Dict = None):
        self.cognition_events.append({
            "type": event_type, "bug_id": bug_id,
            "detail": detail[:200],
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

    def compute_metrics(self) -> Dict:
        n = len(self.results)
        if n == 0:
            return {}
        bs = sum(1 for r in self.results if r.baseline_solved)
        cs = sum(1 for r in self.results if r.cogni_solved)
        br = sum(r.baseline_repeated for r in self.results)
        cr = sum(r.cogni_repeated for r in self.results)
        ba = sum(r.baseline_attempts for r in self.results)
        ca = sum(r.cogni_attempts for r in self.results)
        tm = sum(r.cogni_memory_hits for r in self.results)
        tr = sum(r.cogni_reflections for r in self.results)
        tmu = sum(r.cogni_mutations for r in self.results)
        tj = sum(r.cogni_rejections for r in self.results)

        return {
            "total_bugs": n,
            "baseline_solved": bs, "cognicore_solved": cs,
            "success_improvement_pct": round(((cs - bs) / max(1, bs)) * 100, 1),
            "baseline_attempts": ba, "cognicore_attempts": ca,
            "attempt_reduction_pct": round(((ba - ca) / max(1, ba)) * 100, 1),
            "baseline_repeated": br, "cognicore_repeated": cr,
            "repeat_reduction_pct": round(((br - cr) / max(1, br)) * 100, 1) if br > 0 else 0,
            "memory_retrievals": tm, "reflections": tr,
            "strategy_mutations": tmu, "patch_rejections": tj,
            "memory_retrieval_rate": round(tm / max(1, ca) * 100, 1),
            "reflection_rate": round(tr / max(1, ca) * 100, 1),
            "mutation_rate": round(tmu / max(1, ca) * 100, 1),
        }

    def save(self, patch_store=None):
        report = {
            "config": self.config.to_dict(),
            "metrics": self.compute_metrics(),
            "results": [r.to_dict() for r in self.results],
            "cognition_events": self.cognition_events,
        }
        if patch_store:
            report["patch_history"] = patch_store.to_dict()

        fname = f"{self.config.experiment_name}_{self.config.timestamp.replace(':', '-')}.json"
        path = self.output_dir / fname
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return str(path)

    def print_report(self):
        m = self.compute_metrics()
        if not m:
            print("  No results to report.")
            return

        print(f"\n{'='*64}")
        print(f"  COGNICORE RESEARCH BENCHMARK — RESULTS")
        print(f"  {self.config.experiment_name} | {self.config.model} | seed={self.config.seed}")
        print(f"{'='*64}")

        print(f"\n  {'Bug':<10} {'Cat':<15} {'Base':>5} {'Cogni':>6} {'Mem':>4} {'Refl':>5} {'Mut':>4} {'Rej':>4}")
        print(f"  {'-'*55}")
        for r in self.results:
            print(f"  {r.bug_id:<10} {r.category:<15} "
                  f"{'PASS' if r.baseline_solved else 'FAIL':>5} "
                  f"{'PASS' if r.cogni_solved else 'FAIL':>6} "
                  f"{r.cogni_memory_hits:>4} {r.cogni_reflections:>5} "
                  f"{r.cogni_mutations:>4} {r.cogni_rejections:>4}")

        print(f"\n  {'Metric':<35} {'Baseline':>10} {'CogniCore':>10} {'Delta':>10}")
        print(f"  {'-'*65}")
        print(f"  {'Bugs Solved':<35} {m['baseline_solved']:>10} {m['cognicore_solved']:>10} {m['success_improvement_pct']:>+9.1f}%")
        print(f"  {'Total Attempts':<35} {m['baseline_attempts']:>10} {m['cognicore_attempts']:>10} {m['attempt_reduction_pct']:>+9.1f}%")
        print(f"  {'Repeated Patches':<35} {m['baseline_repeated']:>10} {m['cognicore_repeated']:>10} {m['repeat_reduction_pct']:>+9.1f}%")
        print(f"  {'Memory Retrievals':<35} {'--':>10} {m['memory_retrievals']:>10}")
        print(f"  {'Reflections':<35} {'--':>10} {m['reflections']:>10}")
        print(f"  {'Strategy Mutations':<35} {'--':>10} {m['strategy_mutations']:>10}")
        print(f"  {'Patch Rejections':<35} {'--':>10} {m['patch_rejections']:>10}")

        print(f"\n  VERDICT:")
        checks = [
            ("CogniCore solves >= baseline", m['cognicore_solved'] >= m['baseline_solved']),
            ("Repeated patches reduced", m['cognicore_repeated'] <= m['baseline_repeated']),
            ("Memory retrieval active", m['memory_retrievals'] > 0),
            ("Reflections generated", m['reflections'] > 0),
            ("Strategy mutations applied", m['strategy_mutations'] > 0),
            ("Patch rejections active", m['patch_rejections'] > 0),
        ]
        for label, ok in checks:
            s = "\033[32mPROVEN\033[0m" if ok else "\033[31mNOT PROVEN\033[0m"
            print(f"  [{s}] {label}")
        print(f"{'='*64}")
