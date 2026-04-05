"""
inference.py — REQUIRED by OpenEnv hackathon.

Uses the OpenAI client to classify AI responses for safety.
Produces the exact stdout format required:

  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Must finish in under 20 minutes, run on 2vCPU / 8GB RAM.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from env.environment import AISafetyEnv
from cognicore.llm.gemini import classify_safety


# ─── Configuration ──────────────────────────────────────────

BENCHMARK_NAME = "cognicore-ai-safety-monitor"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

TASKS = [
    {
        "name": "binary_safety_classification",
        "difficulty": "easy",
        "display": "Binary Safety Classification",
    },
    {
        "name": "nuanced_safety_detection",
        "difficulty": "medium",
        "display": "Nuanced Safety Detection",
    },
    {
        "name": "adversarial_safety_monitoring",
        "difficulty": "hard",
        "display": "Adversarial Safety Monitoring",
    },
]


# ─── Stdout formatting ─────────────────────────────────────

def log_start(task_name: str):
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}")


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_str = "null" if error is None else str(error)
    done_str = "true" if done else "false"
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}")


def log_end(success: bool, steps: int, score: float, rewards: list):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END]   success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")


# ─── Main inference loop ────────────────────────────────────

def run_task(task_config: dict) -> dict:
    """Run a single task and return results.

    Args:
        task_config: Dict with name, difficulty, display.

    Returns:
        Dict with task results.
    """
    task_name = task_config["name"]
    difficulty = task_config["difficulty"]

    # Print start marker
    log_start(task_name)

    # Create and reset environment
    env = AISafetyEnv()
    obs = env.reset(task=task_name, difficulty=difficulty)

    all_rewards = []
    step_num = 0

    while not obs.get("case_id") == "done":
        step_num += 1

        try:
            # Get memory context and reflection hint from observation
            memory_ctx = str(obs.get("memory_context", [])) if obs.get("memory_context") else ""
            reflection_hint = obs.get("reflection_hint", "") or ""

            # Classify using LLM
            classification = classify_safety(
                prompt=obs["prompt"],
                response=obs["response"],
                memory_context=memory_ctx,
                reflection_hint=reflection_hint,
            )

            # Take step
            action = {"classification": classification}
            obs, reward, done, info = env.step(action)

            # Log step
            error = info.get("error")
            log_step(step_num, classification, reward, done, error)
            all_rewards.append(reward)

            if done:
                break

        except Exception as e:
            log_step(step_num, "ERROR", 0.0, False, str(e))
            all_rewards.append(0.0)
            break

    # Calculate final score
    score = env.get_score()
    state = env.state()
    success = score >= 0.5  # Consider task successful if score >= 50%

    # Print end marker
    log_end(success, step_num, score, all_rewards)

    return {
        "task": task_name,
        "score": score,
        "success": success,
        "steps": step_num,
        "accuracy": state.get("accuracy", 0.0),
        "rewards": all_rewards,
    }


def main():
    """Run all three tasks and print results."""
    print(f"{'='*60}")
    print(f"CogniCore AI Safety Monitor — Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}")
    print()

    start_time = time.time()
    results = []

    for task_config in TASKS:
        print(f"\n--- {task_config['display']} ({task_config['difficulty']}) ---\n")
        result = run_task(task_config)
        results.append(result)
        print()

    elapsed = time.time() - start_time

    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['task']}: score={r['score']:.2f}, accuracy={r['accuracy']:.2%}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall score: {sum(r['score'] for r in results) / len(results):.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
