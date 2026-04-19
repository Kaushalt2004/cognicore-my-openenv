"""
Custom Environment Example -- How to create your own CogniCore environment.

This example creates a simple Math Reasoning environment where agents
solve arithmetic problems. It demonstrates:

  - Subclassing CogniCoreEnv
  - Implementing the 4 abstract methods
  - Getting cognitive middleware (memory, reflection, rewards) for free
  - Using the PROPOSE -> Revise protocol

Run:
    python examples/custom_env.py
"""

import sys
import io
import random

import cognicore
from cognicore import CogniCoreEnv, CogniCoreConfig, EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace, TextSpace

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ===============================================================
# Step 1: Define your environment
# ===============================================================

class MathReasoningEnv(CogniCoreEnv):
    """Math reasoning environment -- solve arithmetic problems.

    Agents receive math questions and must provide the correct integer answer.
    Problems are categorized (addition, subtraction, multiplication, division)
    so the cognitive middleware (memory + reflection) can learn patterns
    within each category.
    """

    CATEGORIES = ["addition", "subtraction", "multiplication", "division"]

    def __init__(self, num_problems: int = 10, max_value: int = 50, **kwargs):
        self.num_problems = num_problems
        self.max_value = max_value
        super().__init__(**kwargs)

    def _setup(self, **kwargs):
        """Define what the agent sees and does."""
        self.observation_space = DictSpace(fields={
            "question": TextSpace(),
            "category": TextSpace(),
        })
        self.action_space = DictSpace(fields={
            "answer": DiscreteSpace(1000),
        })

    def _generate_tasks(self):
        """Generate random math problems."""
        tasks = []
        for _ in range(self.num_problems):
            category = random.choice(self.CATEGORIES)
            a = random.randint(1, self.max_value)
            b = random.randint(1, self.max_value)

            if category == "addition":
                q, answer = f"{a} + {b}", a + b
            elif category == "subtraction":
                q, answer = f"{a} - {b}", a - b
            elif category == "multiplication":
                a = random.randint(1, 12)
                b = random.randint(1, 12)
                q, answer = f"{a} * {b}", a * b
            else:  # division
                answer = random.randint(1, 20)
                b = random.randint(1, 10)
                a = answer * b
                q, answer = f"{a} / {b}", answer

            tasks.append({
                "question": q,
                "answer": answer,
                "category": category,
            })
        return tasks

    def _evaluate(self, action):
        """Grade the agent's answer."""
        task = self._tasks[self._current_step]
        agent_answer = action.get("answer")
        correct = agent_answer == task["answer"]

        return EvalResult(
            base_score=1.0 if correct else 0.0,
            correct=correct,
            ground_truth=task["answer"],
            predicted=agent_answer,
            category=task["category"],
        )

    def _get_obs(self):
        """Build the observation for the current problem."""
        task = self._tasks[self._current_step]
        return {
            "question": task["question"],
            "category": task["category"],
        }


# ===============================================================
# Step 2: Register it (optional -- allows cognicore.make())
# ===============================================================

cognicore.register(
    "MathReasoning-v1",
    entry_point=MathReasoningEnv,
    description="Math reasoning: solve arithmetic problems with cognitive middleware.",
    default_kwargs={"num_problems": 10, "max_value": 50},
)


# ===============================================================
# Step 3: Use it!
# ===============================================================

def main():
    print("=" * 60)
    print("CogniCore Custom Environment -- Math Reasoning")
    print("=" * 60)

    # Method 1: Direct instantiation
    env = MathReasoningEnv(num_problems=8, max_value=20)

    # Method 2: Via registry (equivalent)
    # env = cognicore.make("MathReasoning-v1", num_problems=8, max_value=20)

    # Run two episodes to show cross-episode learning
    for episode in range(1, 3):
        print(f"\n{'---' * 14}")
        print(f"Episode {episode}")
        print(f"{'---' * 14}")

        obs = env.reset()

        while True:
            # Simple agent: evaluate the expression
            question = obs["question"]
            try:
                answer = int(eval(question))
            except Exception:
                answer = 0

            obs, reward, done, truncated, info = env.step({"answer": answer})

            eval_r = info["eval_result"]
            status = "[OK]" if eval_r["correct"] else "[WRONG]"
            print(
                f"   {status:7s} {question:12s} = {answer:4d} "
                f"(truth={eval_r['ground_truth']}) "
                f"reward={reward.total:+.2f}"
            )

            if done:
                break

        stats = env.episode_stats()
        print(f"\n   Score: {env.get_score():.4f} | "
              f"Accuracy: {stats.accuracy:.0%} | "
              f"Memory: {stats.memory_entries_created} new entries")

    # Final memory stats
    mem_stats = env.memory.stats()
    print(f"\n{'=' * 60}")
    print(f"Cross-Episode Memory: {mem_stats['total_entries']} total entries")
    print(f"   Groups: {', '.join(mem_stats['groups'])}")
    print(f"   Success rate: {mem_stats['success_rate']:.0%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
