"""
Example 5: Human Agent — Interactive CLI where YOU are the agent.

CogniCore evaluates your answers and shows you your memory,
streaks, and structured reward breakdown in real time.
"""

import sys, io
if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

import cognicore


def run_human_agent():
    """Interactive human-in-the-loop environment."""

    print("\n" + "=" * 65)
    print("  Agent Type: HUMAN (you!)")
    print("  You will classify AI responses as SAFE, UNSAFE, or NEEDS_REVIEW")
    print("  Type your answer for each prompt. Type 'quit' to stop.")
    print("=" * 65)

    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    obs = env.reset()
    step = 0

    while True:
        step += 1
        print(f"\n--- Step {step}/10 ---")
        print(f"Category: {obs.get('category', '?')}")
        print(f"Prompt: {obs.get('prompt', '?')[:200]}")

        # Show memory context if available
        if obs.get("memory_context"):
            print(f"\n  [MEMORY] You've seen this category before:")
            for m in obs["memory_context"][:2]:
                status = "correct" if m.get("correct") else "WRONG"
                print(f"    - You said '{m.get('predicted','?')}' -> {status}")

        # Show reflection hints
        if obs.get("reflection_hints"):
            print(f"  [HINT] {obs['reflection_hints']}")

        # Get human input
        while True:
            answer = input("\n  Your answer (SAFE/UNSAFE/NEEDS_REVIEW): ").strip().upper()
            if answer == "QUIT":
                print("\n  Quitting early.")
                stats = env.episode_stats()
                print(f"  Score: {env.get_score():.4f} | Accuracy: {stats.accuracy:.0%}")
                return
            if answer in ("SAFE", "UNSAFE", "NEEDS_REVIEW", "S", "U", "N"):
                if answer == "S": answer = "SAFE"
                if answer == "U": answer = "UNSAFE"
                if answer == "N": answer = "NEEDS_REVIEW"
                break
            print("  Please enter SAFE, UNSAFE, NEEDS_REVIEW (or S/U/N)")

        obs, reward, done, _, info = env.step({"classification": answer})
        correct = info["eval_result"]["correct"]
        truth = info["eval_result"]["ground_truth"]

        if correct:
            print(f"  >> CORRECT! Reward: {reward.total:+.2f}")
        else:
            print(f"  >> WRONG. The answer was: {truth}. Reward: {reward.total:+.2f}")

        # Show reward components
        if reward.memory_bonus > 0:
            print(f"     Memory bonus: +{reward.memory_bonus:.2f} (you remembered this category!)")
        if reward.streak_penalty < 0:
            print(f"     Streak penalty: {reward.streak_penalty:.2f} (consecutive mistakes)")
        if reward.novelty_bonus > 0:
            print(f"     Novelty bonus: +{reward.novelty_bonus:.2f} (first time seeing this!)")

        if done:
            break

    stats = env.episode_stats()
    state = env.state()
    print(f"\n{'=' * 65}")
    print(f"  FINAL RESULTS")
    print(f"  Score: {env.get_score():.4f}")
    print(f"  Accuracy: {stats.accuracy:.0%} ({stats.correct_count}/{stats.steps})")
    print(f"  Memory: {state['memory_stats']['total_entries']} entries")
    print(f"  Status: {state['agent_status']}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_human_agent()
