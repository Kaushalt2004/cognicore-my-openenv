"""
CogniCore Quickstart -- Minimal example of using the framework.

Run:
    python examples/quickstart.py
"""

import sys
import io
import cognicore

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    print("=" * 60)
    print("CogniCore v{} -- Quickstart".format(cognicore.__version__))
    print("=" * 60)

    # List available environments
    print("\nAvailable environments:")
    for env_info in cognicore.list_envs():
        print(f"   - {env_info['id']}: {env_info['description'][:60]}...")

    # Create an environment
    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    print(f"\nCreated: {env}")
    print(f"   Action space: {env.action_space}")

    # Run one episode with known answers
    obs = env.reset()
    print(f"\nEpisode started -- {obs['max_steps']} steps")

    ground_truths = [
        "SAFE", "UNSAFE", "SAFE", "UNSAFE", "SAFE",
        "UNSAFE", "SAFE", "UNSAFE", "SAFE", "UNSAFE",
    ]

    for i, gt in enumerate(ground_truths):
        obs, reward, done, truncated, info = env.step({"classification": gt})

        correct = info["eval_result"]["correct"]
        status = "[OK]" if correct else "[WRONG]"
        print(
            f"   Step {i+1:2d}: {status:7s} action={gt:12s} "
            f"reward={reward.total:+.2f} "
            f"(base={reward.base_score:.1f} mem={reward.memory_bonus:+.2f})"
        )

    # Episode summary
    stats = env.episode_stats()
    print(f"\nEpisode Results:")
    print(f"   Score:    {env.get_score():.4f}")
    print(f"   Accuracy: {stats.accuracy:.0%}")
    print(f"   Correct:  {stats.correct_count}/{stats.steps}")
    print(f"   Memory entries: {stats.memory_entries_created}")

    # Show state
    state = env.state()
    print(f"\nAgent Status: {state['agent_status']}")
    print(f"   Memory: {state['memory_stats']['total_entries']} entries")

    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
