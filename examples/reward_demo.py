"""Demo: Streak penalties, memory bonuses, and all 8 reward components."""
import sys, io
if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

import cognicore

env = cognicore.make("SafetyClassification-v1", difficulty="easy")
obs = env.reset()

# Answers: mix of correct and wrong to trigger streaks + memory
answers = [
    ("SAFE",    True,   "Correct - first encounter"),
    ("UNSAFE",  True,   "Correct - streak building"),
    ("SAFE",    True,   "Correct - 3 in a row!"),
    ("SAFE",    False,  "WRONG - streak broken"),
    ("SAFE",    False,  "WRONG x2 - streak penalty kicks in"),
    ("SAFE",    False,  "WRONG x3 - penalty continues"),
    ("UNSAFE",  True,   "Correct! Memory bonus (seen this category before)"),
    ("SAFE",    True,   "Correct - recovering"),
    ("UNSAFE",  True,   "Correct - memory helping"),
    ("UNSAFE",  True,   "Correct - strong finish"),
]

print("=" * 90)
print("CogniCore Reward Breakdown Demo")
print("=" * 90)
print(f"{'Step':<5} {'Action':<14} {'Base':>6} {'Memory':>8} {'Streak':>8} {'Novelty':>8} {'Reflect':>8} {'TOTAL':>8}  Notes")
print("-" * 90)

for i, (ans, should_be_correct, note) in enumerate(answers):
    obs, reward, done, _, info = env.step({"classification": ans})

    correct = info["eval_result"]["correct"]
    icon = "[OK]" if correct else "[XX]"

    print(
        f"{i+1:<5} {icon} {ans:<10} "
        f"{reward.base_score:>+6.2f} "
        f"{reward.memory_bonus:>+8.2f} "
        f"{reward.streak_penalty:>+8.2f} "
        f"{reward.novelty_bonus:>+8.2f} "
        f"{reward.reflection_bonus:>+8.2f} "
        f"{reward.total:>+8.2f}  "
        f"{note}"
    )

    if done:
        break

print("-" * 90)
stats = env.episode_stats()
state = env.state()
print(f"\nFinal Score: {env.get_score():.4f}")
print(f"Accuracy: {stats.accuracy:.0%} ({stats.correct_count}/{stats.steps})")
print(f"Memory entries: {state['memory_stats']['total_entries']}")
print(f"Agent status: {state['agent_status']}")

# Show memory contents
print(f"\n{'=' * 90}")
print("Memory Contents (what the agent learned):")
print(f"{'=' * 90}")
for entry in env.memory.entries[:5]:
    cat = entry.get("category", "?")
    pred = entry.get("predicted", "?")
    ok = entry.get("correct", False)
    icon = "[OK]" if ok else "[XX]"
    print(f"  {icon} category={cat:<25s} answered={pred}")
print(f"  ... ({len(env.memory.entries)} total entries)")
