"""CogniCore -- Real AI Agent Demo."""
import cognicore

R = lambda r: float(getattr(r, "total", r))

print("=" * 70)
print("  CogniCore -- AI Agent Cognitive Testing")
print("=" * 70)

# 1. CODE DEBUGGING
print("\n[1] CODE DEBUGGING")
print("-" * 50)
env = cognicore.make("CodeDebugging-v1")
correct = 0
for ep in range(10):
    obs = env.reset()
    lines = obs["buggy_code"].strip().split("\n")
    action = {"bug_line": len(lines)//2, "fix_type": obs["category"]}
    _, reward, _, _, info = env.step(action)
    ic = info["eval_result"]["correct"]
    if ic: correct += 1
    if ep < 3:
        gt = info["eval_result"]["ground_truth"]
        print(f"  Ep {ep+1} | {obs['category']:15s} | {'OK' if ic else 'XX'} | r={R(reward):.2f} | expected: {gt}")
print(f"  Score: {correct}/10 | {env.episode_stats()}")

# 2. SAFETY CLASSIFICATION (action = {"classification": "SAFE/UNSAFE/NEEDS_REVIEW"})
print("\n[2] SAFETY CLASSIFICATION")
print("-" * 50)
env2 = cognicore.make("SafetyClassification-v1")
c2 = 0
for ep in range(10):
    obs = env2.reset()
    txt = str(obs).lower()
    bad = ["hack","attack","kill","bomb","weapon","steal","exploit","harm","malware","inject","phish"]
    guess = "UNSAFE" if any(w in txt for w in bad) else "SAFE"
    _, r2, _, _, info2 = env2.step({"classification": guess})
    ic = info2["eval_result"]["correct"]
    if ic: c2 += 1
    if ep < 3:
        p = str(list(obs.values())[0])[:80]
        print(f"  {'OK' if ic else 'XX'} | {p}... -> {guess} (r={R(r2):.2f})")
print(f"  Accuracy: {c2}/10")

# 3. MATH (action = {"answer": "..."})
print("\n[3] MATH REASONING")
print("-" * 50)
env3 = cognicore.make("MathReasoning-Easy-v1")
for ep in range(3):
    obs = env3.reset()
    problem = str(list(obs.values())[0])[:120]
    print(f"  {problem}")
    _, r3, _, _, info3 = env3.step({"answer": "42"})
    gt = info3["eval_result"]["ground_truth"]
    print(f"  Agent: 42 | Expected: {gt} | r={R(r3):.2f}")

# 4. CONVERSATION (action = {"response": "..."})
print("\n[4] CONVERSATION")
print("-" * 50)
env4 = cognicore.make("Conversation-Easy-v1")
for ep in range(3):
    obs = env4.reset()
    scenario = str(list(obs.values())[0])[:100]
    print(f"  {scenario}...")
    _, r4, _, _, info4 = env4.step({"response": "I understand your concern."})
    ic = info4["eval_result"]["correct"]
    print(f"  {'OK' if ic else 'XX'} | r={R(r4):.2f}")

# 5. MEMORY LEARNING
print("\n[5] COGNITIVE MEMORY -- Learning across episodes")
print("-" * 50)
env5 = cognicore.make("CodeDebugging-Easy-v1")
scores = []
for ep in range(10):
    obs = env5.reset()
    mem = len(obs.get("memory_context", []))
    lines = obs["buggy_code"].strip().split("\n")
    _, r5, _, _, info5 = env5.step({"bug_line": len(lines)//2, "fix_type": obs["category"]})
    ic = info5["eval_result"]["correct"]
    rc = info5["reward_components"]
    scores.append(1 if ic else 0)
    acc = sum(scores)/len(scores)*100
    print(f"  Ep {ep+1:2d} | Mem:{mem:2d} | MemBonus:{rc['memory_bonus']:+.2f} | ReflBonus:{rc['reflection_bonus']:+.2f} | {'OK' if ic else 'XX'} | Acc:{acc:.0f}%")

print(f"\n  Final: {sum(scores)}/{len(scores)}")

print("\n" + "=" * 70)
print("  50 envs | Memory + Reflection + Rewards")
print("  CogniCore: cognitive testing for AI agents")
print("=" * 70)
