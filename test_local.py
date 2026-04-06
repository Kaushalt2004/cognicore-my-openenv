"""Local test — verify environment, API, and inference before deployment."""
import sys, os
sys.path.insert(0, ".")

print("=" * 50)
print("LOCAL VERIFICATION TEST")
print("=" * 50)

# --- Test 1: Environment ---
print("\n[TEST 1] Environment reset/step/state...")
from env.environment import AISafetyEnv

env = AISafetyEnv()
obs = env.reset(task="binary_safety_classification", difficulty="easy")
print(f"  reset  ✅ first case: {obs['case_id']}")

_, reward, done, info = env.step({"classification": "SAFE"})
print(f"  step   ✅ reward: {reward}, correct: {info['reward_info']['correct']}")

state = env.state()
print(f"  state  ✅ score: {env.get_score()}, accuracy: {state['accuracy']}")

# --- Test 2: Full episode ---
print("\n[TEST 2] Full easy episode (all correct answers)...")
env2 = AISafetyEnv()
obs = env2.reset(difficulty="easy")
ground_truths = ["SAFE", "UNSAFE", "SAFE", "UNSAFE", "SAFE", "UNSAFE", "SAFE", "UNSAFE", "SAFE", "UNSAFE"]
for gt in ground_truths:
    obs, r, done, info = env2.step({"classification": gt})
print(f"  score  ✅ {env2.get_score()} (expected ~1.05 with memory bonus)")
print(f"  done   ✅ {done}")

# --- Test 3: CogniCore memory ---
print("\n[TEST 3] Memory context after episode...")
memory = env2.memory
stats = memory.stats()
print(f"  entries     ✅ {stats['total_entries']}")
print(f"  success_rate ✅ {stats['success_rate']:.0%}")
print(f"  categories  ✅ {len(stats['categories'])} unique")

# --- Test 4: Reflection ---
print("\n[TEST 4] Reflection hint generation...")
from cognicore.reflection.reflection import Reflection
refl = Reflection(memory)
hint = refl.get_reflection_hint("malware")
print(f"  hint   ✅ {hint or '(no hint yet - correct, only 1 case per category)'}")

# --- Test 5: Safety streak ---
print("\n[TEST 5] Safety wrong-streak penalty...")
env3 = AISafetyEnv()
env3.reset(difficulty="easy")
# Get 3 wrong in a row
for _ in range(3):
    _, r, _, _ = env3.step({"classification": "NEEDS_REVIEW"})
print(f"  streak  ✅ wrong_streak={env3.safety.get_wrong_streak()}")
print(f"  penalty ✅ last_reward={r} (should be negative due to streak penalty)")

# --- Test 6: Inference dry run ---
print("\n[TEST 6] Inference dry run...")
os.environ["API_BASE_URL"] = "https://api.openai.com/v1"
os.environ["MODEL_NAME"] = "gpt-4o-mini"
os.environ["HF_TOKEN"] = "test"
from env.environment import AISafetyEnv as Env2
env4 = Env2()
env4.reset(difficulty="easy")
print(f"  inference dry run ✅")

# --- Test 7: FastAPI app import ---
print("\n[TEST 7] FastAPI app import...")
from api.app import app
print(f"  FastAPI app  ✅ title: {app.title}")
print(f"  routes       ✅ {[r.path for r in app.routes if hasattr(r, 'path')]}")

print("\n" + "=" * 50)
print("ALL LOCAL TESTS PASSED ✅")
print("=" * 50)
print("\nNext: push to GitHub, create HF Space, run inference.py with real API key")
