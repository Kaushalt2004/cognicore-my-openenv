import cognicore as cc
from cognicore.smart_agents import AutoLearner

def run_benchmark():
    print("="*50)
    print("🧠 COGNICORE BENCHMARK: Baseline vs Middleware")
    print("="*50)
    print("\nSimulating 20 episodes of the Safety Classification task...\n")

    # 1. BASELINE: Memory and Reflection DISABLED (Standard Env)
    print("--- [1] Standard Environment (No Memory) ---")
    config_baseline = cc.CogniCoreConfig(enable_memory=False, enable_reflection=False)
    env_baseline = cc.make("SafetyClassification-v1", config=config_baseline)
    agent_baseline = AutoLearner()
    
    score_baseline = cc.evaluate(agent_baseline, env_baseline, episodes=20)
    print(f"Agent accuracy without memory: {score_baseline * 100:.1f}%\n")

    # 2. COGNICORE: Memory and Reflection ENABLED (CogniCore Env)
    print("--- [2] CogniCore Environment (With Memory & Feedback) ---")
    config_cognicore = cc.CogniCoreConfig(enable_memory=True, enable_reflection=True)
    env_cognicore = cc.make("SafetyClassification-v1", config=config_cognicore)
    agent_cognicore = AutoLearner()
    
    score_cognicore = cc.evaluate(agent_cognicore, env_cognicore, episodes=20)
    print(f"Agent accuracy with memory:    {score_cognicore * 100:.1f}%")

    print("\n" + "="*50)
    improvement = (score_cognicore - score_baseline) * 100
    print(f"🚀 Improvement via CogniCore: +{improvement:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmark()
