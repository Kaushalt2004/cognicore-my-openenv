"""
Test real company models on CogniCore environments.

This script shows how to plug ANY company's AI model into CogniCore.
Just set the API key and run.

Usage:
  # Test OpenAI GPT-4o-mini on safety
  set OPENAI_API_KEY=sk-...
  python examples/test_company_models.py --provider openai

  # Test Google Gemini on GridWorld
  set GEMINI_API_KEY=AI...
  python examples/test_company_models.py --provider gemini

  # Test Ollama (FREE, local) on everything
  ollama pull llama3
  python examples/test_company_models.py --provider ollama

  # Test all available providers
  python examples/test_company_models.py --all
"""
import cognicore as cc
import argparse
import os
import sys
import time


def test_agent_on_env(agent, env_id, episodes=5, label="Agent"):
    """Test a single agent on a single environment."""
    env = cc.make(env_id)
    rewards = []
    correct_count = 0
    total_steps = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        steps = 0

        while True:
            try:
                action = agent.act(obs)
            except Exception as e:
                print(f"    [!] Agent error: {e}")
                action = {"action": "UP", "classification": "SAFE"}

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward.total
            steps += 1

            if done or truncated:
                break

        rewards.append(ep_reward)
        total_steps += steps
        print(f"    Episode {ep}: reward={ep_reward:+.1f} | steps={steps}")

    avg = sum(rewards) / len(rewards)
    print(f"    --- Avg reward: {avg:+.1f} over {episodes} episodes ---")
    return avg


def test_provider(provider_name, env_ids=None):
    """Test a specific provider across environments."""
    if env_ids is None:
        env_ids = ["RealWorldSafety-v1", "GridWorld-v1", "ResourceGathering-v1"]

    print()
    print("=" * 65)
    print(f"  Testing: {provider_name}")
    print("=" * 65)

    agent = None

    if provider_name == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  [!] Set OPENAI_API_KEY env var first")
            print("      export OPENAI_API_KEY='sk-...'")
            return
        agent = cc.OpenAIAgent(model="gpt-4o-mini")
        print(f"  Model: gpt-4o-mini (OpenAI)")

    elif provider_name == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            print("  [!] Set GEMINI_API_KEY env var first")
            print("      export GEMINI_API_KEY='AI...'")
            return
        agent = cc.GeminiAgent(model="gemini-2.0-flash")
        print(f"  Model: gemini-2.0-flash (Google)")

    elif provider_name == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("  [!] Set ANTHROPIC_API_KEY env var first")
            print("      export ANTHROPIC_API_KEY='sk-ant-...'")
            return
        agent = cc.ClaudeAgent(model="claude-3-haiku-20240307")
        print(f"  Model: claude-3-haiku (Anthropic)")

    elif provider_name == "ollama":
        agent = cc.OllamaAgent(model="llama3")
        print(f"  Model: llama3 (Ollama, FREE local)")
        print(f"  Make sure Ollama is running: ollama serve")

    elif provider_name == "huggingface":
        agent = cc.HuggingFaceAgent(model="mistralai/Mistral-7B-Instruct-v0.2")
        print(f"  Model: Mistral-7B (HuggingFace, FREE tier)")

    elif provider_name == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            print("  [!] Set GROQ_API_KEY env var first")
            print("      Get free key at: console.groq.com")
            return
        agent = cc.OpenAICompatibleAgent(
            base_url="https://api.groq.com/openai/v1",
            model="llama3-8b-8192",
            api_key=key,
        )
        print(f"  Model: llama3-8b (Groq, FREE tier)")

    elif provider_name == "lmstudio":
        agent = cc.OpenAICompatibleAgent(
            base_url="http://localhost:1234/v1",
            model="local-model",
        )
        print(f"  Model: local (LM Studio)")

    else:
        print(f"  Unknown provider: {provider_name}")
        print(f"  Available: openai, gemini, claude, ollama, huggingface, groq, lmstudio")
        return

    print()

    for env_id in env_ids:
        print(f"  [{env_id}]")
        start = time.time()
        avg = test_agent_on_env(agent, env_id, episodes=3, label=provider_name)
        elapsed = time.time() - start
        print(f"    Time: {elapsed:.1f}s | Avg latency: {elapsed/3:.1f}s/episode")
        print()


def compare_all_on_safety():
    """Compare all available providers on RealWorldSafety-v1."""
    print()
    print("=" * 65)
    print("  HEAD-TO-HEAD: All Models on RealWorldSafety-v1")
    print("=" * 65)

    results = {}

    # Always test baseline
    print("\n  [Random Baseline]")
    results["Random"] = test_agent_on_env(cc.RandomAgent(), "RealWorldSafety-v1", 5)

    # Test Q-Learning baseline
    print("\n  [Q-Learning]")
    results["Q-Learning"] = test_agent_on_env(
        cc.BanditAgent(["SAFE", "UNSAFE", "NEEDS_REVIEW"]),
        "RealWorldSafety-v1", 5,
    )

    # Test each available provider
    providers = {
        "openai": ("OPENAI_API_KEY", lambda: cc.OpenAIAgent(model="gpt-4o-mini")),
        "gemini": ("GEMINI_API_KEY", lambda: cc.GeminiAgent(model="gemini-2.0-flash")),
        "claude": ("ANTHROPIC_API_KEY", lambda: cc.ClaudeAgent(model="claude-3-haiku-20240307")),
        "groq": ("GROQ_API_KEY", lambda: cc.OpenAICompatibleAgent(
            base_url="https://api.groq.com/openai/v1",
            model="llama3-8b-8192",
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )),
    }

    for name, (env_var, factory) in providers.items():
        if os.environ.get(env_var):
            print(f"\n  [{name}]")
            try:
                agent = factory()
                results[name] = test_agent_on_env(agent, "RealWorldSafety-v1", 5)
            except Exception as e:
                print(f"    [!] Failed: {e}")
        else:
            print(f"\n  [{name}] -- SKIPPED (no {env_var})")

    # Leaderboard
    print()
    print("=" * 65)
    print("  LEADERBOARD")
    print("=" * 65)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * max(0, int(score))
        print(f"  {name:<20} {score:+6.1f}  {bar}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test company models on CogniCore")
    parser.add_argument("--provider", type=str, default=None,
                        help="Provider: openai, gemini, claude, ollama, huggingface, groq, lmstudio")
    parser.add_argument("--all", action="store_true", help="Compare all available models")
    parser.add_argument("--env", type=str, default=None, help="Specific env to test")
    args = parser.parse_args()

    print()
    print("*" * 65)
    print(f"  CogniCore v{cc.__version__} -- Company Model Testing")
    print("*" * 65)

    if args.all:
        compare_all_on_safety()
    elif args.provider:
        envs = [args.env] if args.env else None
        test_provider(args.provider, envs)
    else:
        print()
        print("  Usage:")
        print("    python test_company_models.py --provider openai")
        print("    python test_company_models.py --provider gemini")
        print("    python test_company_models.py --provider ollama  (FREE)")
        print("    python test_company_models.py --provider groq    (FREE)")
        print("    python test_company_models.py --all")
        print()
        print("  Available providers:")
        print("    openai      GPT-4o-mini      (needs OPENAI_API_KEY)")
        print("    gemini      Gemini 2.0 Flash  (needs GEMINI_API_KEY)")
        print("    claude      Claude 3 Haiku    (needs ANTHROPIC_API_KEY)")
        print("    groq        Llama3 8B         (needs GROQ_API_KEY, FREE)")
        print("    ollama      Llama3 local      (needs 'ollama serve', FREE)")
        print("    huggingface Mistral-7B        (needs HF_API_KEY, FREE)")
        print("    lmstudio    Local model       (needs LM Studio running)")
        print()
        print("  Set your API key, then run:")
        print("    set OPENAI_API_KEY=sk-...")
        print("    python test_company_models.py --provider openai --env RealWorldSafety-v1")
