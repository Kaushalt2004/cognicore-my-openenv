"""
CogniCore CLI — Full command-line interface for every feature.

18 commands covering environments, training, testing, and analysis.

Usage::

    cognicore list                                        # List all 24 environments
    cognicore run SafetyClassification-v1 -v              # Run with random agent
    cognicore info MathReasoning-v1                       # Show environment details
    cognicore serve --port 8000                           # Start API server
    cognicore dashboard --port 8050                       # Start web dashboard
    cognicore leaderboard                                 # Show rankings

    cognicore benchmark --envs all --episodes 3           # Benchmark across all envs
    cognicore curriculum MathReasoning-v1                  # Auto-difficulty training
    cognicore improve SafetyClassification-v1              # Self-improvement loop
    cognicore evolve SafetyClassification-v1 --pop 20     # Evolutionary training

    cognicore stress SafetyClassification-v1              # Adversarial stress test
    cognicore battle --rounds 50                          # Red vs Blue simulation
    cognicore debug SafetyClassification-v1               # AI debugger

    cognicore explain --steps 20                          # Explainable AI demo
    cognicore iq SafetyClassification-v1                  # Intelligence scoring
    cognicore cost --model gemini-flash                   # Cost estimation

    cognicore transfer                                    # Knowledge transfer demo
    cognicore doctor                                      # Health check everything
"""

from __future__ import annotations

import argparse
import sys
import io
import time

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass

import cognicore
from cognicore.agents.base_agent import RandomAgent


# =====================================================================
# Original Commands (Phase 3)
# =====================================================================


def cmd_list(args):
    """List all registered environments."""
    envs = cognicore.list_envs()
    print(f"\nCogniCore v{cognicore.__version__} -- {len(envs)} environments\n")
    print(f"{'ID':<40s} {'Description'}")
    print("-" * 80)
    for e in envs:
        desc = (
            e["description"][:55] + "..."
            if len(e["description"]) > 55
            else e["description"]
        )
        print(f"{e['id']:<40s} {desc}")
    print()


def cmd_info(args):
    """Show detailed info about an environment."""
    try:
        env = cognicore.make(args.env_id, difficulty=args.difficulty)
    except KeyError as e:
        print(f"Error: {e}")
        return

    env.reset()
    print(f"\nEnvironment: {args.env_id}")
    print(f"  Class:      {env.__class__.__name__}")
    print(f"  Difficulty:  {getattr(env, 'difficulty', 'N/A')}")
    print(f"  Steps/Episode: {env._max_steps}")
    print(f"  Action Space:  {env.action_space}")
    print(f"  Obs Space:     {env.observation_space}")
    print()

    # Show first observation
    obs = env.reset()
    print("First observation keys:", list(obs.keys()))
    print()


def cmd_run(args):
    """Run an environment with a random agent."""
    try:
        env = cognicore.make(args.env_id, difficulty=args.difficulty)
    except KeyError as e:
        print(f"Error: {e}")
        return

    agent = RandomAgent(env.action_space)

    print(f"\nCogniCore v{cognicore.__version__}")
    print(f"Environment: {args.env_id} (difficulty={args.difficulty})")
    print("Agent: RandomAgent")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        total_correct = 0
        step_count = 0

        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

            if info.get("eval_result", {}).get("correct"):
                total_correct += 1

            if args.verbose:
                status = (
                    "[OK]" if info.get("eval_result", {}).get("correct") else "[  ]"
                )
                print(f"  Step {step_count:3d}: {status} reward={reward.total:+.2f}")

            if done:
                break

        stats = env.episode_stats()
        print(
            f"Episode {ep}: "
            f"score={env.get_score():.4f} "
            f"accuracy={stats.accuracy:.0%} "
            f"correct={stats.correct_count}/{stats.steps} "
            f"memory={stats.memory_entries_created}"
        )

    print("=" * 60)
    final_state = env.state()
    print(f"Memory: {final_state['memory_stats']['total_entries']} entries")
    print(f"Status: {final_state['agent_status']}")
    print()


def cmd_serve(args):
    """Start the API server."""
    try:
        import uvicorn
        from cognicore.server.app import create_app
    except ImportError:
        print("Error: Server requires FastAPI and Uvicorn.")
        print("Install with: pip install cognicore-env[server]")
        return

    print(f"\nCogniCore API Server v{cognicore.__version__}")
    print(f"  Docs:   http://{args.host}:{args.port}/docs")   # nosec B105 -- local dev URL only, not a live endpoint
    print(f"  Redoc:  http://{args.host}:{args.port}/redoc")  # nosec B105 -- local dev URL only, not a live endpoint
    print(f"  Envs:   {len(cognicore.list_envs())} available")
    print()

    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info" if args.verbose else "warning",
    )


def cmd_dashboard(args):
    """Start the web dashboard."""
    try:
        from cognicore.dashboard import serve_dashboard
    except ImportError:
        print("Error: Dashboard requires FastAPI and Uvicorn.")
        print("Install with: pip install cognicore-env[server]")
        return

    serve_dashboard(host=args.host, port=args.port)


def cmd_leaderboard(args):
    """Show leaderboard rankings."""
    from cognicore.leaderboard import Leaderboard

    lb = Leaderboard(storage_dir=args.storage)
    rankings = lb.get_rankings(env_id=args.env_id, top_k=args.top)

    if not rankings:
        print("\nNo leaderboard entries yet.")
        print("Run episodes and submit scores to populate the leaderboard.\n")
        return

    stats = lb.get_stats()
    print(
        f"\nCogniCore Leaderboard ({stats['total_submissions']} submissions, {stats['unique_agents']} agents)"
    )
    print("=" * 80)
    print(
        f"{'Rank':<6} {'Agent':<20} {'Environment':<30} {'Score':<10} {'Accuracy':<10}"
    )
    print("-" * 80)

    for r in rankings:
        print(
            f"#{r['rank']:<5} "
            f"{r['agent_id']:<20} "
            f"{r['env_id']:<30} "
            f"{r['score']:<10.4f} "
            f"{r['accuracy'] * 100:<9.0f}%"
        )
    print()


# =====================================================================
# Training Commands (Phase 4-6)
# =====================================================================


def cmd_benchmark(args):
    """Benchmark an agent across environments."""
    from cognicore.benchmark import benchmark_agent

    print("\nCogniCore Benchmark")
    print(f"  Environments: {args.envs}")
    print(f"  Difficulties: {args.difficulties}")
    print(f"  Episodes: {args.episodes}")
    print("=" * 60)

    difficulties = args.difficulties.split(",")
    results = benchmark_agent(
        envs=args.envs,
        difficulties=difficulties,
        episodes=args.episodes,
    )
    results.print_report()


def cmd_curriculum(args):
    """Run curriculum learning (auto-difficulty progression)."""
    from cognicore.curriculum import CurriculumRunner

    print("\nCogniCore Curriculum Learning")
    print(f"  Env: {args.env_id}")
    print(f"  Promotion: {args.threshold}")
    print(f"  Max episodes: {args.episodes}")
    print("=" * 60)

    runner = CurriculumRunner(
        args.env_id,
        promotion_threshold=args.threshold,
        demotion_threshold=args.demotion,
        window=args.window,
    )
    result = runner.run(max_episodes=args.episodes)
    print(f"\nResult: {result}")


def cmd_improve(args):
    """Self-improvement loop (run -> analyze -> improve -> repeat)."""
    from cognicore.auto_improve import auto_improve

    auto_improve(
        env_id=args.env_id,
        difficulty=args.difficulty,
        target_accuracy=args.target,
        max_cycles=args.cycles,
        episodes_per_cycle=args.episodes,
        patience=args.patience,
        verbose=True,
    )


def cmd_evolve(args):
    """Evolutionary training — natural selection for AI agents."""
    from cognicore.evolution import EvolutionEngine

    engine = EvolutionEngine(
        env_id=args.env_id,
        difficulty=args.difficulty,
        population_size=args.pop,
        elite_count=max(2, args.pop // 5),
        mutation_rate=args.mutation,
    )
    engine.evolve(generations=args.generations, verbose=True)


# =====================================================================
# Testing Commands (Phase 5-6)
# =====================================================================


def cmd_stress(args):
    """Adversarial stress test — find agent weaknesses."""
    from cognicore.adversarial import AdversarialTester

    tester = AdversarialTester(args.env_id, difficulty="hard")
    report = tester.stress_test(None, rounds=args.rounds, verbose=True)
    report.print_vulnerabilities()


def cmd_battle(args):
    """Red vs Blue battle simulation."""
    from cognicore.red_blue import RedVsBlue

    battle = RedVsBlue()
    result = battle.run(rounds=args.rounds, verbose=True)
    result.print_battle_report()


def cmd_debug(args):
    """AI debugger — breakpoints and step-through."""
    from cognicore.debugger import AIDebugger

    dbg = AIDebugger(args.env_id, difficulty=args.difficulty)

    if args.on_wrong:
        dbg.breakpoint(on_wrong_only=True, name="on_wrong")
    if args.category:
        dbg.breakpoint(category=args.category, name=f"bp_{args.category}")

    # If no breakpoints specified, break on wrong
    if not dbg.breakpoints:
        dbg.breakpoint(on_wrong_only=True, name="on_wrong")

    trace = dbg.run(verbose=True)
    trace.print_trace()


# =====================================================================
# Analysis Commands (Phase 5-6)
# =====================================================================


def cmd_explain(args):
    """Explainable AI — run agent and show why it fails."""
    from cognicore.explainer import Explainer

    env = cognicore.make(args.env_id, difficulty=args.difficulty)
    agent = RandomAgent(env.action_space)
    exp = Explainer()

    print(f"\nExplainable AI Analysis: {args.env_id}")
    print("=" * 60)

    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        er = info.get("eval_result", {})

        result = exp.record_step(
            step=step,
            category=er.get("category", "?"),
            predicted=str(er.get("predicted", "?")),
            truth=str(er.get("ground_truth", "?")),
            correct=er.get("correct", False),
            reward=reward.total,
            confidence=er.get("confidence", 0),
        )

        if not er.get("correct", True):
            print(
                f"  Step {step} [{er.get('category', '?')}]: {result.get('why_wrong', '')[:80]}"
            )

        if done:
            break

    report = exp.explain()
    report.print_report()


def cmd_iq(args):
    """Intelligence scoring — rate an agent across 6 dimensions."""
    from cognicore.intelligence import IntelligenceScorer

    env = cognicore.make(args.env_id, difficulty=args.difficulty)
    scorer = IntelligenceScorer()

    # Use AutoLearner for more interesting results
    try:
        from cognicore.smart_agents import AutoLearner

        agent = AutoLearner()
    except Exception:
        agent = RandomAgent(env.action_space)

    print(f"\nIntelligence Test: {args.env_id} ({args.difficulty})")
    print(f"  Agent: {getattr(agent, 'name', 'Agent')}")
    print(f"  Episodes: {args.episodes}")
    print("=" * 50)

    for ep in range(args.episodes):
        obs = env.reset()
        while True:
            t0 = time.time()
            action = agent.act(obs)
            latency = (time.time() - t0) * 1000

            obs, reward, done, _, info = env.step(action)
            er = info.get("eval_result", {})

            if hasattr(agent, "learn"):
                agent.learn(reward, info)

            scorer.record(
                step=ep * 100 + er.get("step", 0),
                category=er.get("category", "?"),
                correct=er.get("correct", False),
                reward_total=reward.total,
                memory_bonus=reward.memory_bonus,
                predicted=str(er.get("predicted", "")),
                truth=str(er.get("ground_truth", "")),
                latency_ms=latency,
                is_hard=(args.difficulty == "hard"),
            )

            if done:
                break

    iq = scorer.compute()
    iq.print_card()


def cmd_cost(args):
    """Cost estimation — estimate API costs for your usage."""
    from cognicore.cost_tracker import CostTracker

    tracker = CostTracker(model_name=args.model)

    # Simulate typical usage
    env = cognicore.make(args.env_id, difficulty=args.difficulty)
    obs = env.reset()
    step = 0

    while True:
        step += 1
        agent = RandomAgent(env.action_space)
        action = agent.act(obs)

        # Estimate token costs for this step
        prompt_text = str(obs.get("prompt", "")) + str(obs.get("memory_context", ""))
        response_text = str(action)

        tracker.record_text(
            input_text=prompt_text,
            output_text=response_text,
            episode=1,
            step=step,
        )

        obs, reward, done, _, info = env.step(action)
        if done:
            break

    print(f"\nCost Estimation: {args.env_id}")
    print(f"  Model: {args.model}")
    print(f"  Steps simulated: {step}")
    tracker.print_summary()

    # Scale projection
    total_per_ep = tracker.total_cost
    print("  Projected costs:")
    print(f"    10 episodes:    ${total_per_ep * 10:.4f}")
    print(f"    100 episodes:   ${total_per_ep * 100:.4f}")
    print(f"    1000 episodes:  ${total_per_ep * 1000:.4f}")
    print()


def cmd_transfer(args):
    """Knowledge transfer demo — expert teaches student."""
    from cognicore.smart_agents import AutoLearner
    from cognicore.knowledge_transfer import transfer_knowledge

    print("\nKnowledge Transfer Demo")
    print("=" * 55)

    # Train expert
    print("\n  Phase 1: Training expert agent...")
    expert = AutoLearner()
    env = cognicore.make(args.env_id, difficulty="easy")

    for ep in range(args.expert_episodes):
        obs = env.reset()
        while True:
            action = expert.act(obs)
            obs, reward, done, _, info = env.step(action)
            expert.learn(reward, info)
            if done:
                break
        stats = env.episode_stats()
        print(f"    Expert episode {ep + 1}: accuracy={stats.accuracy:.0%}")

    # Transfer
    print(f"\n  Phase 2: Transferring knowledge ({args.method})...")
    student = AutoLearner()
    result = transfer_knowledge(expert, student, method=args.method)
    print(f"    Transferred: {result['transferred']} categories")
    print(f"    Skipped: {result['skipped']}")

    # Test student
    print("\n  Phase 3: Testing student agent...")
    env = cognicore.make(args.env_id, difficulty="easy")
    obs = env.reset()
    while True:
        action = student.act(obs)
        obs, reward, done, _, info = env.step(action)
        student.learn(reward, info)
        if done:
            break

    stats = env.episode_stats()
    print(f"    Student accuracy (first try): {stats.accuracy:.0%}")
    print("\n  Knowledge transfer complete!")
    print("=" * 55)
    print()


# =====================================================================
# Phase 8 Commands
# =====================================================================


def cmd_swarm(args):
    """Swarm intelligence — multi-agent collaboration."""
    from cognicore.swarm import Swarm

    swarm = Swarm(size=args.size, diversity=True)
    result = swarm.solve(
        args.env_id, difficulty=args.difficulty, episodes=args.episodes
    )
    result.print_report()


def cmd_lifelong(args):
    """Lifelong learning — persistent agent across sessions."""
    from cognicore.lifelong import LifelongAgent

    agent = LifelongAgent(args.agent_id, storage_dir=args.storage)
    print(f"\n  Lifelong Agent: {args.agent_id}")
    print(f"  Environment: {args.env_id}")
    print(f"  Episodes: {args.episodes}")
    print("=" * 55)

    agent.run_session(args.env_id, difficulty=args.difficulty, episodes=args.episodes)
    agent.save()
    agent.print_biography()


def cmd_build(args):
    """Autonomous agent builder."""
    from cognicore.agent_builder import build_agent, describe_agent

    agent = build_agent(goal=args.goal, risk_tolerance=args.risk)
    info = describe_agent(agent)

    print("\n  Agent Builder")
    print(f"  Goal: {args.goal}")
    print(f"  Risk: {args.risk}")
    print("=" * 55)
    for k, v in info.items():
        print(f"  {k:25s} {v}")
    print("=" * 55)

    # Test the agent
    env = cognicore.make(args.env_id, difficulty="easy")
    obs = env.reset()
    while True:
        action = agent.act(obs)
        obs, reward, done, _, info_d = env.step(action)
        if hasattr(agent, "learn"):
            agent.learn(reward, info_d)
        if done:
            break
    stats = env.episode_stats()
    print(f"\n  Test run: accuracy={stats.accuracy:.0%} score={env.get_score():.4f}")
    print()


def cmd_causal(args):
    """Causal reasoning analysis."""
    from cognicore.causal import CausalEngine

    engine = CausalEngine()
    env = cognicore.make(args.env_id, difficulty=args.difficulty)
    agent = RandomAgent(env.action_space)

    for ep in range(args.episodes):
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            er = info.get("eval_result", {})
            engine.observe_step(
                obs, action, er.get("correct", False), er.get("category", "")
            )
            if done:
                break

    engine.print_graph()


def cmd_report(args):
    """Generate HTML report."""
    from cognicore.report import ReportGenerator

    report = ReportGenerator(f"CogniCore Report: {args.env_id}")
    for ep in range(args.episodes):
        report.add_episode(args.env_id, args.difficulty)

    path = report.export(args.output)
    print(f"\n  Report exported: {path}")
    print(f"  Episodes: {args.episodes}")
    print("  Open in browser to view.\n")


def cmd_doctor(args):
    """Health check — verify everything works."""
    import importlib

    print(f"\nCogniCore Doctor v{cognicore.__version__}")
    print("=" * 55)

    checks = 0
    passed = 0
    failed = 0

    def check(name, fn):
        nonlocal checks, passed, failed
        checks += 1
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    # Core imports
    print("\n  Core Modules:")
    for mod_name in [
        "cognicore.core.base_env",
        "cognicore.core.types",
        "cognicore.core.spaces",
        "cognicore.middleware.memory",
        "cognicore.middleware.reflection",
        "cognicore.middleware.rewards",
        "cognicore.middleware.propose_revise",
        "cognicore.middleware.safety_monitor",
    ]:
        check(mod_name.split(".")[-1], lambda m=mod_name: importlib.import_module(m))

    # Environments
    print("\n  Environments:")
    envs = cognicore.list_envs()
    check(
        f"{len(envs)} environments registered",
        lambda: (
            None
            if len(envs) == 24
            else (_ for _ in ()).throw(Exception(f"Expected 24, got {len(envs)}"))
        ),
    )

    for env_id in ["SafetyClassification-v1", "MathReasoning-v1", "CodeDebugging-v1"]:
        check(f"make({env_id})", lambda eid=env_id: cognicore.make(eid).reset())

    # Premium modules
    print("\n  Premium Modules:")
    for mod_name in [
        "cognicore.advanced_memory",
        "cognicore.explainer",
        "cognicore.adversarial",
        "cognicore.smart_agents",
        "cognicore.auto_improve",
        "cognicore.safety_layer",
        "cognicore.cost_tracker",
    ]:
        check(mod_name.split(".")[-1], lambda m=mod_name: importlib.import_module(m))

    # Research modules
    print("\n  Research Modules:")
    for mod_name in [
        "cognicore.predictive",
        "cognicore.multi_memory",
        "cognicore.red_blue",
        "cognicore.debugger",
        "cognicore.intelligence",
        "cognicore.thought_trace",
        "cognicore.knowledge_transfer",
        "cognicore.evolution",
    ]:
        check(mod_name.split(".")[-1], lambda m=mod_name: importlib.import_module(m))

    # Platform modules
    print("\n  Platform Modules:")
    for mod_name in [
        "cognicore.persistence",
        "cognicore.report",
        "cognicore.replay",
        "cognicore.profiles",
        "cognicore.prompt_optimizer",
        "cognicore.webhooks",
        "cognicore.augmentation",
        "cognicore.fingerprint",
        "cognicore.difficulty",
        "cognicore.rate_limiter",
        "cognicore.cache",
    ]:
        check(mod_name.split(".")[-1], lambda m=mod_name: importlib.import_module(m))

    # Phase 8 modules
    print("\n  Phase 8 Modules:")
    for mod_name in [
        "cognicore.meta_rewards",
        "cognicore.causal",
        "cognicore.agent_builder",
        "cognicore.strategy",
        "cognicore.lifelong",
        "cognicore.swarm",
        "cognicore.knowledge_transfer",
        "cognicore.evolution",
    ]:
        check(mod_name.split(".")[-1], lambda m=mod_name: importlib.import_module(m))

    # Functional tests
    print("\n  Functional Tests:")
    check("RandomAgent works", lambda: _test_random_agent())
    check("AutoLearner works", lambda: _test_auto_learner())
    check("SemanticMemory works", lambda: _test_semantic_memory())
    check("FailurePredictor works", lambda: _test_predictor())
    check("CognitiveMemory works", lambda: _test_cognitive_memory())

    # Optional dependencies
    print("\n  Optional Dependencies:")
    for pkg, label in [
        ("fastapi", "FastAPI (server)"),
        ("uvicorn", "Uvicorn (server)"),
        ("openai", "OpenAI (LLM)"),
        ("pytest", "Pytest (dev)"),
    ]:
        try:
            importlib.import_module(pkg)
            check(label, lambda: None)
        except ImportError:
            print(f"  [SKIP] {label} (not installed)")

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  {passed}/{checks} checks passed, {failed} failed")
    if failed == 0:
        print("  CogniCore is healthy!")
    else:
        print(
            "  Some checks failed. Run 'pip install cognicore-env[all]' for full features."
        )
    print(f"  Exports: {len(cognicore.__all__)}")
    print(f"  Environments: {len(envs)}")
    print(f"{'=' * 55}\n")


def _test_random_agent():
    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    agent = RandomAgent(env.action_space)
    obs = env.reset()
    action = agent.act(obs)
    env.step(action)


def _test_auto_learner():
    from cognicore.smart_agents import AutoLearner

    agent = AutoLearner()
    obs = {
        "category": "test",
        "prompt": "hello",
        "memory_context": [],
        "reflection_hints": "",
    }
    action = agent.act(obs)
    assert "classification" in action


def _test_semantic_memory():
    mem = cognicore.SemanticMemory()
    mem.store({"text": "test input", "correct": True})
    assert mem.stats()["total_entries"] == 1


def _test_predictor():
    pred = cognicore.FailurePredictor()
    pred.observe("test", correct=True)
    risk = pred.predict_risk("test")
    assert "risk" in risk


def _test_cognitive_memory():
    mem = cognicore.CognitiveMemory()
    mem.perceive("test", "cat", True, "A")
    assert mem.stats()["working_memory"] == 1



# =====================================================================
# CLI: train (config-driven)
# =====================================================================


def cmd_train(args):
    """Train an agent using a YAML config file or CLI args."""
    import json
    import random

    config_path = getattr(args, "config", None)
    experiment = {}

    if config_path:
        # Load YAML config
        try:
            import yaml
            with open(config_path) as f:
                experiment = yaml.safe_load(f)
        except ImportError:
            # Fallback: simple YAML parser for basic configs
            experiment = _parse_simple_yaml(config_path)

    # Merge CLI overrides
    env_id = getattr(args, "env_id", None) or experiment.get("environment", {}).get("id", "SafetyClassification-v1")
    difficulty = getattr(args, "difficulty", None) or experiment.get("environment", {}).get("difficulty", "easy")
    episodes = getattr(args, "episodes", None) or experiment.get("experiment", {}).get("episodes", 10)
    seed = getattr(args, "seed", None) or experiment.get("experiment", {}).get("seed", 42)
    agent_type = getattr(args, "agent_type", None) or experiment.get("agent", {}).get("type", "auto_learner")
    enable_memory = experiment.get("middleware", {}).get("memory", True)
    enable_reflection = experiment.get("middleware", {}).get("reflection", True)
    verbose = getattr(args, "verbose", False) or experiment.get("output", {}).get("verbose", False)

    random.seed(seed)

    print(f"\nCogniCore Training v{cognicore.__version__}")
    print(f"{'=' * 55}")
    print(f"  Environment:  {env_id} ({difficulty})")
    print(f"  Agent:        {agent_type}")
    print(f"  Episodes:     {episodes}")
    print(f"  Seed:         {seed}")
    print(f"  Memory:       {'ON' if enable_memory else 'OFF'}")
    print(f"  Reflection:   {'ON' if enable_reflection else 'OFF'}")
    print(f"{'=' * 55}\n")

    # Create agent
    from cognicore.smart_agents import AutoLearner, SafeAgent, AdaptiveAgent
    agents_map = {
        "auto_learner": AutoLearner,
        "safe": SafeAgent,
        "adaptive": AdaptiveAgent,
        "random": RandomAgent,
    }
    agent_cls = agents_map.get(agent_type, AutoLearner)
    agent = agent_cls()

    # Create env
    config = cognicore.CogniCoreConfig(
        enable_memory=enable_memory,
        enable_reflection=enable_reflection,
    )
    env = cognicore.make(env_id, difficulty=difficulty, config=config)

    # Train
    results = []
    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if hasattr(agent, "learn"):
                agent.learn(reward, info)
            if done:
                break
        stats = env.episode_stats()
        results.append(stats.accuracy)
        if verbose:
            print(f"  Episode {ep+1:>3}: accuracy={stats.accuracy*100:.0f}% | reward={stats.total_reward:.2f} | memory={stats.memory_entries_created}")

    import statistics
    mean_acc = statistics.mean(results) * 100
    print(f"\n  Training Complete!")
    print(f"  Mean Accuracy: {mean_acc:.1f}%")
    if len(results) > 1:
        std_acc = statistics.stdev(results) * 100
        print(f"  Std Dev:       +/- {std_acc:.1f}%")
    print(f"  Memory Stats:  {env.memory.stats()['total_entries']} entries stored\n")

    # Save report if configured
    save_report = experiment.get("output", {}).get("save_report", False) or getattr(args, "save_report", False)
    if save_report:
        report_path = experiment.get("output", {}).get("report_path", "./cognicore_report.json")
        report = {
            "environment": env_id,
            "difficulty": difficulty,
            "agent": agent_type,
            "episodes": episodes,
            "seed": seed,
            "mean_accuracy": round(mean_acc, 2),
            "per_episode": results,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved to: {report_path}\n")


def _parse_simple_yaml(path):
    """Minimal YAML parser for basic key: value configs (no PyYAML needed)."""
    result = {}
    with open(path) as f:
        current_section = None
        for line in f:
            line = line.rstrip()
            if not line or line.strip().startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current_section = line[:-1].strip()
                result[current_section] = {}
            elif ":" in line and current_section:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.isdigit():
                    val = int(val)
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                result[current_section][key] = val
    return result


# =====================================================================
# CLI: demo
# =====================================================================


def cmd_demo(args):
    """Run the CogniCore demo — agent improves with memory."""
    from cognicore.smart_agents import AutoLearner
    import random
    random.seed(42)

    print(f"\nCogniCore Demo v{cognicore.__version__}")
    print("=" * 60)
    print("  Watch an agent improve by learning from its mistakes!\n")

    # Without memory
    print("  [1/2] Training WITHOUT memory...")
    config_off = cognicore.CogniCoreConfig(enable_memory=False, enable_reflection=False)
    env_off = cognicore.make("SafetyClassification-v1", config=config_off)
    agent_off = AutoLearner()
    score_off = cognicore.evaluate(agent_off, env_off, episodes=10)
    print(f"        Accuracy: {score_off*100:.1f}%\n")

    # With memory
    random.seed(42)
    print("  [2/2] Training WITH CogniCore memory...")
    config_on = cognicore.CogniCoreConfig(enable_memory=True, enable_reflection=True)
    env_on = cognicore.make("SafetyClassification-v1", config=config_on)
    agent_on = AutoLearner()
    score_on = cognicore.evaluate(agent_on, env_on, episodes=10)
    print(f"        Accuracy: {score_on*100:.1f}%")

    improvement = (score_on - score_off) * 100
    print(f"\n{'=' * 60}")
    print(f"  Result: +{improvement:.1f}% improvement with CogniCore memory!")
    print(f"  The agent remembers past mistakes and avoids repeating them.")
    print(f"{'=' * 60}\n")


# =====================================================================
# CLI: metrics
# =====================================================================


def cmd_metrics(args):
    """Show live training metrics for an environment."""
    from cognicore.smart_agents import AutoLearner
    from cognicore.utils.logging import get_logger, log_episode_end

    logger = get_logger("cognicore.metrics")

    env_id = args.env_id
    episodes = args.episodes

    env = cognicore.make(env_id, difficulty="easy")
    agent = AutoLearner()

    print(f"\nCogniCore Metrics — {env_id}")
    print("=" * 70)
    print(f"  {'Ep':>3} | {'Accuracy':>8} | {'Reward':>8} | {'Memory':>6} | {'Hints':>5} | {'Status'}")
    print("-" * 70)

    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if hasattr(agent, "learn"):
                agent.learn(reward, info)
            if done:
                break

        stats = env.episode_stats()
        status = "HEALTHY" if stats.accuracy >= 0.7 else "WARNING" if stats.accuracy >= 0.4 else "CRITICAL"

        print(f"  {ep+1:>3} | {stats.accuracy*100:>7.1f}% | {stats.total_reward:>8.2f} | {stats.memory_entries_created:>6} | {stats.reflection_hints_given:>5} | {status}")

        log_episode_end(logger, ep+1, stats.steps, stats.total_reward, stats.accuracy)

    mem_stats = env.memory.stats()
    print("-" * 70)
    print(f"  Total memory entries: {mem_stats['total_entries']}")
    print(f"  Success rate: {mem_stats['success_rate']*100:.1f}%")
    print(f"  Categories seen: {len(mem_stats['groups'])}")
    print("=" * 70 + "\n")


# =====================================================================
# Main Entry Point
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        prog="cognicore",
        description="CogniCore -- Cognitive AI Platform (71 exports, 24 environments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  cognicore list                                    List all environments
  cognicore run SafetyClassification-v1 -v          Run with verbose output
  cognicore benchmark --episodes 3                  Benchmark across all envs
  cognicore stress SafetyClassification-v1          Adversarial stress test
  cognicore battle --rounds 50                      Red vs Blue simulation
  cognicore evolve SafetyClassification-v1          Evolutionary training
  cognicore swarm --size 5                          Multi-agent swarm
  cognicore build --goal "maximize safety"          Auto-build agent
  cognicore lifelong agent-001                      Persistent agent
  cognicore report --episodes 3                     Generate HTML report
  cognicore doctor                                  Health check everything
""",
    )
    parser.add_argument(
        "--version", action="version", version=f"cognicore {cognicore.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ---- Core ----
    sub_list = subparsers.add_parser("list", help="List available environments")
    sub_list.set_defaults(func=cmd_list)

    sub_info = subparsers.add_parser("info", help="Show environment details")
    sub_info.add_argument("env_id", help="Environment ID")
    sub_info.add_argument("--difficulty", default="easy")
    sub_info.set_defaults(func=cmd_info)

    sub_run = subparsers.add_parser("run", help="Run environment with random agent")
    sub_run.add_argument("env_id", help="Environment ID")
    sub_run.add_argument("--difficulty", default="easy")
    sub_run.add_argument("--episodes", type=int, default=1)
    sub_run.add_argument("-v", "--verbose", action="store_true")
    sub_run.set_defaults(func=cmd_run)

    sub_serve = subparsers.add_parser("serve", help="Start the API server")
    sub_serve.add_argument("--host", default="127.0.0.1")
    sub_serve.add_argument("--port", type=int, default=8000)
    sub_serve.add_argument("-v", "--verbose", action="store_true")
    sub_serve.set_defaults(func=cmd_serve)

    sub_dash = subparsers.add_parser("dashboard", help="Start the web dashboard")
    sub_dash.add_argument("--host", default="127.0.0.1")
    sub_dash.add_argument("--port", type=int, default=8050)
    sub_dash.set_defaults(func=cmd_dashboard)

    sub_lb = subparsers.add_parser("leaderboard", help="Show leaderboard rankings")
    sub_lb.add_argument("--env-id", default=None, help="Filter by environment")
    sub_lb.add_argument("--top", type=int, default=20)
    sub_lb.add_argument("--storage", default="./cognicore_data")
    sub_lb.set_defaults(func=cmd_leaderboard)

    # ---- Training ----
    sub_bench = subparsers.add_parser(
        "benchmark", help="Benchmark agent across all envs"
    )
    sub_bench.add_argument(
        "--envs", default="all", help="'all' or comma-separated env IDs"
    )
    sub_bench.add_argument("--difficulties", default="easy,medium,hard")
    sub_bench.add_argument("--episodes", type=int, default=1)
    sub_bench.set_defaults(func=cmd_benchmark)

    sub_cur = subparsers.add_parser(
        "curriculum", help="Curriculum learning (auto-difficulty)"
    )
    sub_cur.add_argument("env_id", help="Environment ID")
    sub_cur.add_argument("--episodes", type=int, default=15)
    sub_cur.add_argument(
        "--threshold", type=float, default=0.8, help="Promotion threshold"
    )
    sub_cur.add_argument(
        "--demotion", type=float, default=0.3, help="Demotion threshold"
    )
    sub_cur.add_argument("--window", type=int, default=3)
    sub_cur.set_defaults(func=cmd_curriculum)

    sub_imp = subparsers.add_parser("improve", help="Self-improvement loop")
    sub_imp.add_argument("env_id", help="Environment ID")
    sub_imp.add_argument("--difficulty", default="easy")
    sub_imp.add_argument("--target", type=float, default=0.9, help="Target accuracy")
    sub_imp.add_argument("--cycles", type=int, default=10)
    sub_imp.add_argument("--episodes", type=int, default=2)
    sub_imp.add_argument("--patience", type=int, default=3)
    sub_imp.set_defaults(func=cmd_improve)

    sub_evo = subparsers.add_parser("evolve", help="Evolutionary training")
    sub_evo.add_argument("env_id", help="Environment ID")
    sub_evo.add_argument("--difficulty", default="easy")
    sub_evo.add_argument("--pop", type=int, default=10, help="Population size")
    sub_evo.add_argument("--generations", type=int, default=5)
    sub_evo.add_argument("--mutation", type=float, default=0.2)
    sub_evo.set_defaults(func=cmd_evolve)

    # ---- Testing ----
    sub_stress = subparsers.add_parser("stress", help="Adversarial stress test")
    sub_stress.add_argument("env_id", nargs="?", default="SafetyClassification-v1")
    sub_stress.add_argument("--rounds", type=int, default=10)
    sub_stress.set_defaults(func=cmd_stress)

    sub_battle = subparsers.add_parser("battle", help="Red vs Blue battle simulation")
    sub_battle.add_argument("--rounds", type=int, default=30)
    sub_battle.set_defaults(func=cmd_battle)

    sub_dbg = subparsers.add_parser("debug", help="AI debugger with breakpoints")
    sub_dbg.add_argument("env_id", nargs="?", default="SafetyClassification-v1")
    sub_dbg.add_argument("--difficulty", default="easy")
    sub_dbg.add_argument(
        "--on-wrong", action="store_true", default=True, help="Break on wrong answers"
    )
    sub_dbg.add_argument("--category", default=None, help="Break on this category")
    sub_dbg.set_defaults(func=cmd_debug)

    # ---- Analysis ----
    sub_exp = subparsers.add_parser("explain", help="Explainable AI analysis")
    sub_exp.add_argument("env_id", nargs="?", default="SafetyClassification-v1")
    sub_exp.add_argument("--difficulty", default="easy")
    sub_exp.set_defaults(func=cmd_explain)

    sub_iq = subparsers.add_parser("iq", help="Intelligence scoring (6 dimensions)")
    sub_iq.add_argument("env_id", nargs="?", default="SafetyClassification-v1")
    sub_iq.add_argument("--difficulty", default="easy")
    sub_iq.add_argument("--episodes", type=int, default=3)
    sub_iq.set_defaults(func=cmd_iq)

    sub_cost = subparsers.add_parser("cost", help="Cost estimation for LLM usage")
    sub_cost.add_argument(
        "--model",
        default="gemini-flash",
        choices=[
            "gemini-flash",
            "gemini-pro",
            "gpt-4o-mini",
            "gpt-4o",
            "claude-sonnet",
        ],
    )
    sub_cost.add_argument("--env-id", dest="env_id", default="SafetyClassification-v1")
    sub_cost.add_argument("--difficulty", default="easy")
    sub_cost.set_defaults(func=cmd_cost)

    sub_xfer = subparsers.add_parser("transfer", help="Knowledge transfer demo")
    sub_xfer.add_argument("--env-id", dest="env_id", default="SafetyClassification-v1")
    sub_xfer.add_argument("--expert-episodes", type=int, default=3)
    sub_xfer.add_argument(
        "--method", default="full", choices=["full", "successes_only", "selective"]
    )
    sub_xfer.set_defaults(func=cmd_transfer)

    # ---- Phase 8 ----
    sub_swarm = subparsers.add_parser("swarm", help="Swarm intelligence (multi-agent)")
    sub_swarm.add_argument("env_id", nargs="?", default="SafetyClassification-v1")
    sub_swarm.add_argument("--size", type=int, default=5)
    sub_swarm.add_argument("--difficulty", default="easy")
    sub_swarm.add_argument("--episodes", type=int, default=1)
    sub_swarm.set_defaults(func=cmd_swarm)

    sub_life = subparsers.add_parser("lifelong", help="Lifelong learning agent")
    sub_life.add_argument("agent_id", help="Agent identifier")
    sub_life.add_argument("--env-id", dest="env_id", default="SafetyClassification-v1")
    sub_life.add_argument("--difficulty", default="easy")
    sub_life.add_argument("--episodes", type=int, default=3)
    sub_life.add_argument("--storage", default="./cognicore_agents")
    sub_life.set_defaults(func=cmd_lifelong)

    sub_build = subparsers.add_parser("build", help="Auto-build agent from goal")
    sub_build.add_argument(
        "--goal", default="maximize accuracy", help="High-level goal"
    )
    sub_build.add_argument(
        "--risk", default="medium", choices=["low", "medium", "high"]
    )
    sub_build.add_argument("--env-id", dest="env_id", default="SafetyClassification-v1")
    sub_build.set_defaults(func=cmd_build)

    sub_report = subparsers.add_parser("report", help="Generate HTML report")
    sub_report.add_argument(
        "--env-id", dest="env_id", default="SafetyClassification-v1"
    )
    sub_report.add_argument("--difficulty", default="easy")
    sub_report.add_argument("--episodes", type=int, default=3)
    sub_report.add_argument("--output", default="cognicore_report.html")
    sub_report.set_defaults(func=cmd_report)

    # ---- New: train, demo, metrics ----
    sub_train = subparsers.add_parser("train", help="Train agent (config-driven)")
    sub_train.add_argument("config", nargs="?", default=None, help="YAML config file")
    sub_train.add_argument("--env-id", dest="env_id", default=None)
    sub_train.add_argument("--difficulty", default=None)
    sub_train.add_argument("--episodes", type=int, default=None)
    sub_train.add_argument("--seed", type=int, default=None)
    sub_train.add_argument("--agent", dest="agent_type", default=None, choices=["auto_learner", "safe", "adaptive", "random"])
    sub_train.add_argument("-v", "--verbose", action="store_true")
    sub_train.add_argument("--save-report", action="store_true")
    sub_train.set_defaults(func=cmd_train)

    sub_demo = subparsers.add_parser("demo", help="Run the CogniCore demo")
    sub_demo.set_defaults(func=cmd_demo)

    sub_metrics = subparsers.add_parser("metrics", help="Show live training metrics")
    sub_metrics.add_argument("env_id", nargs="?", default="SafetyClassification-v1")
    sub_metrics.add_argument("--episodes", type=int, default=5)
    sub_metrics.set_defaults(func=cmd_metrics)

    sub_doc = subparsers.add_parser("doctor", help="Health check everything")
    sub_doc.set_defaults(func=cmd_doctor)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
