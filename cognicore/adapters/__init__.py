"""
CogniCore Adapters — Plug into real AI frameworks.

Each adapter wraps a specific framework's agent type and injects
CogniCore Runtime cognition (memory, reflection, adaptation).

Supported:
  - OpenAI API (chat completions)
  - LangChain (chains/agents)
  - CrewAI (crews)
  - AutoGen (conversable agents)
  - Stable-Baselines3 (RL policies)
  - Custom Python callables
"""

from cognicore.runtime import CogniCoreRuntime, RuntimeConfig, ExecutionResult

from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger("cognicore.adapters")


# ---------------------------------------------------------------------------
# OpenAI Adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """Wrap OpenAI API calls with CogniCore cognition.

    Usage:
        from openai import OpenAI
        client = OpenAI()
        adapter = OpenAIAdapter(client)
        result = adapter.chat("Fix this Python bug: ...", category="code_fix")
    """

    def __init__(self, client=None, model: str = "gpt-4o-mini",
                 runtime: Optional[CogniCoreRuntime] = None):
        self.client = client
        self.model = model
        self.runtime = runtime or CogniCoreRuntime(name="openai-adapter")

    def chat(self, prompt: str, category: str = "chat",
             system: str = "", evaluator: Optional[Callable] = None,
             max_retries: int = 1, **kwargs) -> ExecutionResult:
        """Send a chat completion with CogniCore cognition."""

        def agent_fn(task, context, **kw):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})

            # Inject cognition context into system prompt
            cogni_ctx = ""
            if context.get("failures_to_avoid"):
                cogni_ctx += f"\nAvoid these approaches (they failed before): {context['failures_to_avoid']}"
            if context.get("successful_patterns"):
                cogni_ctx += f"\nThese approaches worked before: {context['successful_patterns']}"
            if context.get("reflection_hint"):
                cogni_ctx += f"\n{context['reflection_hint']}"

            if cogni_ctx:
                messages.append({"role": "system",
                                "content": f"[CogniCore Context]{cogni_ctx}"})

            messages.append({"role": "user", "content": task})

            if self.client is None:
                raise RuntimeError("OpenAI client not provided. "
                                   "Install openai and pass client=OpenAI()")

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **kw
            )
            return response.choices[0].message.content

        return self.runtime.execute(
            agent_fn=agent_fn, task=prompt, category=category,
            evaluator=evaluator, max_retries=max_retries,
        )


# ---------------------------------------------------------------------------
# LangChain Adapter
# ---------------------------------------------------------------------------

class LangChainAdapter:
    """Wrap LangChain chains/agents with CogniCore cognition.

    Usage:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        llm = ChatOpenAI()
        adapter = LangChainAdapter(llm)
        result = adapter.invoke("Summarize this doc", category="summarize")
    """

    def __init__(self, chain=None, runtime: Optional[CogniCoreRuntime] = None):
        self.chain = chain
        self.runtime = runtime or CogniCoreRuntime(name="langchain-adapter")

    def invoke(self, input_text: str, category: str = "langchain",
               evaluator: Optional[Callable] = None,
               max_retries: int = 1, **kwargs) -> ExecutionResult:
        """Invoke a LangChain chain with CogniCore cognition."""

        def agent_fn(task, context, **kw):
            if self.chain is None:
                raise RuntimeError("LangChain chain not provided.")

            # Inject context into input
            enhanced_input = task
            if context.get("reflection_hint"):
                enhanced_input = f"{context['reflection_hint']}\n\n{task}"
            if context.get("failures_to_avoid"):
                enhanced_input += f"\n\n[Avoid: {', '.join(context['failures_to_avoid'][:3])}]"

            # Try .invoke() first (LCEL), fall back to __call__
            if hasattr(self.chain, 'invoke'):
                return self.chain.invoke(enhanced_input, **kw)
            else:
                return self.chain(enhanced_input, **kw)

        return self.runtime.execute(
            agent_fn=agent_fn, task=input_text, category=category,
            evaluator=evaluator, max_retries=max_retries,
        )


# ---------------------------------------------------------------------------
# Stable-Baselines3 Adapter
# ---------------------------------------------------------------------------

class SB3Adapter:
    """Wrap Stable-Baselines3 policies with CogniCore cognition.

    Adds episodic memory across training runs and failure pattern detection.

    Usage:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", "CartPole-v1")
        adapter = SB3Adapter(model, env)
        adapter.train_with_cognition(total_timesteps=10000)
    """

    def __init__(self, model=None, env=None,
                 runtime: Optional[CogniCoreRuntime] = None):
        self.model = model
        self.env = env
        self.runtime = runtime or CogniCoreRuntime(name="sb3-adapter")
        self._episode_rewards: List[float] = []
        self._episode_count = 0

    def predict_with_cognition(self, obs, category: str = "rl_step",
                                deterministic: bool = True) -> ExecutionResult:
        """Get action with CogniCore cognition context."""

        def agent_fn(task, context, **kw):
            if self.model is None:
                raise RuntimeError("SB3 model not provided.")
            action, _states = self.model.predict(task, deterministic=deterministic)
            return action

        return self.runtime.execute(
            agent_fn=agent_fn, task=obs, category=category,
        )

    def train_with_cognition(self, total_timesteps: int = 10000,
                              eval_interval: int = 1000,
                              category: str = "rl_training"):
        """Train SB3 model with CogniCore episode tracking."""
        if self.model is None or self.env is None:
            raise RuntimeError("SB3 model and env must be provided.")

        logger.info(f"Training {total_timesteps} timesteps with CogniCore cognition")

        for step in range(0, total_timesteps, eval_interval):
            self.model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
            self._episode_count += 1

            # Evaluate
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            ep_reward = 0
            done = False
            steps = 0
            while not done and steps < 1000:
                action, _ = self.model.predict(obs, deterministic=True)
                result = self.env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                ep_reward += reward
                steps += 1

            self._episode_rewards.append(ep_reward)
            success = ep_reward > 0

            # Store in CogniCore memory
            self.runtime.memory.store({
                "category": category,
                "predicted": f"ep_{self._episode_count}",
                "correct": success,
                "reward": ep_reward,
                "steps": steps,
                "timestep": step + eval_interval,
            })

            # Get reflection
            hint = self.runtime.reflection.get_hint(category)
            if hint:
                logger.info(f"  Reflection: {hint}")

            logger.info(f"  Episode {self._episode_count}: "
                       f"reward={ep_reward:.2f}, steps={steps}")

        return {
            "episodes": self._episode_count,
            "rewards": self._episode_rewards,
            "stats": self.runtime.get_stats(),
        }


# ---------------------------------------------------------------------------
# Generic Callable Adapter
# ---------------------------------------------------------------------------

class CallableAdapter:
    """Wrap any Python callable with CogniCore cognition.

    This is the simplest adapter — works with any function.

    Usage:
        def my_solver(task, context):
            return solve(task)

        adapter = CallableAdapter(my_solver)
        result = adapter.run("problem X", category="solving")
    """

    def __init__(self, fn: Callable, runtime: Optional[CogniCoreRuntime] = None):
        self.fn = fn
        self.runtime = runtime or CogniCoreRuntime(name="callable-adapter")

    def run(self, task: Any, category: str = "default",
            evaluator: Optional[Callable] = None,
            max_retries: int = 0, **kwargs) -> ExecutionResult:
        return self.runtime.execute(
            agent_fn=self.fn, task=task, category=category,
            evaluator=evaluator, max_retries=max_retries, **kwargs
        )
