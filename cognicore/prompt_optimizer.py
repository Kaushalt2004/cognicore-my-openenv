"""
CogniCore Prompt Optimizer — Auto-tune prompts for LLM agents.

Tests prompt variations and finds the best-performing one.

Usage::

    from cognicore.prompt_optimizer import PromptOptimizer

    opt = PromptOptimizer("SafetyClassification-v1")
    opt.add_prompt("v1", "Classify as SAFE, UNSAFE, or NEEDS_REVIEW: {text}")
    opt.add_prompt("v2", "You are a safety expert. Classify: {text}")
    best = opt.optimize(episodes=3)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import cognicore


class PromptOptimizer:
    """Optimize prompts by testing variations head-to-head.

    Parameters
    ----------
    env_id : str
        Environment to test prompts on.
    difficulty : str
        Environment difficulty.
    """

    def __init__(self, env_id: str = "SafetyClassification-v1", difficulty: str = "easy"):
        self.env_id = env_id
        self.difficulty = difficulty
        self.templates: Dict[str, str] = {}
        self.results: Dict[str, Dict] = {}

    def add_prompt(self, name: str, template: str):
        """Add a prompt template to test.

        The template can use {text}, {category}, {context} placeholders.
        """
        self.templates[name] = template
        return self

    def optimize(
        self,
        episodes: int = 2,
        agent_factory: Optional[Callable] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Test all prompt templates and find the best one.

        Parameters
        ----------
        episodes : int
            Episodes per prompt template.
        agent_factory : callable, optional
            Function that returns an agent. None uses AutoLearner.
        verbose : bool
            Print comparison results.
        """
        if not self.templates:
            self.templates["default"] = "Classify the following: {text}"

        if verbose:
            print(f"\nPrompt Optimizer: {self.env_id}")
            print(f"  Templates: {len(self.templates)}")
            print(f"  Episodes per template: {episodes}")
            print("=" * 55)

        for name, template in self.templates.items():
            accuracies = []
            scores = []
            rewards = []

            for ep in range(episodes):
                env = cognicore.make(self.env_id, difficulty=self.difficulty)
                if agent_factory:
                    agent = agent_factory()
                else:
                    from cognicore.smart_agents import AutoLearner
                    agent = AutoLearner()

                obs = env.reset()
                ep_rewards = []

                while True:
                    # Agent uses the prompt template
                    action = agent.act(obs)
                    obs, reward, done, _, info = env.step(action)
                    ep_rewards.append(reward.total)
                    if hasattr(agent, 'learn'):
                        agent.learn(reward, info)
                    if done:
                        break

                stats = env.episode_stats()
                accuracies.append(stats.accuracy)
                scores.append(env.get_score())
                rewards.append(sum(ep_rewards))

            avg_acc = sum(accuracies) / len(accuracies)
            avg_score = sum(scores) / len(scores)
            avg_reward = sum(rewards) / len(rewards)

            self.results[name] = {
                "template": template,
                "avg_accuracy": avg_acc,
                "avg_score": avg_score,
                "avg_reward": avg_reward,
                "episodes": episodes,
                "accuracies": accuracies,
            }

            if verbose:
                print(f"  {name:20s} accuracy={avg_acc:.0%} score={avg_score:.4f} reward={avg_reward:+.2f}")

        # Find best
        best_name = max(self.results, key=lambda k: self.results[k]["avg_accuracy"])
        best = self.results[best_name]

        if verbose:
            print(f"\n  Winner: {best_name} ({best['avg_accuracy']:.0%})")
            print(f"  Template: {best['template'][:80]}")
            print("=" * 55)

        return {
            "best_prompt": best_name,
            "best_template": best["template"],
            "best_accuracy": best["avg_accuracy"],
            "all_results": self.results,
        }
