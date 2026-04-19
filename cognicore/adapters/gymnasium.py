"""
CogniCore Gymnasium Adapter — Use CogniCore environments with Gymnasium.

Wraps any CogniCoreEnv as a ``gymnasium.Env`` so it works with:
  - Stable Baselines 3
  - RLlib
  - Any Gymnasium-compatible training framework

Usage::

    from cognicore.adapters.gymnasium import GymnasiumAdapter
    import cognicore

    # Wrap any CogniCore env
    gym_env = GymnasiumAdapter(cognicore.make("MathReasoning-v1"))

    # Standard Gymnasium API
    obs, info = gym_env.reset()
    obs, reward, terminated, truncated, info = gym_env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from cognicore.core.base_env import CogniCoreEnv


class GymnasiumAdapter:
    """Adapter that wraps a CogniCoreEnv as a Gymnasium-compatible env.

    The adapter maps CogniCore's dict-based API to Gymnasium's standard
    interface, converting ``StructuredReward`` to a float for RL
    compatibility while preserving the full reward in ``info``.

    Parameters
    ----------
    env : CogniCoreEnv
        The CogniCore environment to wrap.
    reward_mode : str
        How to convert StructuredReward to float:
        - ``"total"`` (default): use ``reward.total``
        - ``"base"`` : use only ``reward.base_score``
        - ``"shaped"`` : use total minus time_decay (for RL stability)
    """

    # Gymnasium metadata
    metadata = {"render_modes": []}

    def __init__(
        self,
        env: CogniCoreEnv,
        reward_mode: str = "total",
    ) -> None:
        self.env = env
        self.reward_mode = reward_mode

        # Expose CogniCore spaces (compatible with Gymnasium's Dict/Discrete)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment (Gymnasium signature).

        Returns
        -------
        tuple
            ``(observation, info)``
        """
        kwargs = options or {}
        obs = self.env.reset(seed=seed, **kwargs)
        info = {
            "episode": self.env._episode_count,
            "max_steps": self.env._max_steps,
        }
        return obs, info

    def step(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take one step (Gymnasium signature).

        Returns
        -------
        tuple
            ``(observation, reward_float, terminated, truncated, info)``
            where ``info["structured_reward"]`` contains the full
            8-component reward.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert structured reward to float
        if self.reward_mode == "base":
            reward_float = reward.base_score
        elif self.reward_mode == "shaped":
            reward_float = reward.total - reward.time_decay
        else:  # "total"
            reward_float = reward.total

        # Preserve full reward in info
        info["structured_reward"] = reward.to_dict()

        return obs, reward_float, terminated, truncated, info

    def render(self) -> None:
        """Render (no-op for text-based environments)."""
        pass

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    # --- Convenience: expose CogniCore-exclusive methods ---

    def propose(self, action: Dict[str, Any]):
        """Proxy to CogniCore's propose() method."""
        return self.env.propose(action)

    def revise(self, action: Dict[str, Any]):
        """Proxy to CogniCore's revise() method."""
        obs, reward, terminated, truncated, info = self.env.revise(action)
        reward_float = reward.total
        info["structured_reward"] = reward.to_dict()
        return obs, reward_float, terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        """Proxy to CogniCore's state() method."""
        return self.env.state()

    @property
    def unwrapped(self) -> CogniCoreEnv:
        """Access the underlying CogniCore environment."""
        return self.env

    def __repr__(self) -> str:
        return f"GymnasiumAdapter({self.env!r})"
