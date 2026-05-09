"""
CogniCore Training Logger — TensorBoard + CSV + SB3 Callback.
"""
from __future__ import annotations
import os, csv, time, json, logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cognicore.logging")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class TrainingLogger:
    """Logs training metrics to CSV and optionally TensorBoard."""

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["time", "episode", "step", "reward", "length", "success"])
        self.tb_writer = SummaryWriter(log_dir) if use_tensorboard and HAS_TB else None
        self.episodes = 0
        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.successes = 0
        self._start = time.time()

    def log_episode(self, reward: float, length: int, success: bool = False,
                    custom: Optional[Dict[str, float]] = None):
        self.episodes += 1
        self.total_steps += length
        self.episode_rewards.append(reward)
        if success:
            self.successes += 1
        self._csv_writer.writerow([
            round(time.time() - self._start, 2), self.episodes,
            self.total_steps, round(reward, 4), length, int(success),
        ])
        self._csv_file.flush()
        if self.tb_writer:
            self.tb_writer.add_scalar("reward/episode", reward, self.episodes)
            recent = self.episode_rewards[-100:]
            self.tb_writer.add_scalar("reward/mean_100",
                sum(recent) / len(recent), self.episodes)
            self.tb_writer.add_scalar("episode/length", length, self.episodes)
            if custom:
                for k, v in custom.items():
                    self.tb_writer.add_scalar(f"custom/{k}", v, self.episodes)

    @property
    def stats(self):
        r = self.episode_rewards[-100:]
        return {"episodes": self.episodes, "total_steps": self.total_steps,
                "mean_reward_100": sum(r)/len(r) if r else 0,
                "success_rate": self.successes / max(self.episodes, 1)}

    def close(self):
        self._csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()


if HAS_SB3:
    class CogniCoreCallback(BaseCallback):
        """SB3 callback logging CogniCore metrics to TensorBoard + CSV."""
        def __init__(self, log_dir="runs/cognicore", verbose=0):
            super().__init__(verbose)
            self.cc_logger = TrainingLogger(log_dir)
            self._ep_rewards: Dict[int, float] = {}
            self._ep_lengths: Dict[int, int] = {}

        def _on_step(self):
            for i, done in enumerate(self.locals.get("dones", [])):
                self._ep_rewards.setdefault(i, 0.0)
                self._ep_lengths.setdefault(i, 0)
                self._ep_rewards[i] += self.locals["rewards"][i]
                self._ep_lengths[i] += 1
                if done:
                    info = self.locals.get("infos", [{}])[i]
                    self.cc_logger.log_episode(
                        self._ep_rewards[i], self._ep_lengths[i],
                        success=info.get("event") == "goal",
                        custom={"distance": info.get("distance", -1)})
                    self._ep_rewards[i] = 0.0
                    self._ep_lengths[i] = 0
            return True

        def _on_training_end(self):
            if self.verbose:
                s = self.cc_logger.stats
                print(f"\n  CogniCore: {s['episodes']} eps, "
                      f"mean={s['mean_reward_100']:.1f}, "
                      f"success={s['success_rate']:.0%}")
            self.cc_logger.close()
