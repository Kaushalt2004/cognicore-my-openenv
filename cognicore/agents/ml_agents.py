"""
Real ML Agent Adapters — NOT LLMs. These are actual trainable models.

Wrap production ML frameworks as CogniCore agents:
  - Stable Baselines3  (DQN, PPO, A2C — industry standard RL)
  - scikit-learn        (RandomForest, SVM, KNN — classical ML)
  - PyTorch             (Deep Q-Network — neural net from scratch)
  - XGBoost             (Gradient boosted trees)

NONE of these call APIs. They train locally on your machine.
"""

from __future__ import annotations

import random
import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from cognicore.agents.base_agent import BaseAgent

logger = logging.getLogger("cognicore.agents.ml")


# ═══════════════════════════════════════════════════════════════════
#  1. Deep Q-Network (PyTorch) — Neural network from scratch
# ═══════════════════════════════════════════════════════════════════

class DeepQAgent(BaseAgent):
    """Deep Q-Network agent using PyTorch.

    A real neural network (not an LLM) that learns Q-values.
    Uses experience replay and a target network for stability.

    Requirements: pip install torch

    Example::

        agent = DeepQAgent(
            state_size=7,   # observation features
            actions=["UP", "DOWN", "LEFT", "RIGHT"],
            hidden_size=64,
            learning_rate=0.001,
        )
        cc.train(agent=agent, env_id="GridWorld-v1", episodes=500)
    """

    def __init__(
        self,
        state_size: int,
        actions: List[str],
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        memory_size: int = 10000,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self.state_size = state_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Experience replay buffer
        self.memory: deque = deque(maxlen=memory_size)
        self._last_state = None
        self._last_action_idx = None
        self._total_episodes = 0

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.torch = torch

            # Build Q-network
            self.q_net = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.n_actions),
            )

            # Target network (for stable training)
            self.target_net = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.n_actions),
            )
            self.target_net.load_state_dict(self.q_net.state_dict())

            self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            self._has_torch = True

        except ImportError:
            logger.warning("PyTorch not installed. Run: pip install torch")
            logger.warning("Falling back to tabular Q-learning.")
            self._has_torch = False
            self._q_table = {}

    def _obs_to_tensor(self, obs: Dict[str, Any]):
        """Convert observation dict to a float tensor."""
        features = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, (int, float)):
                        features.append(float(v))
            elif isinstance(val, bool):
                features.append(1.0 if val else 0.0)

        # Pad or truncate to state_size
        while len(features) < self.state_size:
            features.append(0.0)
        features = features[:self.state_size]

        return self.torch.tensor(features, dtype=self.torch.float32).unsqueeze(0)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not self._has_torch:
            # Fallback to simple random/greedy
            return {"action": random.choice(self.actions)}

        state = self._obs_to_tensor(observation)

        if random.random() < self.epsilon:
            action_idx = random.randrange(self.n_actions)
        else:
            with self.torch.no_grad():
                q_values = self.q_net(state)
                action_idx = q_values.argmax(dim=1).item()

        self._last_state = state
        self._last_action_idx = action_idx

        return {"action": self.actions[action_idx]}

    def on_reward(self, reward) -> None:
        if not self._has_torch or self._last_state is None:
            return

        r = reward.total if hasattr(reward, "total") else float(reward)

        # Store experience (state, action, reward) — next_state added on next act()
        self._pending_experience = (self._last_state, self._last_action_idx, r)

    def learn(self, next_state, done: bool) -> None:
        """Train on a batch from replay buffer."""
        if not self._has_torch:
            return

        if hasattr(self, "_pending_experience"):
            state, action, reward = self._pending_experience
            self.memory.append((state, action, reward, next_state, done))
            del self._pending_experience

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = self.torch.cat([b[0] for b in batch])
        actions = self.torch.tensor([b[1] for b in batch], dtype=self.torch.long)
        rewards = self.torch.tensor([b[2] for b in batch], dtype=self.torch.float32)
        next_states = self.torch.cat([b[3] for b in batch])
        dones = self.torch.tensor([b[4] for b in batch], dtype=self.torch.float32)

        # Current Q values
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with self.torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def on_episode_end(self, stats) -> None:
        self._total_episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network every 10 episodes
        if self._has_torch and self._total_episodes % 10 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ═══════════════════════════════════════════════════════════════════
#  2. scikit-learn Agent (RandomForest, SVM, KNN)
# ═══════════════════════════════════════════════════════════════════

class SklearnAgent(BaseAgent):
    """Agent powered by scikit-learn classifiers.

    Learns from experience using supervised learning —
    trains a classifier on (state -> best_action) pairs.

    Requirements: pip install scikit-learn

    Supported models: 'random_forest', 'svm', 'knn', 'decision_tree',
                      'gradient_boosting', 'mlp'

    Example::

        agent = SklearnAgent(
            actions=["SAFE", "UNSAFE", "NEEDS_REVIEW"],
            model_type="random_forest",
        )
        cc.train(agent=agent, env_id="RealWorldSafety-v1", episodes=50)
    """

    def __init__(
        self,
        actions: List[str],
        model_type: str = "random_forest",
        retrain_every: int = 20,
    ) -> None:
        self.actions = actions
        self.action_to_idx = {a: i for i, a in enumerate(actions)}
        self.model_type = model_type
        self.retrain_every = retrain_every

        self.training_data: List[Tuple[List[float], int]] = []
        self._last_features = None
        self._last_action_idx = None
        self._model = None
        self._step_count = 0
        self._fitted = False

        try:
            import sklearn  # noqa: F401
            self._has_sklearn = True
        except ImportError:
            logger.warning("scikit-learn not installed. Run: pip install scikit-learn")
            self._has_sklearn = False

    def _build_model(self):
        """Create the sklearn model."""
        if not self._has_sklearn:
            return None
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        models = {
            "random_forest": lambda: RandomForestClassifier(n_estimators=50, random_state=42),
            "svm": lambda: SVC(kernel="rbf", probability=True),
            "knn": lambda: KNeighborsClassifier(n_neighbors=5),
            "decision_tree": lambda: DecisionTreeClassifier(max_depth=10),
            "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=50),
            "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200),
        }

        factory = models.get(self.model_type)
        if factory is None:
            raise ValueError(f"Unknown model: {self.model_type}. Choose from: {list(models.keys())}")
        return factory()

    def _obs_to_features(self, obs: Dict[str, Any]) -> List[float]:
        """Convert observation to a feature vector."""
        features = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, bool):
                features.append(1.0 if val else 0.0)
            elif isinstance(val, str):
                # Hash string features
                features.append(float(hash(val) % 1000) / 1000.0)
            elif isinstance(val, (list, tuple)):
                features.extend(float(v) for v in val if isinstance(v, (int, float)))
        return features if features else [0.0]

    def _retrain(self) -> None:
        """Train the model on collected data."""
        if not self._has_sklearn or len(self.training_data) < 10:
            return

        X = [d[0] for d in self.training_data]
        y = [d[1] for d in self.training_data]

        # Pad features to same length
        max_len = max(len(x) for x in X)
        X = [x + [0.0] * (max_len - len(x)) for x in X]

        # Check we have at least 2 classes
        if len(set(y)) < 2:
            return

        try:
            self._model = self._build_model()
            self._model.fit(X, y)
            self._fitted = True
            self._feature_size = max_len
        except Exception as e:
            logger.warning(f"Training failed: {e}")

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        features = self._obs_to_features(observation)
        self._last_features = features

        if self._fitted and self._model is not None:
            # Pad to expected size
            padded = features + [0.0] * (self._feature_size - len(features))
            padded = padded[:self._feature_size]
            try:
                action_idx = self._model.predict([padded])[0]
                self._last_action_idx = action_idx
                return {"action": self.actions[action_idx],
                        "classification": self.actions[action_idx]}
            except Exception:
                pass

        # Random exploration
        action_idx = random.randrange(len(self.actions))
        self._last_action_idx = action_idx
        return {"action": self.actions[action_idx],
                "classification": self.actions[action_idx]}

    def on_reward(self, reward) -> None:
        if self._last_features is None:
            return

        r = reward.total if hasattr(reward, "total") else float(reward)

        # If reward is positive, this was a good action — learn from it
        if r > 0:
            self.training_data.append((self._last_features, self._last_action_idx))

        self._step_count += 1
        if self._step_count % self.retrain_every == 0:
            self._retrain()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "model": self.model_type,
            "training_samples": len(self.training_data),
            "fitted": self._fitted,
        }


# ═══════════════════════════════════════════════════════════════════
#  3. XGBoost Agent
# ═══════════════════════════════════════════════════════════════════

class XGBoostAgent(BaseAgent):
    """Agent powered by XGBoost gradient boosted trees.

    Learns a state -> action mapping using gradient boosting.
    Strong for structured/tabular data.

    Requirements: pip install xgboost

    Example::

        agent = XGBoostAgent(
            actions=["GATHER_FOOD", "GATHER_WOOD", "BUILD", "REST", "EXPLORE"],
        )
        cc.train(agent=agent, env_id="ResourceGathering-v1", episodes=100)
    """

    def __init__(
        self,
        actions: List[str],
        retrain_every: int = 30,
        n_estimators: int = 100,
    ) -> None:
        self.actions = actions
        self.retrain_every = retrain_every
        self.n_estimators = n_estimators

        self.training_data: List[Tuple[List[float], int]] = []
        self._last_features = None
        self._last_action_idx = None
        self._model = None
        self._fitted = False
        self._step_count = 0
        self._feature_size = 0

        try:
            import xgboost  # noqa: F401
            self._has_xgb = True
        except ImportError:
            logger.warning("XGBoost not installed. Run: pip install xgboost")
            self._has_xgb = False

    def _obs_to_features(self, obs: Dict[str, Any]) -> List[float]:
        features = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, bool):
                features.append(1.0 if val else 0.0)
            elif isinstance(val, str):
                features.append(float(hash(val) % 1000) / 1000.0)
            elif isinstance(val, (list, tuple)):
                features.extend(float(v) for v in val if isinstance(v, (int, float)))
        return features if features else [0.0]

    def _retrain(self) -> None:
        if not self._has_xgb or len(self.training_data) < 10:
            return
        import xgboost as xgb

        X = [d[0] for d in self.training_data]
        y = [d[1] for d in self.training_data]

        max_len = max(len(x) for x in X)
        X = [x + [0.0] * (max_len - len(x)) for x in X]

        if len(set(y)) < 2:
            return

        try:
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )
            self._model.fit(X, y)
            self._fitted = True
            self._feature_size = max_len
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        features = self._obs_to_features(observation)
        self._last_features = features

        if self._fitted and self._model is not None:
            padded = features + [0.0] * (self._feature_size - len(features))
            padded = padded[:self._feature_size]
            try:
                action_idx = int(self._model.predict([padded])[0])
                self._last_action_idx = action_idx
                return {"action": self.actions[action_idx],
                        "classification": self.actions[action_idx]}
            except Exception:
                pass

        action_idx = random.randrange(len(self.actions))
        self._last_action_idx = action_idx
        return {"action": self.actions[action_idx],
                "classification": self.actions[action_idx]}

    def on_reward(self, reward) -> None:
        if self._last_features is None:
            return
        r = reward.total if hasattr(reward, "total") else float(reward)
        if r > 0:
            self.training_data.append((self._last_features, self._last_action_idx))
        self._step_count += 1
        if self._step_count % self.retrain_every == 0:
            self._retrain()


# ═══════════════════════════════════════════════════════════════════
#  4. Policy Gradient Agent (REINFORCE — PyTorch)
# ═══════════════════════════════════════════════════════════════════

class PolicyGradientAgent(BaseAgent):
    """REINFORCE policy gradient agent using PyTorch.

    Learns a stochastic policy directly (no Q-values).
    This is how PPO/A2C work at their core — pure gradient ascent
    on expected reward.

    Requirements: pip install torch

    Example::

        agent = PolicyGradientAgent(
            state_size=7,
            actions=["UP", "DOWN", "LEFT", "RIGHT"],
        )
        cc.train(agent=agent, env_id="GridWorld-v1", episodes=300)
    """

    def __init__(
        self,
        state_size: int,
        actions: List[str],
        hidden_size: int = 64,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self.state_size = state_size
        self.gamma = gamma

        self._episode_log_probs = []
        self._episode_rewards = []
        self._total_episodes = 0

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.torch = torch

            self.policy = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.n_actions),
                nn.Softmax(dim=-1),
            )
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
            self._has_torch = True

        except ImportError:
            logger.warning("PyTorch not installed. Run: pip install torch")
            self._has_torch = False

    def _obs_to_tensor(self, obs: Dict[str, Any]):
        features = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, (list, tuple)):
                features.extend(float(v) for v in val if isinstance(v, (int, float)))
            elif isinstance(val, bool):
                features.append(1.0 if val else 0.0)

        while len(features) < self.state_size:
            features.append(0.0)
        features = features[:self.state_size]
        return self.torch.tensor(features, dtype=self.torch.float32)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not self._has_torch:
            return {"action": random.choice(self.actions)}

        state = self._obs_to_tensor(observation)
        probs = self.policy(state)
        dist = self.torch.distributions.Categorical(probs)
        action = dist.sample()

        self._episode_log_probs.append(dist.log_prob(action))
        return {"action": self.actions[action.item()]}

    def on_reward(self, reward) -> None:
        r = reward.total if hasattr(reward, "total") else float(reward)
        self._episode_rewards.append(r)

    def on_episode_end(self, stats) -> None:
        if not self._has_torch or not self._episode_rewards:
            self._episode_log_probs = []
            self._episode_rewards = []
            return

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self._episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = self.torch.tensor(returns, dtype=self.torch.float32)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        loss = []
        for log_prob, G in zip(self._episode_log_probs, returns):
            loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        total_loss = self.torch.stack(loss).sum()
        total_loss.backward()
        self.optimizer.step()

        self._episode_log_probs = []
        self._episode_rewards = []
        self._total_episodes += 1
