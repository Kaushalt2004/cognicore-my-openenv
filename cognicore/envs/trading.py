"""
TradingEnv — Financial decision-making under uncertainty.

NOT text classification. Real sequential decisions:
  - BUY, SELL, HOLD across multiple assets
  - Portfolio management with transaction costs
  - Risk management (drawdown limits, position sizing)
  - Market regime detection (trending vs mean-reverting)

Memory helps: "Last time volatility spiked, HOLD was better than BUY"
"""

from __future__ import annotations
import random
import math
import logging
from typing import Any, Dict, List

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult
from cognicore.core.spaces import DiscreteSpace, DictSpace

logger = logging.getLogger("cognicore.envs.trading")


class TradingEnv(CogniCoreEnv):
    """Simulated trading environment with realistic market dynamics.

    The agent manages a portfolio, making BUY/SELL/HOLD decisions
    each step. Features synthetic price data with trends,
    volatility regimes, and mean reversion.

    Difficulty:
      easy:   1 asset, low volatility, clear trends
      medium: 3 assets, moderate volatility, regime changes
      hard:   5 assets, high volatility, fat tails, transaction costs
    """

    def _setup(self, **kwargs: Any) -> None:
        self.difficulty = kwargs.get("difficulty", "easy")
        configs = {
            "easy":   {"assets": 1, "volatility": 0.02, "trend": 0.001,
                       "steps": 100, "tx_cost": 0.0},
            "medium": {"assets": 3, "volatility": 0.03, "trend": 0.0005,
                       "steps": 200, "tx_cost": 0.001},
            "hard":   {"assets": 5, "volatility": 0.05, "trend": 0.0,
                       "steps": 300, "tx_cost": 0.002},
        }
        cfg = configs.get(self.difficulty, configs["easy"])

        self.n_assets = cfg["assets"]
        self.base_volatility = cfg["volatility"]
        self.trend = cfg["trend"]
        self.tx_cost = cfg["tx_cost"]
        self.num_tasks = kwargs.get("num_tasks", cfg["steps"])

        actions = ["HOLD", "BUY", "SELL"]
        self.action_space = DiscreteSpace(n=3, labels=actions)
        self.observation_space = DictSpace(fields={
            "prices": "Current asset prices",
            "returns": "Recent returns",
            "portfolio_value": "Current portfolio value",
            "cash": "Available cash",
            "positions": "Current holdings",
            "volatility": "Current market volatility",
            "step": "Current timestep",
        })

        self.prices: List[float] = []
        self.returns_history: List[List[float]] = []
        self.cash = 10000.0
        self.positions: List[int] = []
        self.initial_value = 10000.0
        self.peak_value = 10000.0
        self.volatility_regime = "low"

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        self.prices = [100.0] * self.n_assets
        self.returns_history = []
        self.cash = 10000.0
        self.positions = [0] * self.n_assets
        self.initial_value = 10000.0
        self.peak_value = 10000.0
        self.volatility_regime = random.choice(["low", "medium", "high"])
        return [{"step": i} for i in range(self.num_tasks)]

    def _portfolio_value(self) -> float:
        return self.cash + sum(p * q for p, q in zip(self.prices, self.positions))

    def _simulate_market(self) -> List[float]:
        """Generate next price move with regime-dependent dynamics."""
        vol_mult = {"low": 0.5, "medium": 1.0, "high": 2.0}
        vol = self.base_volatility * vol_mult.get(self.volatility_regime, 1.0)

        returns = []
        for i in range(self.n_assets):
            # Trend + noise + occasional jumps
            noise = random.gauss(0, vol)
            jump = 0
            if random.random() < 0.02:  # 2% chance of jump
                jump = random.gauss(0, vol * 5)
            ret = self.trend + noise + jump
            returns.append(ret)
            self.prices[i] *= (1 + ret)
            self.prices[i] = max(1.0, self.prices[i])

        # Regime switching
        if random.random() < 0.05:
            self.volatility_regime = random.choice(["low", "medium", "high"])

        self.returns_history.append(returns)
        return returns

    def _get_obs(self) -> Dict[str, Any]:
        recent_returns = self.returns_history[-5:] if self.returns_history else [[0.0] * self.n_assets]
        avg_returns = [sum(r[i] for r in recent_returns) / len(recent_returns)
                       for i in range(self.n_assets)]

        return {
            "prices": [round(p, 2) for p in self.prices],
            "recent_returns": [round(r, 4) for r in avg_returns],
            "portfolio_value": round(self._portfolio_value(), 2),
            "cash": round(self.cash, 2),
            "positions": list(self.positions),
            "volatility_regime": self.volatility_regime,
            "step": self._current_step,
            "max_steps": self.num_tasks,
            "drawdown": round(1 - self._portfolio_value() / self.peak_value, 4),
        }

    def _evaluate(self, action: Dict[str, Any]) -> EvalResult:
        move = action.get("action", "HOLD")
        if isinstance(move, int):
            move = ["HOLD", "BUY", "SELL"][move % 3]
        move = str(move).upper()

        old_value = self._portfolio_value()

        # Execute trade on first asset (or specified asset)
        asset_idx = min(int(action.get("asset", 0)), self.n_assets - 1)

        if move == "BUY" and self.cash >= self.prices[asset_idx]:
            qty = int(self.cash * 0.3 / self.prices[asset_idx])  # Buy 30% of cash
            if qty > 0:
                cost = qty * self.prices[asset_idx] * (1 + self.tx_cost)
                if cost <= self.cash:
                    self.positions[asset_idx] += qty
                    self.cash -= cost

        elif move == "SELL" and self.positions[asset_idx] > 0:
            qty = max(1, self.positions[asset_idx] // 2)  # Sell half
            revenue = qty * self.prices[asset_idx] * (1 - self.tx_cost)
            self.positions[asset_idx] -= qty
            self.cash += revenue

        # Simulate market AFTER trade
        self._simulate_market()
        new_value = self._portfolio_value()
        self.peak_value = max(self.peak_value, new_value)

        # Reward = portfolio return
        pnl = new_value - old_value
        pnl_pct = pnl / old_value if old_value > 0 else 0

        # Risk-adjusted score
        drawdown = 1 - new_value / self.peak_value
        if drawdown > 0.2:
            score = max(0, 0.1 - drawdown)  # Heavy penalty for large drawdowns
        elif pnl_pct > 0:
            score = min(1.0, 0.5 + pnl_pct * 10)
        else:
            score = max(0, 0.3 + pnl_pct * 5)

        if self._current_step >= self.num_tasks - 1:
            # Final score based on total return
            total_return = (new_value - self.initial_value) / self.initial_value
            score = min(1.0, max(0, 0.5 + total_return))

        return EvalResult(
            base_score=score, correct=pnl > 0, category="trading",
            ground_truth="PROFIT", predicted=move,
            metadata={"event": "trade", "pnl": round(pnl, 2),
                      "portfolio": round(new_value, 2),
                      "regime": self.volatility_regime,
                      "drawdown": round(drawdown, 4)},
        )
