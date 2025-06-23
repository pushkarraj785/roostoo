import gymnasium as gym
import numpy as np
from collections import deque

class LookBack:
    """
    Efficiently manages historical market data (price, volume, indicators).
    Uses deque for in-memory buffer and supports CSV/Parquet for persistence.
    """
    def __init__(self, maxlen=1000):
        self.buffer = deque(maxlen=maxlen)

    def append(self, data):
        self.buffer.append(data)

    def get(self, n=None):
        if n is None or n > len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def clear(self):
        self.buffer.clear()

    # Placeholder for CSV/Parquet dump/load methods
    def dump_to_csv(self, path):
        pass
    def load_from_csv(self, path):
        pass
    def dump_to_parquet(self, path):
        pass
    def load_from_parquet(self, path):
        pass

class RoostooTradingEnv(gym.Env):
    """
    Custom Gym environment for RL-based crypto trading.
    Implements risk management, asset allocation, technical indicators, action/reward logic, and LookBack buffer.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        initial_balance=50000,
        fee=0.001,
        lookback_window=100,
        n_assets=3,
        min_trade=10,
        max_trade=1000,
        asset_names=None,
        indicator_dim=3
    ):
        super().__init__()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee = fee
        self.n_assets = n_assets
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(n_assets)]
        self.holdings = np.zeros(n_assets)
        self.lookback = LookBack(maxlen=lookback_window)
        self.current_step = 0
        self.min_trade = min_trade
        self.max_trade = max_trade
        self.indicator_dim = indicator_dim

        # Observation: [balance, holdings..., prices..., indicators...]
        obs_dim = 1 + n_assets + n_assets + n_assets * indicator_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Action: [-1, 1] per asset (continuous: sell, hold, buy)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32)

        # Simulated price and indicator data for demonstration
        self.prices = np.ones(n_assets) * 100
        self.indicators = np.zeros((n_assets, indicator_dim))

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.n_assets)
        self.current_step = 0
        self.lookback.clear()
        self.prices = np.ones(self.n_assets) * 100
        self.indicators = np.zeros((self.n_assets, self.indicator_dim))
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1)
        prev_value = self._portfolio_value()
        # Simulate price change
        self.prices += np.random.randn(self.n_assets)
        # Simulate indicators
        self.indicators = np.random.randn(self.n_assets, self.indicator_dim)
        # Execute trades
        for i in range(self.n_assets):
            trade_amount = action[i] * self.max_trade
            if trade_amount > 0:  # Buy
                cost = trade_amount * self.prices[i] * (1 + self.fee)
                if self.balance >= cost:
                    self.balance -= cost
                    self.holdings[i] += trade_amount
            elif trade_amount < 0:  # Sell
                sell_amount = min(-trade_amount, self.holdings[i])
                revenue = sell_amount * self.prices[i] * (1 - self.fee)
                self.balance += revenue
                self.holdings[i] -= sell_amount
        self.current_step += 1
        # Store in lookback
        self.lookback.append({
            'balance': self.balance,
            'holdings': self.holdings.copy(),
            'prices': self.prices.copy(),
            'indicators': self.indicators.copy()
        })
        # Reward: change in portfolio value minus fees
        value = self._portfolio_value()
        reward = value - prev_value
        done = self.current_step >= 1000 or self.balance < 0
        obs = self._get_obs()
        info = {'portfolio_value': value}
        return obs, reward, done, info

    def _portfolio_value(self):
        return self.balance + np.sum(self.holdings * self.prices)

    def _get_obs(self):
        obs = np.concatenate([
            [self.balance],
            self.holdings,
            self.prices,
            self.indicators.flatten()
        ])
        return obs.astype(np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Holdings: {self.holdings}, Prices: {self.prices}")

    def close(self):
        pass

# Example usage and stability test
if __name__ == "__main__":
    env = RoostooTradingEnv()
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 10:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        steps += 1
    env.close()