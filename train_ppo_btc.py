import pandas as pd
import pandas_ta as ta
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import RoostooTradingEnv
import gymnasium as gym

# Load 1-hour BTC data for 1 year
btc_df = pd.read_csv('btc_usdt_1h_binance.csv', parse_dates=['timestamp'])

# Define 'moderate trader' indicators (RSI, MACD, EMA)
btc_df['rsi'] = ta.rsi(btc_df['close'], length=14)
btc_df['macd'] = ta.macd(btc_df['close'])['MACD_12_26_9']
btc_df['ema'] = ta.ema(btc_df['close'], length=21)

# Drop NaN rows from indicator calculation
btc_df = btc_df.dropna().reset_index(drop=True)

# Custom environment using real data and indicators
class BTCTradingEnv(RoostooTradingEnv):
    def __init__(self, df, **kwargs):
        super().__init__(n_assets=1, indicator_dim=3, **kwargs)
        self.df = df
        self.data_len = len(df)

    def reset(self, *, seed=None, options=None):
        super().reset()
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.n_assets)
        self.lookback.clear()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1, 1)
        prev_value = self._portfolio_value()
        # Use real price and indicators
        row = self.df.iloc[self.current_step]
        self.prices = np.array([row['close']])
        self.indicators = np.array([[row['rsi'], row['macd'], row['ema']]])
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
        self.lookback.append({
            'balance': self.balance,
            'holdings': self.holdings.copy(),
            'prices': self.prices.copy(),
            'indicators': self.indicators.copy()
        })
        value = self._portfolio_value()
        reward = value - prev_value
        terminated = self.current_step >= self.data_len - 1 or self.balance < 0
        truncated = False
        obs = self._get_obs()
        info = {'portfolio_value': value}
        return obs, reward, terminated, truncated, info

# Instantiate and check the environment
env = BTCTradingEnv(btc_df)
check_env(env, warn=True)

# Train PPO agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save('ppo_btc_moderate_trader')

print('Training complete. Model saved as ppo_btc_moderate_trader.')
