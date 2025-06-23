# RL Bitcoin Trading System

This project implements a modular reinforcement learning (RL) trading system for Bitcoin (BTC) using real 1-hour historical data, technical indicators, and the PPO algorithm from stable-baselines3. The system is designed for research, experimentation, and extensibility.

---

## Architecture Overview

- **Custom Trading Environment**: `gym_env.py` implements a Gymnasium-compatible environment for trading simulation.
- **Training Script**: `train_ppo_btc.py` prepares data, calculates indicators, defines a real-data environment, and trains a PPO agent.
- **Data**: `btc_usdt_1h_binance.csv` contains 1 year of hourly BTC/USDT OHLCV data.
- **Model Output**: `ppo_btc_moderate_trader.zip` is the trained RL agent.

---

## Data Pipeline

- **btc_usdt_1h_binance.csv**: 1 year of hourly OHLCV data for BTC/USDT.
- **Indicators**: RSI, MACD, and EMA are calculated on the close price using `pandas_ta` and added as columns.

---

## Custom Environment (`gym_env.py`)

### LookBack Class
- Manages a rolling buffer of historical data (prices, indicators, etc.) using a deque.
- Can be extended to save/load data from CSV/Parquet.

### RoostooTradingEnv Class
- Inherits from `gymnasium.Env`.
- Simulates a trading environment with:
  - **State**: balance, asset holdings, prices, indicators.
  - **Action**: continuous vector per asset (buy/sell/hold).
  - **Reward**: change in portfolio value after each step.
  - **Step logic**: executes trades, updates balance/holdings, simulates price/indicator changes (in the base class).
- Designed to be extended for real data and custom indicators.

---

## Training Script (`train_ppo_btc.py`)

### Data Preparation
- Loads the CSV into a pandas DataFrame.
- Calculates RSI, MACD, and EMA using `pandas_ta`.
- Drops rows with NaN values (due to indicator lookback).

### Custom Environment for Real Data
- **BTCTradingEnv**: Inherits from `RoostooTradingEnv`.
  - Uses real price and indicator data from the DataFrame.
  - At each step, sets the current price and indicators from the DataFrame row.
  - Handles trading logic and reward calculation as before.
  - Follows the Gymnasium API: `reset()` returns `(obs, info)`, `step()` returns `(obs, reward, terminated, truncated, info)`.

### Training
- Instantiates the environment with the prepared DataFrame.
- Checks environment compliance with `check_env`.
- Trains a PPO agent (`stable-baselines3`) for 10,000 timesteps.
- Saves the trained model as `ppo_btc_moderate_trader.zip`.

---

## RL Loop
- At each step, the agent receives an observation (balance, holdings, price, indicators).
- The agent outputs an action (how much to buy/sell).
- The environment executes the trade, updates the state, and returns the new observation and reward.
- The PPO agent learns to maximize the portfolio value over time.

---

## Key Technologies
- **pandas**: Data manipulation.
- **pandas_ta**: Technical indicator calculation.
- **gymnasium**: RL environment interface.
- **stable-baselines3**: RL algorithms (PPO).
- **numpy**: Numerical operations.

---

## Output
- The trained PPO model is saved and can be used for evaluation, backtesting, or live trading simulation.

---

## Summary
You have a modular RL trading system using real BTC data, technical indicators, a custom Gymnasium environment, and PPO. The architecture is clean and extensible for further research or production use.

---

## Getting Started
1. Install dependencies from `requirements.txt`.
2. Run `train_ppo_btc.py` to train the agent.
3. Use the saved model for evaluation or further research.

---

For questions or extensions, see the code comments or open an issue.
