import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse


class TimTradingEnv(gym.Env):
    """
    A custom trading environment template for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): The market data.
            **kwargs: Additional arguments for customization.
        """
        super(TimTradingEnv, self).__init__()

        # Load and preprocess data
        self.data = data  # Replace with your dataset

        # Define other variables
        self.lookback_window = kwargs.get("lookback_window", 50)  # Placeholder for lookback window size
        self.done = False
        self.current_step = self.lookback_window + 1

        self.n_stocks = data["Ticker"].nunique()
        self.features = ["High", "Low", "Open", "Close", "Volume", "MA50",
                         "RSI", "MACD", "Bollinger_High", "Bollinger_Low",
                         "OBV", "VWAP", "ATR", "Stochastic_%K", "Williams_%R",
                         "EMA50", "ADX", "Log_Returns", "Pct_Change"]

        # Set initial balance and portfolio
        self.initial_balance = kwargs.get("initial_balance", 1000.)
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.held_shares = np.zeros(self.n_stocks)
        self.transaction_cost_ratio = kwargs.get("transaction_cost", 0.001)

        self.portfolio_features = np.zeros((self.n_stocks, 2))
        self.portfolio_features[:, 0] = self.portfolio_value
        self.portfolio_features[:, 1] = self.balance

        # Reward gating
        self.grace_period = 5
        self.threshold = -100.0
        self.heavy_penalty = 10.0
        self.soft_penalty = 5.0

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)
        market_feature_dim =  self.n_stocks * len(self.features)
        portfolio_feature_dim = self.n_stocks * 2
        aggregate_lookback_dim = self.n_stocks * 2
        observation_dim = market_feature_dim + portfolio_feature_dim + aggregate_lookback_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)


    def get_aggregate_features(self):
        recent_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()).iloc[
                        self.current_step - self.lookback_window:self.current_step].values

        mean_features = recent_prices.mean(axis=0).reshape(29, 1)  # Shape: (n_stocks, 1)
        std_features = recent_prices.std(axis=0).reshape(29, 1)  # Shape: (n_stocks, 1)

        aggregate_lookback_features = np.concatenate([mean_features, std_features], axis=1)

        return aggregate_lookback_features

    def reset(self):
        # Reset internal state
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.held_shares = np.zeros(self.n_stocks)
        self.portfolio_features = np.zeros((self.n_stocks, 2))  # Reset portfolio features
        self.portfolio_features[:, 0] = self.portfolio_value
        self.portfolio_features[:, 1] = self.balance
        self.current_step = self.lookback_window + 1
        self.done = False
        self.grace_period = 5  # Reset grace period if applicable

        # Get initial observation
        observation = self._get_observation()
        return observation

    def _calculate_portfolio_features(self):
        """
        Calculate portfolio-related features such as portfolio value and cash balance.

        Returns:
            portfolio_features (np.array): Array of shape (n_stocks, 2) with columns:
                - Column 0: Portfolio value replicated for each stock.
                - Column 1: Cash balance replicated for each stock.
        """
        # Get current prices
        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()
        ).iloc[self.current_step].values

        # Update portfolio value (sum of held shares and cash balance)
        self.portfolio_value = np.sum(self.held_shares * current_prices) + self.balance

        # Update portfolio features
        portfolio_features = np.zeros((self.n_stocks, 2))
        portfolio_features[:, 0] = self.portfolio_value
        portfolio_features[:, 1] = self.balance

        self.portfolio_features = portfolio_features
        return self.portfolio_features

    def step(self, action):
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)
        else:
            action = action / np.sum(action)

        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()
        ).iloc[self.current_step].values

        old_portfolio_value = self.portfolio_value
        target_portfolio_value = old_portfolio_value * action
        target_shares = target_portfolio_value / current_prices

        shares_to_trade = target_shares - self.held_shares
        transaction_costs = np.sum(np.abs(shares_to_trade) * current_prices * self.transaction_cost_ratio)

        self.held_shares += shares_to_trade
        self.balance -= transaction_costs
        self._calculate_portfolio_features()

        portfolio_change = (self.portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-6)
        transaction_penalty = transaction_costs / (old_portfolio_value + 1e-6)
        reward = portfolio_change - transaction_penalty

        if self.balance < self.threshold:
            negative_balance_penalty = -1.0 * abs(self.balance) / abs(self.threshold)
            reward += negative_balance_penalty
            self.grace_period -= 1
            if self.grace_period <= 0:
                self.done = True
        else:
            if self.grace_period < 5:
                self.grace_period += 1

        self.current_step += 1
        self.done = self.done or (self.current_step >= len(self.data["Date"].unique()) - 1)

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        # Ensure we pivot by "Ticker" to get (n_stocks, n_features)
        market_features = self.data.pivot(index="Date", columns="Ticker", values=self.features).iloc[
            self.current_step  # Get the most recent timestep
        ].unstack().values.reshape(self.n_stocks, len(self.features))

        recent_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()).iloc[
                        self.current_step - self.lookback_window:self.current_step].values

        mean_features = recent_prices.mean(axis=0).reshape(self.n_stocks, 1)
        std_features = recent_prices.std(axis=0).reshape(self.n_stocks, 1)

        aggregate_lookback_features = np.concatenate([mean_features, std_features], axis=1)
        portfolio_features = self._calculate_portfolio_features()
        total_features = np.concatenate([market_features, aggregate_lookback_features], axis=1)

        observation = np.hstack([total_features, portfolio_features])

        return observation

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value}")
        print(f"Cash Balance: {self.balance}")
        print(f"Held Shares: {self.held_shares}")

    def close(self):
        """
        Clean up the environment (optional).
        """
        pass
