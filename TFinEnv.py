import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse


class CustomTradingEnv(gym.Env):
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
        super(CustomTradingEnv, self).__init__()

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
        self.initial_balance = 1000.
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.held_shares = np.zeros(self.n_stocks)
        self.portfolio_weights = np.zeros(self.n_stocks)
        self.portfolio_features = np.repeat(self.portfolio_value, self.n_stocks).reshape(self.n_stocks, 1)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks,), dtype=np.float32)

        market_feature_dim =  self.n_stocks * len(self.features)
        portfolio_feature_dim = len(self.portfolio_features)
        aggregate_lookback_dim = self.n_stocks * 2
        observation_dim = market_feature_dim + portfolio_feature_dim + aggregate_lookback_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

    def reset(self):
        # Reset internal state
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.held_shares = np.zeros(self.n_stocks)
        self.portfolio_weights = np.zeros(self.n_stocks)
        self.current_step = self.lookback_window + 1
        self.done = False

        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()).iloc[
            self.current_step].values

        edge_index = self.compute_edge_index(stock_prices=current_prices)

        observation = self._get_observation()  # Ensure this is flattened in `_get_observation()`
        return observation, edge_index

    def _calculate_portfolio_features(self):
        """
        Calculate portfolio-related features such as portfolio weights, value, and held shares.

        Returns:
            portfolio_features (np.array): The portfolio features.
        """
        # Get current prices
        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()).iloc[
            self.current_step].values

        # Update portfolio value
        self.portfolio_value = np.sum(self.held_shares * current_prices)

        if self.portfolio_value > 0:
            self.portfolio_weights = (self.held_shares * current_prices) / self.portfolio_value
        else:
            self.portfolio_weights = np.zeros(self.n_stocks)
            self.portfolio_value = self.balance

        # # Create portfolio features array
        # portfolio_features = np.concatenate([
        #     self.portfolio_weights,  # Shape: (n_tickers,)
        #     [self.portfolio_value],  # Shape: (1,)
        #     self.held_shares  # Shape: (n_tickers,)
        # ])

        self.portfolio_features = np.repeat(self.portfolio_value, self.n_stocks).reshape(self.n_stocks, 1)

        return self.portfolio_features

    def compute_edge_index(self, correlation_threshold=0.8, stock_prices=None):
        """
        Compute a dynamic edge index based on stock price correlations.
        Args:
            correlation_threshold (float): Threshold to consider connections.
            stock_prices (np.array): Matrix of stock prices (shape: [n_stocks, lookback_window]).
        Returns:
            edge_index (torch.Tensor): Edge index for GATv2.
        """
        if stock_prices is None:
            raise ValueError("Stock prices data is required to compute edge index.")

        print(f'stock prices: {stock_prices}')
        # Calculate correlations between stock price series
        correlations = np.corrcoef(stock_prices)
        print(f'correlations: {correlations}')
        adjacency_matrix = (correlations >= correlation_threshold).astype(float)
        print(f'adjacency_matrix: {adjacency_matrix}')
        # Convert adjacency matrix to sparse edge_index
        edge_index, _ = dense_to_sparse(torch.tensor(adjacency_matrix))

        print(f'edge index: {edge_index}')
        return edge_index

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.array): The portfolio allocation action.

        Returns:
            observation (np.array): The next observation.
            reward (float): The reward for the step.
            done (bool): Whether the episode is complete.
            info (dict): Additional info (if any).
        """
        # Normalize the action (e.g., ensure allocations sum to 1)
        if np.sum(action) == 0:
            action = np.ones_like(action) / len(action)  # Evenly distribute if all-zero action
        else:
            action = action / np.sum(action)

        # Get current prices
        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()).iloc[self.current_step].values

        # Calculate old portfolio value
        old_portfolio_value = self.portfolio_value

        # Target portfolio value and shares
        target_portfolio_value = old_portfolio_value * action
        target_shares = target_portfolio_value / current_prices

        # Calculate shares to trade
        target_shares = target_shares.flatten()  # Ensure it's 1D
        shares_to_trade = target_shares - self.held_shares

        # Update held shares
        self.held_shares += shares_to_trade

        # If shit fucks up try commenting out this at least once
        # Update portfolio-related features
        self.portfolio_features = self._calculate_portfolio_features()

        # Calculate reward
        reward = (self.portfolio_value - old_portfolio_value) / old_portfolio_value

        # Increment time step
        self.current_step += 1

        # Check if done
        self.done = self.current_step >= len(self.data) - 1 or self.balance <= 0


        # Get observation (includes market and portfolio features)
        observation = self._get_observation()

        current_close_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()).iloc[self.current_step].T.values

        current_close_prices = np.nan_to_num(current_close_prices,
                                             nan=np.nanmean(current_close_prices, axis=1, keepdims=True))

        edge_index = self.compute_edge_index(stock_prices=current_close_prices)

        return observation, reward, self.done, edge_index, {}

    def _get_observation(self):
        # Ensure we pivot by "Ticker" to get (n_stocks, n_features)
        market_features = self.data.pivot(index="Date", columns="Ticker", values=self.features).iloc[
            self.current_step  # Get the most recent timestep
        ].unstack().values.reshape(self.n_stocks, len(self.features))  # Shape: (n_stocks, n_features)

        recent_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()).iloc[
                        self.current_step - self.lookback_window:self.current_step].values

        mean_features = recent_prices.mean(axis=0).reshape(29, 1)  # Shape: (n_stocks, 1)
        std_features = recent_prices.std(axis=0).reshape(29, 1)  # Shape: (n_stocks, 1)

        aggregate_lookback_features = np.concatenate([mean_features, std_features], axis=1)
        portfolio_features = self._calculate_portfolio_features()
        total_features = np.concatenate([market_features, aggregate_lookback_features], axis=1)

        observation = np.hstack([total_features, portfolio_features])

        return observation

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value}")
        print(f"Cash Balance: {self.balance}")
        print(f"Portfolio Weights: {self.portfolio_weights}")

    def close(self):
        """
        Clean up the environment (optional).
        """
        pass
