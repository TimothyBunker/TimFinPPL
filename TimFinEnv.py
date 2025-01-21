import gym
from gym import spaces
import numpy as np
import pandas as pd


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

        random_allocations = np.random.dirichlet(np.ones(self.n_stocks), size=1)[0]  # Random proportions summing to 1
        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()
        ).iloc[self.current_step].values
        self.held_shares = (self.initial_balance * random_allocations) / current_prices

        self.transaction_cost_ratio = kwargs.get("transaction_cost", 0.001)

        self.portfolio_features = np.zeros((self.n_stocks, 2))
        self.portfolio_features[:, 0] = self.portfolio_value
        self.portfolio_features[:, 1] = self.balance

        # Reward gating
        self.initial_grace_period = kwargs.get("initial_grace_period", 1)
        self.grace_period = self.initial_grace_period
        self.threshold = -0.1
        self.heavy_penalty = 1.5
        self.soft_penalty = .5

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
        self.portfolio_value = self.balance

        random_allocations = np.random.dirichlet(np.ones(self.n_stocks), size=1)[0]  # Random proportions summing to 1
        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()
        ).iloc[self.current_step].values
        self.held_shares = (self.initial_balance * random_allocations) / current_prices

        self.portfolio_features = np.zeros((self.n_stocks, 2))  # Reset portfolio features
        self.portfolio_features[:, 0] = self.portfolio_value
        self.portfolio_features[:, 1] = self.balance
        self.current_step = self.lookback_window + 1
        self.done = False
        self.grace_period = self.initial_grace_period

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
        # print(f'held_shares: {self.held_shares}')
        # print(f'current_price: {current_prices}')
        # print(f'portfolio value: {self.portfolio_value}')
        # print(f'cash balance: {self.balance}')

        # Update portfolio features
        portfolio_features = np.zeros((self.n_stocks, 2))
        portfolio_features[:, 0] = self.portfolio_value
        portfolio_features[:, 1] = self.balance

        self.portfolio_features = portfolio_features
        return self.portfolio_features

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.array): Portfolio allocation action (including cash allocation).

        Returns:
            observation (np.array): Updated observation.
            reward (float): Reward for the step.
            done (bool): Whether the episode is complete.
            info (dict): Additional info (if any).
        """
        # Normalize the action (ensure allocations sum to 1)
        # action = np.clip(action, 0, 1)
        # action = action / np.sum(action)

        # Separate cash and stock allocations
        cash_allocation = action[-1]
        stock_allocation = action[:-1]

        # Get current prices
        current_prices = self.data.pivot(index="Date", columns="Ticker", values="Close").reindex(
            columns=self.data["Ticker"].unique()
        ).iloc[self.current_step].values

        # Calculate old portfolio value
        old_portfolio_value = self.portfolio_value

        # Calculate target portfolio value and shares for stocks
        stock_portfolio_value = old_portfolio_value * (1 - cash_allocation)
        target_portfolio_value = stock_portfolio_value * stock_allocation
        target_shares = target_portfolio_value / current_prices

        # Transaction costs for re-balancing
        shares_to_trade = target_shares - self.held_shares
        transaction_costs = np.sum(
            np.abs(shares_to_trade) * current_prices * self.transaction_cost_ratio
        )

        # Update held shares directly based on target
        self.held_shares = target_shares

        # Update balance after transaction costs
        self.balance = old_portfolio_value * cash_allocation - transaction_costs

        # Update portfolio features
        self._calculate_portfolio_features()

        # Reward calculation
        scale_factor = 100
        portfolio_change = (self.portfolio_value - old_portfolio_value) / (old_portfolio_value + 1e-6)
        if portfolio_change > 0.:
            portfolio_change *= scale_factor

        transaction_penalty = transaction_costs / (old_portfolio_value + 1e-6) # * scale_factor
        reward = portfolio_change - transaction_penalty
        print(f'reward: {reward}')

        # Penalize negative cash balance
        # if portfolio_change < self.threshold:
        #     # negative_balance_penalty = -1.0 * abs(self.balance) / abs(self.threshold)
        #     reward -= self.soft_penalty
        #     self.grace_period -= 1
        #     if self.grace_period <= 0:
        #         self.done = True
        #         return self._get_observation(), reward, self.done, {}

        # else:
        #     # Reset grace period if balance recovers
        #     self.grace_period = self.initial_grace_period

        if self.balance < 0.:
            reward -= self.heavy_penalty
            self.balance = 0.
            self.done = True
            return self._get_observation(), reward, self.done, {}

        # Increment time step
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
