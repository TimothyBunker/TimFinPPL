import gym
from gym import spaces
import numpy as np

class WindowedTradingEnv(gym.Env):
    """
    Trading environment that consumes precomputed time-series windows.
    Supports single-agent portfolio management over sliding windows.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data_windows: np.ndarray,
        price_index: int = 0,
        initial_balance: float = 1000.0,
        transaction_cost: float = 0.001,
        risk_config: dict = None
    ):
        """
        Args:
            data_windows (np.ndarray): Array of shape (T, n_assets, lookback, n_features).
            price_index (int): Index of the price feature within the last dimension.
            initial_balance (float): Starting cash balance.
            transaction_cost (float): Cost ratio per trade.
        """
        super(WindowedTradingEnv, self).__init__()
        assert data_windows.ndim == 4, (
            f"data_windows must be 4D, got {data_windows.ndim}D"
        )
        self.data_windows = data_windows
        self.price_index = price_index
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        # Risk penalty configuration
        rc = risk_config or {}
        self.volatility_window = rc.get('volatility_window', None)
        self.volatility_penalty = rc.get('volatility_penalty', 0.0)
        self.drawdown_penalty = rc.get('drawdown_penalty', 0.0)
        self.turnover_penalty = rc.get('turnover_penalty', 0.0)
        # Dimensions
        # Time dimension, number of assets, lookback window, features per asset
        self.n_steps, self.n_assets, self.lookback, self.n_features = data_windows.shape
        # Alias for compatibility
        self.n_stocks = self.n_assets

        # Portfolio state
        self.balance = self.initial_balance
        self.held_shares = np.zeros(self.n_assets, dtype=float)
        self.portfolio_value = self.initial_balance
        self.old_portfolio_change = 0.0

        # Define action and observation spaces
        # Observation: (n_assets, lookback, n_features+2)
        obs_shape = (self.n_assets, self.lookback, self.n_features + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        # Action: allocations for each asset and cash (sum to 1)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        self.current_step = 0
        # Track history for risk metrics
        self.return_history = []
        self.portfolio_history = [self.initial_balance]

    def reset(self):
        """Reset environment and return initial observation."""
        # Reset portfolio state
        self.balance = self.initial_balance
        self.held_shares[:] = 0.0
        self.portfolio_value = self.initial_balance
        self.old_portfolio_change = 0.0
        # Randomize starting index for each episode to improve exploration
        self.current_step = np.random.randint(0, self.n_steps)
        return self._get_observation()

    def _get_observation(self):
        """Get current observation combining market window and portfolio features."""
        window = self.data_windows[self.current_step]  # shape: (n_assets, lookback, n_features)
        # Normalize market features per asset and feature
        # Avoid scaling portfolio features (handled separately)
        mean_feat = window.mean(axis=1, keepdims=True)
        std_feat = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mean_feat) / std_feat
        # Portfolio features: normalized ratios to initial balance
        val_ratio = self.portfolio_value / (self.initial_balance + 1e-8)
        cash_ratio = self.balance / (self.initial_balance + 1e-8)
        pf = np.stack([
            np.full((self.n_assets, self.lookback), val_ratio),
            np.full((self.n_assets, self.lookback), cash_ratio)
        ], axis=2)
        obs = np.concatenate([window, pf], axis=2)
        return obs.astype(np.float32)

    def step(self, action):
        """Execute a step: update portfolio, compute reward, advance step."""
        # Normalize allocations to sum to 1
        alloc = np.clip(action, 0.0, 1.0)
        alloc = alloc / (alloc.sum() + 1e-8)
        cash_weight = alloc[-1]
        stock_weights = alloc[:-1]

        # Current prices from the last lookback row
        prices = self.data_windows[self.current_step][:, -1, self.price_index]

        old_value = self.portfolio_value

        # Target shares based on allocations
        total_val = old_value
        stock_val = total_val * (1.0 - cash_weight)
        target_val = stock_val * stock_weights
        target_shares = target_val / (prices + 1e-8)

        # Transaction cost and turnover penalty
        trade_amounts = target_shares - self.held_shares
        turnover = np.sum(np.abs(trade_amounts))
        cost = np.sum(np.abs(trade_amounts) * prices * self.transaction_cost)

        # Update holdings and balance
        self.held_shares = target_shares
        self.balance = total_val * cash_weight - cost
        self.portfolio_value = np.sum(self.held_shares * prices) + self.balance

        # Compute reward: pct change minus cost
        pct_change = (self.portfolio_value - old_value) / (old_value + 1e-8)
        reward = pct_change - (cost / (old_value + 1e-8))
        # Risk penalties
        # Volatility penalty
        if self.volatility_window and len(self.return_history) >= self.volatility_window:
            recent = np.array(self.return_history[-self.volatility_window:], dtype=float)
            vol = float(np.std(recent))
            reward -= vol * self.volatility_penalty
        # Drawdown penalty
        if self.drawdown_penalty > 0:
            peak = max(self.portfolio_history)
            drawdown = (peak - self.portfolio_value) / (peak + 1e-8)
            reward -= drawdown * self.drawdown_penalty
        # Turnover penalty
        if self.turnover_penalty > 0:
            reward -= turnover * self.turnover_penalty
        # Update histories
        self.return_history.append(pct_change)
        self.portfolio_history.append(self.portfolio_value)
        self.old_portfolio_change = pct_change

        # Advance and check done
        self.current_step += 1
        done = self.current_step >= self.n_steps
        obs = None if done else self._get_observation()
        return obs, float(reward), done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print(f"Cash Balance: {self.balance:.2f}")
        print(f"Held Shares: {self.held_shares}")