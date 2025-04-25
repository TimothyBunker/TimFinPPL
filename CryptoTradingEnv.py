import gym
from gym import spaces
import numpy as np

class CryptoTradingEnv(gym.Env):
    """
    Crypto funding-arbitrage environment using sliding windows.
    Each action is a hedged fraction per asset in [-1,1], representing long-spot/short-perp (positive)
    or short-spot/long-perp (negative) positions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data_windows: np.ndarray,
        feature_indices: dict,
        initial_balance: float = 1000.0,
        fee_spot: float = 0.001,
        fee_perp: float = 0.001,
        risk_config: dict = None
    ):
        # data_windows: (T, n_assets, lookback, n_features)
        assert data_windows.ndim == 4, "data_windows must be 4D"
        self.data_windows = data_windows
        self.n_steps, self.n_assets, self.lookback, self.n_features = data_windows.shape
        self.n_stocks = self.n_assets
        # Feature indices mapping, with defaults
        fi = feature_indices or {}
        self.spot_idx = fi.get('spot_price', 0)
        self.perp_idx = fi.get('perp_price', 1)
        self.funding_idx = fi.get('funding_rate', 2)
        self.initial_balance = initial_balance
        self.fee_spot = fee_spot
        self.fee_perp = fee_perp
        # Risk penalties
        rc = risk_config or {}
        self.volatility_window = rc.get('volatility_window')
        self.volatility_penalty = rc.get('volatility_penalty', 0.0)
        self.drawdown_penalty = rc.get('drawdown_penalty', 0.0)
        self.turnover_penalty = rc.get('turnover_penalty', 0.0)

        # Portfolio state
        self._reset_portfolio()

        # Observation: (n_assets, lookback, n_features_normalized)
        obs_shape = (self.n_assets, self.lookback, self.n_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        # Action: fraction per asset in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def _reset_portfolio(self):
        self.balance = self.initial_balance
        self.spot_shares = np.zeros(self.n_assets, dtype=float)
        self.perp_shares = np.zeros(self.n_assets, dtype=float)
        self.portfolio_value = self.initial_balance
        self.return_history = []
        self.portfolio_history = [self.initial_balance]
        # Random start
        self.current_step = np.random.randint(0, self.n_steps)

    def reset(self):
        self._reset_portfolio()
        return self._get_observation()

    def _get_observation(self):
        window = self.data_windows[self.current_step]  # shape (n_assets, lookback, n_features)
        # Normalize per asset/feature
        m = window.mean(axis=1, keepdims=True)
        s = window.std(axis=1, keepdims=True) + 1e-8
        return ((window - m) / s).astype(np.float32)

    def step(self, action):
        # Clip action
        a = np.clip(action, -1.0, 1.0)
        old_val = self.portfolio_value
        # Prices and funding from last lookback index
        win = self.data_windows[self.current_step]
        # Extract prices and funding by configured indices
        spot_p = win[:, -1, self.spot_idx]
        perp_p = win[:, -1, self.perp_idx]
        funding = win[:, -1, self.funding_idx]
        # Determine target notionals
        notionals = np.abs(a) * old_val
        # Spot and perp shares
        new_spot = np.sign(a) * notionals / (spot_p + 1e-8)
        new_perp = -new_spot  # hedge
        # Transaction costs
        cost_spot = self.fee_spot * np.sum(np.abs(new_spot - self.spot_shares) * spot_p)
        cost_perp = self.fee_perp * np.sum(np.abs(new_perp - self.perp_shares) * perp_p)
        # Update holdings
        self.spot_shares = new_spot
        self.perp_shares = new_perp
        # Funding payment (receive if short perp)
        funding_pay = -np.sum(self.perp_shares * perp_p * funding)
        # Cash balance
        invested = np.sum(notionals)
        self.balance = old_val - invested - cost_spot - cost_perp + funding_pay
        # New portfolio value
        val_spot = np.sum(self.spot_shares * spot_p)
        val_perp = np.sum(self.perp_shares * perp_p)
        self.portfolio_value = self.balance + val_spot + val_perp
        # Reward: return
        ret = (self.portfolio_value - old_val) / (old_val + 1e-8)
        reward = ret
        # Risk penalties
        if self.volatility_window and len(self.return_history) >= self.volatility_window:
            vol = float(np.std(self.return_history[-self.volatility_window:]))
            reward -= vol * self.volatility_penalty
        peak = max(self.portfolio_history)
        dd = (peak - self.portfolio_value) / (peak + 1e-8)
        reward -= dd * self.drawdown_penalty
        turnover = np.sum(np.abs(a))
        reward -= turnover * self.turnover_penalty
        # Record
        self.return_history.append(ret)
        self.portfolio_history.append(self.portfolio_value)
        # Advance
        self.current_step += 1
        done = self.current_step >= self.n_steps
        obs = None if done else self._get_observation()
        return obs, float(reward), done, {}

    def render(self, mode='human'):
        print(f"Step {self.current_step}, PV={self.portfolio_value:.2f}, Balance={self.balance:.2f}")