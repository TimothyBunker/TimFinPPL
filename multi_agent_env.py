import gym
from gym import spaces
import numpy as np


class MultiAgentWindowedEnv(gym.Env):
    """
    Multi-agent trading environment using shared windowed market data.
    Each agent manages its own portfolio over the same market windows.
    Actions are dicts mapping agent_id -> allocation vector (n_assets+1).
    Observations, rewards, dones, infos are returned as dicts keyed by agent_id.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        data_windows: np.ndarray,
        agent_ids: list,
        price_index: int = 0,
        initial_balance: float = 1000.0,
        transaction_cost: float = 0.001,
        risk_config: dict = None
    ):
        super(MultiAgentWindowedEnv, self).__init__()
        assert data_windows.ndim == 4, "data_windows must be 4D"
        self.data_windows = data_windows
        self.n_steps, self.n_assets, self.lookback, self.n_features = data_windows.shape
        # Alias for backward compatibility with single-agent code
        self.n_stocks = self.n_assets
        self.price_index = price_index
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        # Risk penalties
        rc = risk_config or {}
        self.volatility_window = rc.get('volatility_window')
        self.volatility_penalty = rc.get('volatility_penalty', 0.0)
        self.drawdown_penalty = rc.get('drawdown_penalty', 0.0)
        self.turnover_penalty = rc.get('turnover_penalty', 0.0)
        # Agents
        self.agent_ids = list(agent_ids)
        # Per-agent state
        self._make_agent_state()

        # Observation space: same as WindowedTradingEnv
        obs_shape = (self.n_assets, self.lookback, self.n_features + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        # Action: per-agent allocation (n_assets + cash)
        act_shape = (self.n_assets + 1,)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=act_shape, dtype=np.float32
        )

    def _make_agent_state(self):
        """Initialize per-agent portfolio states"""
        self.current_step = 0
        self.balance = {a: self.initial_balance for a in self.agent_ids}
        self.held_shares = {a: np.zeros(self.n_assets, dtype=float) for a in self.agent_ids}
        self.portfolio_value = {a: self.initial_balance for a in self.agent_ids}
        self.return_history = {a: [] for a in self.agent_ids}
        self.portfolio_history = {a: [self.initial_balance] for a in self.agent_ids}

    def reset(self):
        """Reset environment state and return initial observations"""
        self._make_agent_state()
        # Randomize starting index for each multi-agent episode
        self.current_step = np.random.randint(0, self.n_steps)
        return self._get_obs()

    def _get_obs(self):
        window = self.data_windows[self.current_step]
        # Normalize window per asset/feature
        mean_feat = window.mean(axis=1, keepdims=True)
        std_feat = window.std(axis=1, keepdims=True) + 1e-8
        normalized = (window - mean_feat) / std_feat
        # Build per-agent obs (same market obs but different portfolio features)
        obs = {}
        for a in self.agent_ids:
            # portfolio ratios
            val_ratio = self.portfolio_value[a] / (self.initial_balance + 1e-8)
            cash_ratio = self.balance[a] / (self.initial_balance + 1e-8)
            pf = np.stack([
                np.full((self.n_assets, self.lookback), val_ratio),
                np.full((self.n_assets, self.lookback), cash_ratio)
            ], axis=2)
            obs[a] = np.concatenate([normalized, pf], axis=2).astype(np.float32)
        return obs

    def step(self, actions: dict):
        """
        actions: dict of agent_id -> allocation vector
        Returns: obs, rewards, dones, infos (all dicts)
        """
        window = self.data_windows[self.current_step]
        prices = window[:, -1, self.price_index]
        rewards = {}
        dones = {}
        infos = {}
        # Apply each agent's action
        for a, action in actions.items():
            alloc = np.clip(action, 0.0, 1.0)
            alloc = alloc / (alloc.sum() + 1e-8)
            cash_w = alloc[-1]
            stock_w = alloc[:-1]
            old_val = self.portfolio_value[a]
            # target shares
            stock_val = old_val * (1.0 - cash_w)
            tgt_val = stock_val * stock_w
            tgt_shares = tgt_val / (prices + 1e-8)
            trades = tgt_shares - self.held_shares[a]
            turnover = np.sum(np.abs(trades))
            cost = np.sum(np.abs(trades) * prices * self.transaction_cost)
            # update state
            self.held_shares[a] = tgt_shares
            self.balance[a] = old_val * cash_w - cost
            self.portfolio_value[a] = np.sum(tgt_shares * prices) + self.balance[a]
            # compute reward
            pct = (self.portfolio_value[a] - old_val) / (old_val + 1e-8)
            r = pct - (cost / (old_val + 1e-8))
            # risk penalties
            rh = self.return_history[a]
            if self.volatility_window and len(rh) >= self.volatility_window:
                vol = float(np.std(rh[-self.volatility_window:]))
                r -= vol * self.volatility_penalty
            if self.drawdown_penalty > 0:
                peak = max(self.portfolio_history[a])
                dd = (peak - self.portfolio_value[a]) / (peak + 1e-8)
                r -= dd * self.drawdown_penalty
            if self.turnover_penalty > 0:
                r -= turnover * self.turnover_penalty
            # record
            self.return_history[a].append(pct)
            self.portfolio_history[a].append(self.portfolio_value[a])
            rewards[a] = float(r)
            dones[a] = False
            infos[a] = {}
        # global done when out of data
        self.current_step += 1
        done_all = self.current_step >= self.n_steps
        for a in self.agent_ids:
            dones[a] = done_all
        dones['__all__'] = done_all
        obs = self._get_obs() if not done_all else None
        return obs, rewards, dones, infos

    def render(self, mode='human'):
        for a in self.agent_ids:
            print(f"Agent {a} Step {self.current_step}: PV={self.portfolio_value[a]:.2f}, Cash={self.balance[a]:.2f}")