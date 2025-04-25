import numpy as np
import pytest

from WindowedTradingEnv import WindowedTradingEnv

@pytest.fixture
def simple_windows():
    # Create data_windows with T=3, n_assets=1, lookback=2, n_features=1
    # For each t, price sequence [t+1, t+2]
    T, n_assets, lookback, n_features = 3, 1, 2, 1
    data = np.zeros((T, n_assets, lookback, n_features), dtype=float)
    for t in range(T):
        data[t, 0, :, 0] = [t + 1, t + 2]
    return data

def test_reset_and_obs_shape(simple_windows):
    env = WindowedTradingEnv(data_windows=simple_windows, price_index=0,
                             initial_balance=100.0, transaction_cost=0.0)
    obs = env.reset()
    # obs shape: (n_assets, lookback, n_features+2)
    assert obs.shape == (1, 2, 3)
    # Price feature is normalized: mean=1.5, std=0.5 → standardized to [-1, 1]
    assert np.allclose(obs[0, :, 0], [-1.0, 1.0])
    # Portfolio features are normalized ratios = 1 (initial_balance/initial_balance)
    assert np.allclose(obs[0, :, 1], [1.0, 1.0])
    assert np.allclose(obs[0, :, 2], [1.0, 1.0])

def test_step_all_cash(simple_windows):
    env = WindowedTradingEnv(data_windows=simple_windows, price_index=0,
                             initial_balance=50.0, transaction_cost=0.0)
    obs0 = env.reset()
    # Action: [stock_weight, cash_weight]
    action = np.array([0.0, 1.0])
    # Step 1
    obs1, reward1, done1, info1 = env.step(action)
    assert not done1
    # No change in portfolio if all cash
    assert pytest.approx(reward1, abs=1e-8) == 0.0
    # Step 2
    obs2, reward2, done2, info2 = env.step(action)
    assert not done2
    assert pytest.approx(reward2, abs=1e-8) == 0.0
    # Step 3
    obs3, reward3, done3, info3 = env.step(action)
    assert done3
    # When done, obs is None
    assert obs3 is None

def test_step_stock_allocation(simple_windows):
    # price at t=0: [1,2] average price = unused; test full stock allocation
    env = WindowedTradingEnv(data_windows=simple_windows, price_index=0,
                             initial_balance=100.0, transaction_cost=0.0)
    obs = env.reset()
    # Allocate all to stock
    action = np.array([1.0, 0.0])
    # Step 1: old_value=100, stock_value=100, purchase at prices [1,2]
    obs1, reward1, done1, _ = env.step(action)
    # holdings = [100/2] = [50]
    assert np.allclose(env.held_shares, [50.0])
    # portfolio value = sum(held_shares * price) = 50*2 = 100
    assert pytest.approx(env.portfolio_value, abs=1e-5) == 100.0
    # reward = pct change = (100-100)/100 = 0
    assert pytest.approx(reward1, abs=1e-6) == 0.0