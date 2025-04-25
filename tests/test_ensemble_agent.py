import numpy as np
import torch as T
import pytest

from agents.ensemble import AggregatorNetwork, EnsembleAgent

def test_aggregator_forward():
    input_dim = 10
    hidden_dims = [8]
    n_assets = 2
    net = AggregatorNetwork(input_dim, hidden_dims, n_assets)
    # Dummy input
    x = T.randn(1, input_dim)
    dist, value = net(x)
    action = dist.sample()
    # Should output one action vector
    assert action.shape == (1, n_assets + 1)
    # Value is scalar per batch
    assert value.shape == (1,)
    assert isinstance(value.item(), float)

def test_ensemble_choose_action():
    n_assets = 3
    lookback = 4
    n_features = 5
    agent = EnsembleAgent(
        n_assets=n_assets,
        lookback=lookback,
        n_features=n_features,
        hidden_dims=[16]
    )
    # Create dummy observation
    obs = T.randn(n_assets, lookback, n_features + 2)
    action, log_prob, value = agent.choose_action(obs)
    # Action shape should match n_assets+1
    assert isinstance(action, np.ndarray)
    assert action.shape == (n_assets + 1,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)