import numpy as np
import torch as T
import pytest

from agents.mlp_agent import MLPAgent

@pytest.mark.parametrize("n_assets,lookback,n_features", [
    (2, 3, 4),
    (5, 10, 6)
])
def test_mlp_choose_action(n_assets, lookback, n_features):
    hidden_dims = [32]
    agent = MLPAgent(
        n_assets=n_assets,
        lookback=lookback,
        n_features=n_features,
        hidden_dims=hidden_dims,
        alpha=1e-3,
        batch_size=4,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        entropy_coef=0.01,
        grad_norm=0.5
    )
    # Create dummy observation
    obs = np.random.randn(n_assets * lookback * (n_features + 2))
    action, log_prob, value = agent.choose_action(obs)
    # Check types and shapes
    assert isinstance(action, np.ndarray)
    assert action.shape == (n_assets + 1,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)