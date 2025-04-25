import numpy as np
import torch as T
from agents.specialized import AnomalyAgent

def test_anomaly_predict_shape():
    # Create constant windows: T=5, n_assets=2, lookback=3, n_features=4
    T_dim, n_assets, lookback, n_feat = 5, 2, 3, 4
    windows = np.ones((T_dim, n_assets, lookback, n_feat), dtype=float)
    # Init AnomalyAgent
    agent = AnomalyAgent(
        lookback=lookback,
        n_assets=n_assets,
        n_features=n_feat,
        hidden_dims=[8,4],
        lr=1e-3
    )
    # After zero training, predict anomaly scores
    scores = agent.predict(windows[0])
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (n_assets,)
    # Scores should be finite
    assert np.all(np.isfinite(scores))