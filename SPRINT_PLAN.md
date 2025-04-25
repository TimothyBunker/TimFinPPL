# Sprint Plan: Implement and Integrate Specialized Sub-Agents

Objectives:
1. Implement concrete architectures for each specialized sub-agent:
   - **AnomalyAgent**: autoencoder MLP already implemented.
   - **ShortTermPredictorAgent**: 1D-CNN to forecast 1-step returns.
   - **LongTermPredictorAgent**: GRU to forecast H-step returns.
   - **SentimentAgent**: simple MLP to incorporate sentiment/macro features.
2. Integrate `learn()` and `predict()` methods for each, enabling supervised training via `train_specialized.py`.
3. Ensure all agents share a common interface and consistent input/output dimensions.
4. Update `main.py` and ensemble training to load and freeze these specialized agents before aggregator training.

Milestones:
---
### Milestone 1: Short-Term Predictor
- File: `agents/specialized.py`
- Build `ShortTermPredictorAgent`:
  * Constructor args: `(lookback, n_assets, n_features, hidden_dims, lr)`
  * Model: `nn.Conv1d(n_features, C1, kernel=3, padding=1) -> ReLU -> Conv1d(C1, C2, kernel=3, padding=1) -> ReLU -> GlobalAvgPool1d -> MLP(C2 -> hidden_dims -> 1)`
  * `predict(window_np) -> np.ndarray` of shape `(n_assets,)`
  * `learn(windows, epochs, batch_size)` supervised trainer: MSE on true next-step returns
  * `save_models(path)`, `load_models(path)` implemented
---
### Milestone 2: Long-Term Predictor
- File: `agents/specialized.py`
- Build `LongTermPredictorAgent`:
  * Constructor args: `(lookback, n_assets, n_features, hidden_size, lr)`
  * Model: `nn.GRU(input_size=n_features, hidden_size=H, batch_first=True)` + MLP([H -> H//2 -> 1])
  * `predict(window_np) -> np.ndarray` forecasting H-day returns
  * `learn(windows, epochs, batch_size)` supervised trainer: MSE on multi-step returns
---
### Milestone 3: Sentiment Agent
- File: `agents/specialized.py`
- Build `SentimentAgent`:
  * Constructor args: `(n_assets, n_sent_features, hidden_dims, lr)`
  * Model: MLP([n_sent_features -> hidden_dims -> 1]) per asset
  * `predict(window_np or sentiment_df_ptr) -> np.ndarray` bias vector
  * `learn(...)` to train on sentiment → return targets
---
### Milestone 4: Update Training Script
- File: `agents/train_specialized.py`
- Call each agent’s `learn(windows, epochs, batch_size)` instead of generic MLP
- Pass `lookback`, `n_assets`, and `n_features` from windows shape and config
---
### Milestone 5: Ensemble Integration
- File: `main.py` and `agents/ensemble.py`
- After instantiating `EnsembleAgent`, re-instantiate sub-agents with correct dims and load their weights
- Ensure `choose_action()` calls real `predict()` methods from each sub-agent
---
### Milestone 6: End-to-End Test
- Run full pipeline:
  ```bash
  python Data/pipeline.py --tickers AAPL,MSFT,GOOGL --start_date ... --test_start_date ... --lookback_window 50
  python agents/train_specialized.py --agent short_term ...
  python agents/train_specialized.py --agent long_term ...
  python agents/train_specialized.py --agent sentiment ...
  python agents/train_specialized.py --agent anomaly ...
  python main.py --config config.yaml --mode train
  python scripts/backtest.py --config config.yaml --agent_type ensemble
  ```
---
Estimated Timeline: 3–5 coding days, starting with ShortTermPredictorAgent.