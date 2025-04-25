# Training and Integration Plan

This document outlines a step-by-step plan to train specialized sub-agents, integrate them via the EnsembleAgent, and fine-tune end-to-end.

## Phase 1: Pre-train Specialized Sub-Agents
For each sub-agent type (Anomaly, ShortTermPredictor, LongTermPredictor, Sentiment):
1. Data Preparation
   - Use `data/processed/train_windows_<L>.npy` for input windows of shape `(T, n_assets, lookback, n_features)`.
   - For ShortTermPredictor: create targets `next_return[t] = (price_{t+1}/price_{t} - 1)` per asset.
   - For LongTermPredictor: choose horizon `H`, targets `multi_return[t] = (price_{t+H}/price_{t} - 1)`.
   - For AnomalyAgent: use autoencoding: input = window, target = same window.
   - For SentimentAgent: input sentiment & macro columns from scaled DataFrame, targets can be returns or classification.
2. Model Definition
   - Define PyTorch `nn.Module` architectures (MLP, CNN, RNN as appropriate).
   - Add training script `agents/train_specialized.py` with CLI flags:
     ```bash
     python agents/train_specialized.py \
       --agent anomaly \
       --windows data/processed/train_windows_50.npy \
       --output models/anomaly.pt \
       --epochs 20 --batch_size 64 --lr 1e-3
     ```
3. Training Loop
   - Standard supervised learning: MSE or cross-entropy loss.
   - Save best checkpoint on validation split.

## Phase 2: Train Aggregator with Frozen Sub-Agents
1. Load pre-trained sub-agent weights (no gradient updates) in `EnsembleAgent`.
2. Create `WindowedTradingEnv` with risk penalties and `agent.type = ensemble` in `config.yaml`.
3. Train aggregator via PPO:
   - `python main.py --config config_ensemble.yaml --mode train`
   - Only calls `ensemble.learn()` which updates the aggregator network.
   - Sub-agent `predict()` calls remain static.

## Phase 3: Joint Fine-Tuning (Optional)
1. Unfreeze sub-agent networks in `EnsembleAgent.learn()`.
2. Define auxiliary losses for sub-agents:
   - Forecast MSE for predictor agents.
   - Reconstruction loss for anomaly agent.
3. Combine PPO loss (actor+critic) + λ_aux * auxiliary losses.
4. Train end-to-end with smaller learning rate for sub-agents.

## Phase 4: Evaluation and Backtesting
1. Run `main.py --mode test` on `data/processed/test_windows_<L>.npy`.
2. Compute metrics: cumulative return, volatility, Sharpe, max drawdown.
3. Visualize equity curves and allocation time series.

## Utilities to Implement
- `agents/train_specialized.py` stub
- Extend `EnsembleAgent.learn()` to accept `freeze_sub_agents` flag and apply auxiliary losses
- `scripts/backtest.py` for detailed evaluation and plotting

This plan ensures modular training, smooth integration, and ability to iterate on individual components before end-to-end fine-tuning.