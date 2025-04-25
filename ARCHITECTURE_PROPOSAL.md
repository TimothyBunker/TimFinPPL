# Architecture Proposal and Deep Analysis

This document describes the current data schema as inferred from the code, highlights fundamental architectural flaws in the existing implementation, and proposes a revised design for a robust, extensible financial portfolio RL system.

## 1) Inferred Data Schema
- Source: `enriched_stock_data_with_sentiment_training.parquet` and related Parquet files
- Primary columns:
  - `Date` (timestamp/date)
  - `Ticker` (string)
  - Price fields: `Open`, `High`, `Low`, `Close`, `Volume`
  - Technical indicators (per Ticker):
    - Moving Average (MA50), RSI, MACD, Bollinger_High, Bollinger_Low
    - On-Balance Volume (OBV), VWAP, ATR, Stochastic_%K, Williams_%R
    - EMA50, ADX, Log_Returns, Pct_Change
  - (Optionally) Sentiment features (analyst recommendation score, earnings surprise, etc.) if present
- After processing: DataFrame indexed by `(Date, Ticker)`, then `reset_index()` to flat columns
- Scaling: MinMax on all numeric features except raw price columns

## 2) Fundamental Flaws
1. **Temporal information loss**: environment aggregates lookback window into mean/std only; no raw time series passed to agent.
2. **Misused GRU**: Agent GRU receives a sequence of stocks, not a sequence of time steps—temporal modeling is ineffective.
3. **Reward function inconsistency**: mixes percentage gains, transaction penalties, and unit bonuses (+1) of vastly different scales; hard to learn stable policy.
4. **Action-space mismatch**: declared `Box(shape=(n_stocks,))` but code treats last action index as cash allocation—shape inconsistency.
5. **Data leakage risk**: scaling performed on entire dataset before train/test split; potential forward-looking bias.
6. **Environment initialization**: random initial allocations introduce uncontrolled variability at episode start.
7. **Code brittleness**: extensive hard-coded magic numbers (e.g. reshape(29,…), Windows paths), impeding portability and maintainability.
8. **Lack of risk-awareness**: no risk-adjusted reward (Sharpe, drawdown), encourages high-volatility strategies.

## 3) Model Purpose and Objectives
- **Primary goal**: learn a continuous multi-asset portfolio allocation strategy that maximizes cumulative returns net of transaction costs.
- **Secondary considerations**: incorporate transaction costs, limit excessive turnover, adapt to changing market regimes.
- **Desired capabilities**:
  - Respond to temporal dynamics in asset prices and technical indicators
  - Balance return vs. risk (e.g. volatility, drawdown)
  - Integrate new data modalities (sentiment, macro) seamlessly

## 4) Proposed Revised Architecture
#### A) Data Pipeline
1. **Feature Store**: centralized ETL pipeline (e.g. using Airflow or Prefect)
2. **Train/Test Split**: split by time before scaling; fit scalers on train only; persist and apply to test/production
3. **Time-Series Windowing**: for each asset, generate rolling windows of length `L` (e.g. 50 days) → shape `(N_assets, L, N_features)`
4. **Optional Augmentation**: add engineered features (macro variables, cross-asset correlations)

#### B) Environment (Gym API)
1. **Observation**: 3D tensor `(N_assets, L, N_features)` plus current cash balance and holdings vector
2. **Action**: continuous allocation vector `(N_assets + 1,)` representing weights of each asset + cash; enforce `sum(weights)=1`
3. **Transaction model**: fixed or variable costs per trade; slippage model
4. **Reward**: daily portfolio P&L minus costs; optionally composite reward: `return - λ*volatility` or risk parity term
5. **Episode**: fixed-length trading horizon (e.g. 252 trading days); no random resets of initial allocation

#### C) Agent Architecture
1. **Backbone network**: per-asset encoder (e.g. 1D-CNN or GRU) to process time-series → asset embedding
2. **Cross-asset layer**: attention or simple MLP on concatenated embeddings to model interdependencies
3. **Policy head**: produces Dirichlet or softmax distribution over asset weights
4. **Value head**: MLP regressor for state-value estimation
5. **Regularization**: layer normalization, dropout, orthogonal init

#### D) Training & Evaluation
1. **PPO with GAE**: batched updates, clipped surrogate objective, entropy bonus
2. **Logging**: integrate TensorBoard or Weights & Biases for scalars, histograms, trade logs
3. **Evaluation metrics**: cumulative return, annualized volatility, Sharpe ratio, max drawdown on validation set
4. **Hyperparameter tuning**: use Optuna for learning rate, clip ratio, λ, network depth

## 5) Next Steps
1. **Data introspection**: write a small script to confirm Parquet schema, data ranges, missing values
2. **Implement modular ETL**: refactor DataLoader/DataProcessor into a unified pipeline with tests
3. **Redesign Gym Env**: rework state/action/reward per proposal, add assertions and unit tests
4. **Build new agent**: start with per-asset encoders, verify on a toy dataset
5. **Integrate and iterate**: gradually swap components and compare performance against baseline

---
*This proposal builds on the initial code review and defines a clear, modular, and risk-aware architecture for an RL-based portfolio allocator.*