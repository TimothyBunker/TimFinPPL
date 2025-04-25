# Agent Architecture Proposal

This document outlines the design of RL agents for the **WindowedTradingEnv** and potential multi-agent extensions.

## 1) Agent Interface
- Agents implement the following methods:
  - `choose_action(observation) -> action, log_prob, value`  
    (single-agent) or  
    `choose_action({agent:obs, ...}) -> {agent:action, ...}, {agent:log_prob, ...}, {agent:value, ...}`  
  - `remember(obs, action, log_prob, value, reward, done)` to store transitions
  - `learn()` to update policy and value networks
  - `save_models()`, `load_models()` for checkpointing

## 2) Single-Agent Architectures
1. **MLP Baseline**
   - Flatten `(n_assets, lookback, n_features+2)` → 1D vector
   - MLP with 2–3 hidden layers → Dirichlet concentration outputs (size `n_assets+1`)
   - Value head: separate MLP to scalar
   - Pros: fast, simple; Cons: ignores temporal and cross-asset structure
2. **Per-Asset 1D-CNN**
   - Apply Conv1D over `lookback` for each asset independently → asset embeddings
   - Concatenate embeddings → MLP → Dirichlet + value
   - Captures short-term temporal patterns per asset
3. **RNN / GRU**
   - GRU over `lookback` per asset → last hidden state embeddings
   - Combine per-asset embeddings via MLP or attention
   - Similar to original `GruVTwo`, but applied per asset

## 3) Multi-Agent Extensions
- Rather than competitive multi-agent, we propose an **ensemble of specialized sub-agents** that each analyze distinct aspects of the market, then feed into a central aggregator:
  1. **Anomaly Detector Agent**
     - Unsupervised model (e.g. autoencoder or robust statistical filter) that flags outlier market states
     - Outputs: anomaly score per asset or flag vector
  2. **Short-Term Predictor Agent**
     - CNN or GRU focusing on very recent window (e.g. 5–10 steps) to predict next-step returns
     - Outputs: predicted return vector or probability of up/down for each asset
  3. **Long-Term Predictor Agent**
     - RNN/Transformer over full lookback to forecast trend over longer horizon (e.g. 20–50 steps)
     - Outputs: multi-step return forecasts per asset
  4. **Sentiment Agent** (optional)
     - Processes sentiment and macro features (news, analyst ratings) via MLP or attention
     - Outputs: sentiment-adjusted bias per asset

### Aggregator Agent
- An MLP or attention-based network that takes:
  - Raw windowed market features `(n_assets, lookback, n_features)`
  - Sub-agent outputs concatenated to high-level state embeddings
  - Portfolio state (current allocations, balances)
- Produces continuous action allocation vector `(n_assets+1,)` (Dirichlet or softmax)

### Training & Execution
- **Centralized Training**: sub-agents and aggregator trained jointly end-to-end via PPO or actor-critic
- **Decoupled Inference**: at execution time, sub-agent models can run in parallel pipelines, then aggregator composes their signals
- **Reward shaping**: aggregator receives composite reward; sub-agents may have auxiliary losses (e.g. forecasting MSE, anomaly detection error)

## 4) Risk-Aware Reward Shaping
- Baseline reward: net portfolio return minus transaction costs
- **Sharpe penalty**: reward -= λ * portfolio volatility over recent window
- **Drawdown control**: penalize drawdowns larger than threshold
- **Turnover constraint**: penalize excessive changes in allocation

## 5) Agent–Environment Goal Alignment
- **Primary objective**: maximize long-term, risk-adjusted portfolio return
- **Secondary objectives**:
  - Minimize transaction costs
  - Control portfolio volatility
  - Respect leverage and cash constraints
- Define custom composite reward:  
  `r_t = ΔPortfolioValue_t – c * Cost_t – λ_vol * StdDev(returns_{t-L:t})`

## 6) Next Steps
1. **Implement MLPAgent** as baseline, verify on `WindowedTradingEnv`  
2. **Integrate risk metrics** into reward calculation  
3. **Develop multi-agent wrapper** for competitive/cooperative portfolios  
4. **Benchmark architectures** (MLP vs CNN vs RNN) on validation set
5. **Hyperparameter tuning** (network sizes, λ_vol, c, learning rates)