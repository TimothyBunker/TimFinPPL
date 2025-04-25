# Environment and Model Objectives

This document clarifies the high-level goals for the custom trading environment and the accompanying reinforcement learning models.  It emphasizes support for multi-agent setups, production readiness, and extensibility.

## 1. Overarching Vision
- Simulate realistic financial markets for automated portfolio management.
- Enable training of one or more RL agents (single- or multi-agent) in parallel, each with its own portfolio or strategy.
- Provide a smooth path from research to production deployment, with clear interfaces to live or historical data.

## 2. Environment Goals
1. **Multi-Agent Support**
   - Each agent should receive its own observation (state) and emit its own actions.
   - The environment should manage and isolate agent portfolios (balances, holdings).
   - Allow for inter-agent interactions: e.g., market impact, shared constraints, or competition for liquidity.
2. **Realism and Fidelity**
   - Time-series observations: sliding windows of market features per asset.
   - Transaction modeling: costs, slippage, and delay.
   - Cash account and leverage support.
3. **Configurability and Extensibility**
   - Flexible asset universe: dynamic list of tickers, ability to add/remove assets at runtime.
   - Customizable reward functions: return, risk-adjusted (Sharpe, drawdown), genre-specific metrics.
   - Plug-in modules for additional data: macroeconomic indicators, alternative data, sentiment, news.
4. **Production Readiness**
   - Deterministic resets: reproducible episodes for backtesting.
   - Streaming mode: step-by-step ingestion of live market data.
   - Serialization: save/load environment state and data pipelines.
   - Monitoring and logging: trade logs, P&L, risk exposures, latency.

## 3. Model Objectives
1. **Scalable Policy Architectures**
   - Support for single- and multi-agent policies (e.g. decentralized, centralized training).
   - Modular backbone networks: per-asset encoders + cross-asset aggregators (attention, graph nets).
2. **Continuous Action Spaces**
   - Allow for fractional portfolio weights plus cash allocation, summing to 1.
   - Respect constraints: non-negativity, leverage limits, sector exposure caps.
3. **Robust Training Algorithms**
   - Standard RL algorithms (PPO, DDPG, A2C, SAC) adapted to financial settings.
   - Multi-agent RL methods (MADDPG, QMIX) for cooperative or competitive scenarios.
4. **Risk-Aware Learning**
   - Incorporate risk terms into objective (volatility penalty, drawdown control).
   - Support for reward shaping and filtering (e.g. only learn on trading days, handle missing data).

## 4. Next Steps
1. **Environment Refactor**
   - Introduce multi-agent API (`reset()`, `step()` accept/return dicts of observations, rewards, dones).
   - Support precomputed sliding windows as inputs and roll in real-time data stream.
2. **Model Integration**
   - Define agent interface and wrappers for single- vs multi-agent training loops.
   - Implement baseline single-agent PPO on the new env; verify invariants.
3. **Production Hooks**
   - Connect data pipeline outputs (Parquet + `.npy` windows) to environment ingestion.
   - Add live data adapters for non-blocking streaming in prod.

---
_Document created to guide the redesign of the trading environment and RL agents._