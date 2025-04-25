# Crypto-Focused Implementation Roadmap

This roadmap lays out the tasks to transition from stock-based PPO models to a full-featured crypto funding-arbitrage trading system. Use this as a living document to track progress.

## Phase 1: Data Pipeline & Feature Windows (Weeks 1–2)
1. Historical Data Ingestion
   - Write `scripts/fetch_crypto_data.py` to pull spot prices, perpetual funding rates, and orderbook snapshots via CCXT.
   - Store results in Parquet: `data/crypto/spot_{ticker}.parquet`, `data/crypto/perp_{ticker}.parquet`.
2. Feature Engineering
   - Compute per-timestamp features:
     - Basis = perp_price – spot_price
     - Funding carry = funding_rate * position_size
     - Volatility (rolling std of returns)
     - Liquidity metrics (bid-ask spread, depth)
   - Merge into a unified DataFrame with columns: Date, Ticker, SpotPrice, PerpPrice, FundingRate, Basis, Vol, Liquidity, ...
3. Sliding Window Generation
   - Adapt `scripts/generate_windows.py` to produce `data/processed/crypto_windows_{lookback}.npy`
   - Include optional sentiment windows if using social data.

## Phase 2: Environment Redesign (Weeks 2–3)
1. CryptoTradingEnv
   - Extend `WindowedTradingEnv` into `CryptoTradingEnv`:
     - Support dual positions (spot & perp) to maintain hedged carry trade.
     - Accrue funding payments on side you are short (or pay if long).
2. Action Space
   - Continuous control:
     - Single scalar in [-1,1]: fraction of capital to allocate to *hedged* trade (positive = long spot/short perp).
     - Or two outputs: spot_weight, perp_weight (constrained to hedge = equal & opposite).
3. Transaction Costs & Slippage
   - Parametrize exchange fees and per-trade slippage model in env.
4. Unit Tests
   - Write pytest tests for PnL, funding flows, margin safety, and action constraints.

## Phase 3: Reward Shaping & Risk Terms (Weeks 3–4)
1. Reward Function
   - Base reward = ΔPortfolioValue (incl. funding receipts) – transaction_costs
   - Risk penalty: `rep = vol_window(returns)*λ_vol + drawdown*λ_dd + turnover*λ_to`
   - Integrate new risk terms into `CryptoTradingEnv.step()`.
2. Hyperparameter Sweep
   - Use CLI `-o env.risk.*` overrides to find balanced λs on pilot runs.
3. Stability Tests
   - Simulate stress scenarios (funding spikes, flash crashes) to ensure reward shaping yields safe policies.

## Phase 4: Meta-Controller & Ensemble (Weeks 4–5)
1. Plug-in Crypto Env
   - Train existing sub-agents (`anomaly_ppo`, `short_term_ppo`, `long_term_ppo`, `sentiment_ppo`) on `crypto_windows`.
2. Aggregator Enhancements
   - Add entropy regularization and load-balancing loss to the aggregator network.
   - Experiment with softmax vs top-1 gating.
3. Training Loop
   - Use `MultiAgentWindowedEnv` or custom meta-env to feed sub-agent outputs into aggregator.

## Phase 5: Backtesting & Paper Trading (Weeks 5–6)
1. Backtest Script
   - Adapt `scripts/backtest.py` to run on `CryptoTradingEnv`, output PnL, drawdown, Sharpe.
2. Paper Trading
   - Connect to exchange sandbox (e.g. Deribit testnet), feed live data, generate signals without execution.
3. Performance Review
   - Log metrics, generate performance report.

## Phase 6: Live Streaming & Production Deployment (Weeks 6–8)
1. Streaming Env
   - Implement `StreamingCryptoEnv` that ingests a real-time feed generator instead of precomputed windows.
2. Inference Service
   - Containerize with Docker; expose REST endpoint to fetch current state and return trade signal.
3. Execution Engine
   - Integrate with CCXT for order placement; ensure atomic spot/perp hedge trades.
4. Monitoring
   - Add Sentry for error capture; push metrics (latency, PnL, exposure) to Prometheus & Grafana.
5. CI/CD
   - Automate retraining pipeline via GitHub Actions or similar; deploy updated model weekly.

---
_Keep this file updated as tasks complete. Check off items and note any blockers._