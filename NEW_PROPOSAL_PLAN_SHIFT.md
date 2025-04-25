# Funding-Rate (Carry-Trade) RL Trading System

## Quickstart
1. Generate sentiment windows:
   ```bash
   python scripts/generate_sentiment_windows.py \
       --parquet data/processed/train.parquet \
       --windows data/processed/train_windows_50.npy \
       --sentiment_col sentiment_pred \
       --output data/processed/sentiment_windows_50.npy
   ```
2. Train specialized agents:
   ```bash
   python agents/train_specialized.py --agent anomaly \
       --windows data/processed/train_windows_50.npy \
       --output models/anomaly.pt \
       --epochs 100 --batch_size 64 --lr 1e-3 --hidden_dims 128 64

   python agents/train_specialized.py --agent short_term \
       --windows data/processed/train_windows_50.npy \
       --output models/short_term.pt \
       --epochs 100 --batch_size 64 --lr 1e-3 --hidden_dims 32 16

   python agents/train_specialized.py --agent long_term \
       --windows data/processed/train_windows_50.npy \
       --output models/long_term.pt \
       --epochs 100 --batch_size 64 --lr 1e-3 --hidden_dims 128 64

   python agents/train_specialized.py --agent sentiment \
       --windows data/processed/train_windows_50.npy \
       --sentiment_windows data/processed/sentiment_windows_50.npy \
       --output models/sentiment.pt \
       --epochs 100 --batch_size 64 --lr 1e-3 --hidden_dims 32 16
   ```
3. Run ensemble training:
   ```bash
   python main.py --config config.yaml --mode train
   ```

This system implements a **safe, funding-rate arbitrage** (cash-and-carry) strategy on crypto (perpetual futures vs spot) using reinforcement learning. Funding-rate arbitrage means hedging a perpetual futures position with the underlying spot to earn the funding-rate spread. For example, when funding rates are high (market is net long), an arbitrageur **shorts the perpetual futures and buys the spot**, capturing the funding payments​[blog.amberdata.io](https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata#:~:text=When%20the%20majority%20of%20the,buy%20spots%20as%20a%20backup)​[discovery.ucl.ac.uk](https://discovery.ucl.ac.uk/id/eprint/10141040/7/Borrageiro_Reinforcement_Learning_for_Systematic_FX_Trading_VoR.pdf#:~:text=%29,EXPERIMENT%20METHODS). This is analogous to FX “carry trades” where traders hold the higher-yielding currency funded by a low-yield one​[discovery.ucl.ac.uk](https://discovery.ucl.ac.uk/id/eprint/10141040/7/Borrageiro_Reinforcement_Learning_for_Systematic_FX_Trading_VoR.pdf#:~:text=%29,EXPERIMENT%20METHODS). The goal is **low-risk, steady returns** on 15-min to hourly data, with strict risk controls (max drawdown, stop-loss, etc.) built in.

- **Key focus:** Prioritize capital preservation with strict risk limits at every step. Balance returns with drawdown control (e.g. via Sharpe-like reward shaping​[mdpi.com](https://www.mdpi.com/2079-9292/11/9/1506#:~:text=Each%20deal%20entails%20transaction%20charges%2C,1%20%2C%20112%2C15)).
    
- **Timeframe:** Medium-frequency (15min–1hr bars) using real-time data feeds.
    
- **Production requirements:** Modular, scalable architecture with real exchange connectivity (order execution, error handling), logging/monitoring, and fail-safes (circuit-breakers, emergency stops).
    

## Model Adaptation

Leverage the existing PPO models (long-term, short-term, anomaly, sentiment) by **retraining and reinterpreting** them for funding-arbitrage trading:

- **Long-Term PPO:** Originally for broad allocation, now use to set **base carry positions**. For example, decide which asset’s spot-futures pair to hold or tilt long/short over multi-day horizons. Retrain on crypto data (spot and perpetual prices, funding rates) so it learns the slow-moving funding trend signals.
    
- **Short-Term PPO:** Use for **intra-day timing** and fine adjustments. It should react to 15-min price/funding fluctuations (e.g., small mispricings or funding jumps) and execute shorter hedge cycles. Retrain it on high-frequency funding and price moves.
    
- **Sentiment PPO:** Keep this model to ingest social/news sentiment indicators. In crypto, extreme bullish/bearish sentiment often drives funding spikes. Adapt its inputs to crypto sentiment feeds (e.g. Twitter, Reddit, on-chain metrics) and let it bias the other models’ signals when sentiment deviates.
    
- **Anomaly PPO:** Use for **risk gating**. Train it to detect outliers (e.g. flash crashes, exchange halts, extreme volatility) in real-time features. Its output can trigger safe modes (e.g. halt new trades, reduce exposure) when unusual patterns appear. Retrain with labeled historical anomalies if available.
    

Each model should be **retrained or fine-tuned** in a new simulation environment that reflects spot-perp trading (see below). Their outputs (trade signals or action values) can be combined by a higher-level controller (see Meta-Controller below) rather than acting independently.

## New Modules and Architecture

Build supporting modules around the RL models to achieve a production-grade system:

- **Real-Time Data & Feature Builder:** Continuously ingest market data (spot prices, perpetual prices, funding rates, volumes) via streaming APIs. Compute derived features: rolling funding averages, basis (futures minus spot), volatility estimates, liquidity/depth metrics, and sentiment scores. Package these into the state vector for the RL agent. Ensure synchronized timestamps between spot and perp feeds.
    
- **Meta-Controller (Sizing Controller):** Implement a **meta-level PPO agent** that learns to allocate capital among strategies or adjust position sizes. For instance, it can weight the base model signals to maximize portfolio Sharpe. This controller enforces capital limits and risk budgets, effectively combining the ensemble’s outputs into final orders.
    
- **Execution Engine:** Integrate with exchange APIs (e.g. via CCXT) to place and manage orders on spot and perpetual markets. Include order management: split large orders if needed, apply slippage limits, and handle partial fills. Ensure atomic execution of hedged trades (e.g. submitting spot and perp orders in tandem). All orders pass pre-trade risk checks (e.g. max notional, leverage limit) before sending.
    
- **Risk Manager / Safety Module:** Continuously monitor portfolio exposure, P&L, and drawdowns. Enforce hard limits: max position size per asset, max portfolio notional, maximum daily loss (e.g. 1–2%). If any limit is breached, automatically **pause trading** and reduce positions. Implement stop-loss rules (e.g. liquidate positions if they lose X%). Use circuit breakers to halt new trades after N consecutive losses or a sharp market move.
    
- **Backtest & Paper-Trade Framework:** Maintain a simulation mode mirroring live architecture. Replay historical data (with funding and fees) to validate models. After sim success, paper-trade (or use exchange sandbox) to test live-feeding data without real capital.
    
- **Logging & Monitoring:** Log all trades, decisions, and market states. Build dashboards for real-time P&L, exposures, drawdown, trade history, and system health. Alert on anomalies (API failures, risk breaches, model divergence).
    
- **Model Retraining Pipeline:** Periodically (e.g. weekly/monthly) retrain the PPO models on recent data to adapt to market changes. Automate data collection and training runs, with validation on hold-out sets.
    

**Implementation notes:** Use asynchronous or multi-threaded design: separate processes for data ingestion, model inference, execution, and monitoring. Containerize each component (e.g. with Docker) so they can run on cloud servers. Employ message queues for data (e.g. Kafka) if throughput is high, though 15-min bars may allow simpler solutions.

## Environment Design (State, Actions, Rewards)

Redefine the RL environment to model a funding arbitrage scenario instead of stock allocation:

- **State Observations:** Include features for the perp-spot pair(s) being traded, for each asset if multiple. Key observations:
    
    - _Prices:_ current and recent history of spot price and perpetual futures price.
        
    - _Funding Rates:_ latest funding rate of the perpetual, plus short-term moving average or recent funding trend.
        
    - _Basis/Spread:_ current difference between futures price (adjusted for funding) and spot price.
        
    - _Volume & Liquidity:_ recent trading volume, order book depth to assess slippage risk.
        
    - _Volatility:_ e.g. ATM implied vol or historical vol of the asset (to adjust position sizing).
        
    - _Sentiment Indicators:_ aggregated sentiment score or its trend (from social media, news).
        
    - _Current Positions:_ current holdings in spot and perp (including leverage used), cash balance.
        
    - _Risk Metrics:_ current drawdown, unrealized P&L, time since last trade, etc., so the agent “knows” its risk state.
        
- **Action Space:** Controls how to adjust hedged positions. Possible designs:
    
    - _Continuous actions:_ e.g. a single action value in [-1,1] representing the fraction of capital to allocate to a fully hedged carry trade (positive means long spot & short perp, negative means short spot & long perp). Or separate continuous controls for spot position and perp position, constrained to a hedged relationship.
        
    - _Discrete actions:_ e.g. “enter long carry (long spot/short perp)”, “enter short carry”, “exit/flatten”, plus “hold”. Possibly with magnitude buckets (small/medium/large).
        
    - The action should enforce a **market-neutral hedge** except for exploiting price drift (e.g. if expecting spot to move). In a pure funding arbitrage mode, actions maintain offsetting spot vs perp positions.
        
    - Always enforce _feasible actions_ – e.g. cannot exceed max leverage or sell more than held. The environment should clamp or block invalid actions.
        
- **Reward Shaping:** The reward should encourage profit from funding arbitrage while penalizing risk. For example:
    
    - **PnL Return:** Base reward = change in portfolio value (log-return or net profit) over the time step, after accounting for funding payments and transaction costs. Funding payments (if short perp) _add_ to P&L; funding receipts (if long perp) subtract. Include commissions/slippage.
        
    - **Risk Penalty:** Subtract a term proportional to recent volatility or drawdown. For instance, use a Sharpe-like reward: R = (net P&L) – λ·(portfolio volatility)​[mdpi.com](https://www.mdpi.com/2079-9292/11/9/1506#:~:text=Each%20deal%20entails%20transaction%20charges%2C,1%20%2C%20112%2C15). This penalizes large swings. Alternatively, directly penalize drawdowns: R = P&L – κ·(current drawdown)^2, discouraging dips.
        
    - **Position Cost:** Penalize excessive leverage or turnover to prevent wild swings. e.g. a small negative reward for high notional.
        
    - **Balance:** Tune coefficients so that moderately steady profit from small spreads is favored over rare big wins with huge swings. One approach is to maximize a risk-adjusted metric (like Calmar ratio), but in RL form. The cited strategy uses Sharpe for trade-off​[mdpi.com](https://www.mdpi.com/2079-9292/11/9/1506#:~:text=Each%20deal%20entails%20transaction%20charges%2C,1%20%2C%20112%2C15); you can similarly incorporate standard deviation of returns.
        
    - Ensure reward is scaled reasonably (e.g. daily return) to stabilize training.
        

Example reward formulation:

ini

CopyEdit

`Reward = Δ PortfolioValue – α·(StdDev(recent returns))` 

where α>0 trades off risk (as in Sharpe)​[mdpi.com](https://www.mdpi.com/2079-9292/11/9/1506#:~:text=Each%20deal%20entails%20transaction%20charges%2C,1%20%2C%20112%2C15).

Adjust reward shaping iteratively in simulation to get the desired behavior (e.g. use small α for conservative policies).

## Leverage and Position Sizing

**Leverage** means borrowing funds to amplify the position. For example, 2× leverage allows twice the notional exposure for the same capital. While funding arbitrage can use leverage (since profits are often small per unit capital), it must be **carefully controlled**:

- **Start Low:** Begin with no or minimal leverage (1×). This ensures extreme moves won’t liquidate positions unexpectedly. Only increase if the strategy is proven stable in sim and live tests.
    
- **Safe Limits:** Cap leverage to a conservative level (e.g. 2×–3× maximum). Higher leverage (10×+) in crypto is possible but very risky. Set the max so that a moderate adverse move (e.g. 10% drop) does not wipe equity.
    
- **Dynamic Leverage:** Adjust allowable leverage based on volatility. Lower the cap if vol spikes, and gradually restore when calm.
    
- **Initial Steps:** Use **isolated margin** per trade so liquidations are contained. Later, as confidence grows, consider cross-margin or portfolio margin for efficiency.
    
- **Monitor Margin Usage:** The system should watch margin ratios in real-time. If margin falls below a threshold, it should auto-close/reduce positions.
    

In summary, **treat leverage as adjustable over time**: begin trading without it, and only **scale into leverage** after extensive live testing. Always enforce margin-stop rules and maintain a healthy cash buffer.

## Deployment Plan (Week-by-Week)

1. **Weeks 1–2 (Sim Environment & Data Pipelines):**
    
    - Set up historical data collection for spot prices, perpetual prices, and funding rates from chosen exchanges (e.g. Binance, Deribit).
        
    - Build the simulated trading environment (e.g. an OpenAI Gym) with the new state/action/reward design above. Validate it by running simple strategies.
        
    - Implement the real-time feature builder (backtest mode) to compute inputs (basis, volatility, sentiment proxies).
        
    - Adapt the existing PPO models’ code to interface with this new env.
        
2. **Weeks 3–4 (Model Training & Evaluation):**
    
    - Retrain or fine-tune the four PPO models on historical crypto data (e.g. last 1–2 years). Use the sim env to provide training episodes.
        
    - Monitor training: ensure the agents learn basic arbitrage (e.g. positive reward) without diverging. Compare portfolio value growth vs baseline.
        
    - Begin integrating risk checks in sim (e.g. disallow actions exceeding leverage).
        
3. **Week 5 (Meta-Controller & Risk Modules):**
    
    - Develop the Meta-PPO sizing controller. For example, use two-stage training: first freeze base models and train meta-agent to allocate capital between them.
        
    - Implement the risk manager logic: define hard limits (max drawdown, max position) in code. Connect it to the sim environment so that hitting a limit ends an episode.
        
    - Test how risk limits affect agent behavior; adjust rewards or penalties if needed.
        
4. **Weeks 6–7 (Backtesting & Stress Tests):**
    
    - Backtest the ensemble (base models + meta-controller) on out-of-sample historical data (simulate live trading with slippage). Measure performance metrics (return, Sharpe, MaxDD).
        
    - Stress-test with extreme scenarios (e.g. sudden 20% crash, funding spike) by injecting shocks into the sim. Check that the anomaly model or risk module reacts (e.g. stops trading).
        
    - Refine models or safety triggers based on results (e.g. lower leverage, adjust stop-loss).
        
5. **Week 8 (Live Infrastructure Setup):**
    
    - Set up real-time pipelines: connect to exchange APIs in testnet mode (if available) or with very small capital. Verify that market data and funding updates are received in real time.
        
    - Containerize components (data-fetcher, model inference, execution, risk-monitor). Perform integration testing in a “paper trading” mode: the system runs on live data but does not execute real orders.
        
    - Ensure logging and monitoring are in place (e.g. send alerts for API disconnects, model errors).
        
6. **Week 9 (Paper Trading & Dry Run):**
    
    - Begin paper trading on live market data. Let the system generate trade signals and simulate orders (but don’t submit to exchange). Compare P&L against paper account data.
        
    - Tune any latencies or bugs. Validate that order execution logic (e.g. simultaneous spot/perp hedge) works as intended.
        
    - Run the system continuously (24/7 if needed) to test stability.
        
7. **Week 10 (Live Trading Pilot):**
    
    - **Deploy with small capital** (e.g. 1–5% of target fund) on a real account. Start with conservative settings (no leverage, tight risk limits).
        
    - Closely monitor early results and system health. Be ready to intervene (e.g. manually close all positions) if something goes wrong.
        
    - Gradually allow higher notional only if performance and stability are verified.
        
8. **Week 11+ (Scale-Up & Maintenance):**
    
    - If pilot goes well, slowly ramp up capital and possibly leverage (within the safe bounds tested). Continue to run multiple validation backtests monthly.
        
    - Keep retraining models on fresh data (possibly weekly) to adapt. Maintain and refine risk parameters.
        
    - Document all procedures, and if possible, add automated tests (e.g. simulate days in 1 hour) to catch regressions.
        

Each stage emphasizes **validation before taking more risk**. By week 10, the system should be a stable, production-ready engine: trading automatically in real-time with safeguards, ready for further scaling.

**References:** The strategy and design principles draw on established practice: funding-rate arbitrage is a known carry strategy​[blog.amberdata.io](https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata#:~:text=When%20the%20majority%20of%20the,buy%20spots%20as%20a%20backup)​[discovery.ucl.ac.uk](https://discovery.ucl.ac.uk/id/eprint/10141040/7/Borrageiro_Reinforcement_Learning_for_Systematic_FX_Trading_VoR.pdf#:~:text=%29,EXPERIMENT%20METHODS), and risk-aware RL often uses reward forms like Sharpe ratios to balance profit and volatility​[mdpi.com](https://www.mdpi.com/2079-9292/11/9/1506#:~:text=Each%20deal%20entails%20transaction%20charges%2C,1%20%2C%20112%2C15). The above plan assembles these ideas into a comprehensive production system.
