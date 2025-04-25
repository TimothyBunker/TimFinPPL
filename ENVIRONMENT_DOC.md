### Purpose of the Environment

Your **RL environment is the laboratory and referee** for every model you train or run live. It must:

1. **Replicate the economics of a real funding-rate carry trade**—spot ↔ perpetual hedging, funding credits/debits, exchange fees, slippage, and margin rules.
    
2. **Expose the information your agents will have in production** (state), let them change the portfolio (action), and give back an immediate, risk-aware reward.
    
3. **Enforce hard constraints and failure modes** (liquidations, exchange halts, API gaps) so agents learn to survive, not just profit in a perfect sim.
    
4. **Run in two modes**
    
    - **Back-test / sim**: fast, offline, reproducible.
        
    - **Live / paper-trade**: streaming real-time data with the _same_ API, so you can switch to production by flipping a flag.
        

---

## 1 Core Capabilities the Environment Must Provide

|Capability|Why it matters|Implementation detail|
|---|---|---|
|**Synchronous spot + perp price feed**|Carry P/L depends on _both_ legs at identical timestamps.|Every step delivers `spot_price_t`, `perp_price_t`, and `funding_rate_t`. Interpolate / forward-fill any missing ticks so time grids align.|
|**Funding cash-flow engine**|Funding payments are the edge you capture.|At each funding interval (e.g., every 8 h), credit or debit `position_perp × funding_rate × interval/365` to P&L.|
|**Order-book & slippage model**|Tiny funding edges are wiped out by execution cost.|Parameterise slippage as `s = k · (order_size / depth)` + fixed tick. Use realistic depth from historical books or a simple square-root model.|
|**Transaction fees & spread**|Exchanges charge 2–10 bps; spread widens in stress.|Deduct maker/taker fees and half-spread cost when positions change.|
|**Margin & liquidation logic**|A “market-neutral” trade can still liquidate if one leg is offside.|Track maintenance margin; if `equity / margin_req < 1`, trigger forced close at a haircut price and large penalty.|
|**Leverage limits & funding direction flips**|Student safety: don’t allow 5× in a 30 % vol regime.|Expose `max_leverage_t` as part of state; clamp actions that exceed it.|
|**Exchange anomalies**|Flash crashes, feed outages, negative funding spikes.|Inject rare events drawn from historical anomalies (or synthetic). Your Anomaly-PPO must recognise & react.|
|**Time-of-day/calendar**|Funding prints at predictable times; liquidity cycles.|Include `sin/cos` encodings of hour-of-day, day-of-week in state.|
|**Multiple assets optional**|Later, diversify to ETH, SOL, etc.|Make asset dimension optional so env handles `N` pairs with shared cash pool.|

---

## 2 Interface Specification

python

CopyEdit

`class CarryTradeEnv(gym.Env):     # ---- spaces ----     observation_space: Box(low, high, shape=(state_dim,))     action_space: Box(low=-1, high=+1, shape=(1,))   # scalar weight in [-1,1]      # ---- step() semantics ----     state, reward, done, info = env.step(action)      # action: desired fraction of equity to allocate #         ( +1 = full-size long-carry; -1 = full-size short-carry )`

### Observation (`state`) breakdown

|Group|Example features|
|---|---|
|**Market**|spot price, perp price, rolling basis, log-returns (1h, 4h), realised vol, order-book depth, funding rate (current & 3-period SMA)|
|**Portfolio**|current spot qty, perp qty, net inventory, unused cash, equity, unrealised P&L, current drawdown, margin ratio|
|**Expert context**|each frozen expert’s suggested action, last 4 h reward, running Sharpe|
|**Timing**|sin(hour/24·2π), cos(hour/24·2π), funding_countdown (mins to next print)|
|**Risk flags**|anomaly score (z-score of return), exchange status bit (normal / halt)|

### Action interpretation

1. Convert scalar `a∈[-1,1]` → _target_ hedged notional:
    
    - `size = a × equity × leverage_cap`
        
2. Determine desired spot & perp positions:
    
    - `spot_target = +size / 2`
        
    - `perp_target = −size / 2` (for long-carry; signs flip if `a<0`)
        
3. Submit orders to reach target, applying slippage & fees.
    
4. Update positions, cash, equity.
    

### Reward

ini

CopyEdit

`carry_pnl   = funding_cashflow basis_pnl   = Δ(spot_price - perp_price) * position_perp tx_costs    = fees + slippage risk_penalty= β * max(0, drawdown - dmax)  reward = carry_pnl + basis_pnl - tx_costs - risk_penalty`

_Tune β so a 1 % breach of drawdown cap wipes out roughly one day’s average carry, forcing the meta-controller to respect risk._

---

## 3 Objectives the Environment Must Support

|Objective|Why it matters|How the env enables it|
|---|---|---|
|**Train agents that maximise _net funding capture_ while limiting drawdown**|Matches the business goal: steady compounding with capital safety.|Accurate funding payments, costs, and drawdown metric baked into reward.|
|**Test risk procedures and anomaly responses**|In live trading, execution glitches kill profits.|Built-in random anomaly generator & liquidation logic let you verify the Anomaly-PPO and risk manager react correctly.|
|**Provide identical API for sim → paper → live**|Avoid training–serving mismatch.|All data feed / execution functions are dependency-injected; in sim they read from CSV, in live they call exchange API.|
|**Allow curriculum & ablation experiments**|You may want to turn off slippage or anomalies during early training.|Feature flags: `env = CarryTradeEnv(slippage=False)` etc.|
|**Fast vectorised rollouts**|RL needs millions of steps.|Support batched environments (e.g., `gym.vector.AsyncVectorEnv`) over multiple historical shards.|

---

## 4 Minimum Viable Milestones

1. **MVP Sim**
    
    - Static CSV for BTC spot & perp, fixed 8 h funding.
        
    - Simple slippage (0.05 %).
        
    - No anomalies.
        
    - Action = scalar in [-1,1].
        
2. **Realistic Back-tester (Week 2)**
    
    - Variable funding schedule & rate.
        
    - T x fees, linear depth-based slippage.
        
    - Basic margin liquidation.
        
3. **Stress-test Mode (Week 5)**
    
    - Historical flash crash replay (12 March 2020, May 2021).
        
    - Synthetic exchange downtime (no fills for N steps).
        
4. **Live-Link Wrapper (Week 8)**
    
    - Swap CSV reader for WebSocket feed.
        
    - Execution calls to exchange sandbox; identical step API.
        

Achieving Milestone 2 lets you begin serious agent training; Milestone 3 validates risk handling; Milestone 4 is the flip-switch to paper-trade and, eventually, real capital.

---

### Bottom Line

Your environment’s job is **fidelity + safety**: faithfully mimic every cash-flow, constraint, and failure mode of a spot-vs-perp carry trade, and embed those into the reward and dynamics so that any model which does well in sim is _already fit for live money_
