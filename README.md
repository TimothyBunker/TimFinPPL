# Tim's Financial PPO Model

### Overview

This project implements a custom Proximal Policy Optimization (PPO) model to enhance financial portfolio stock allocations. The primary goal is to develop a reinforcement learning (RL) agent capable of making informed, adaptive investment decisions in dynamic market environments.

---

## Features

- **Custom Trading Environment**
  - A flexible and dynamic simulation environment tailored for financial markets.
  - Supports continuous portfolio allocation decisions across multiple assets.

- **Reinforcement Learning Architecture**
  - **Actor Model**: A GRU-based network that captures temporal relationships in stock data, enabling better decision-making.
  - **Critic Model**: A GRU-based value network that estimates the value of the current state for improved policy updates.
  - Implements continuous action space for percentage-based portfolio allocations.

- **Reward System**
  - Initially focuses on immediate portfolio changes to encourage learning.
  - Future iterations aim to incorporate sentiment analysis and risk-adjusted reward mechanisms.

---

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- PyTorch
- NumPy
- Pandas
- Gym (for custom environment support)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepositoryName.git
   cd YourRepositoryName
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the dataset:
   - Replace the placeholder data with your own financial data.
   - Ensure the dataset contains features like `Open`, `Close`, `High`, `Low`, and any additional features you plan to use.

---

## How to Use

### Data Pipeline
You can run the end-to-end data pipeline to fetch raw stock data, compute indicators,
merge optional sentiment features, split into training/testing sets, scale features,
and generate sliding windows for RL:
```bash
python Data/pipeline.py \
  --tickers AAPL,MSFT,GOOGL \
  --start_date 2006-01-01 \
  --test_start_date 2020-01-01 \
  --sentiment_file enriched_stock_data_with_sentiment.parquet \
  --macro_file macro_data.parquet \
  --lookback_window 50
```

### Crypto Data Pipeline
Fetch spot and perpetual futures data, plus funding rates, for crypto funding-arbitrage:
```bash
# Activate your venv first:
source finenv/bin/activate
# Install dependencies:
pip install ccxt python-dateutil
# Fetch data (example for Kraken spot, Deribit perp):
python scripts/fetch_crypto_data.py \
  --spot-exchange kraken \
  --perp-exchange deribit \
  --tickers BTC/USD:BTC-PERP,ETH/USD:ETH-PERP \
  --start_date 2022-01-01T00:00:00Z \
  --end_date   2022-06-01T00:00:00Z \
  --timeframe  1h \
  --output      data/crypto/crypto_data.parquet
```
Then generate sliding windows:
```bash
python scripts/generate_crypto_windows.py \
  --parquet data/crypto/crypto_data.parquet \
  --lookback 50 \
  --features spot_price,perp_price,fundingRate,basis \
  --tickers BTC/USD,ETH/USD \
  --output data/processed/crypto_windows_50.npy
```
This saves:
- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `data/processed/train_windows_50.npy`
- `data/processed/test_windows_50.npy`

### Train Specialized Sub-Agents
Before training the ensemble Meta-PPO, pre-train each specialized sub-agent on the training windows:
```bash
LOOKBACK=50
python agents/train_specialized.py \
  --agent anomaly \
  --windows data/processed/train_windows_${LOOKBACK}.npy \
  --output models/anomaly.pt \
  --epochs 20 --batch_size 64 --lr 1e-3

python agents/train_specialized.py \
  --agent short_term \
  --windows data/processed/train_windows_${LOOKBACK}.npy \
  --output models/short_term.pt \
  --epochs 20 --batch_size 64 --lr 1e-3

python agents/train_specialized.py \
  --agent long_term \
  --windows data/processed/train_windows_${LOOKBACK}.npy \
  --output models/long_term.pt \
  --epochs 20 --batch_size 64 --lr 1e-3

python agents/train_specialized.py \
  --agent sentiment \
  --windows data/processed/train_windows_${LOOKBACK}.npy \
  --sentiment_windows data/processed/sentiment_windows_${LOOKBACK}.npy \
  --output models/sentiment.pt \
  --epochs 20 --batch_size 64 --lr 1e-3
```

### Train Ensemble Meta-PPO
Once sub-agents are trained, configure and train the Meta-PPO (ensemble meta-controller):

1. In `config.yaml`, set:
   ```yaml
   agent:
     type: ensemble
     sub_models:
       anomaly: models/anomaly.pt
       short_term: models/short_term.pt
       long_term: models/long_term.pt
       sentiment: models/sentiment.pt
   ```

2. Run the trainer:
   ```bash
   python main.py \
     --config config.yaml \
     --mode train \
     --agent_type ensemble \
     # optionally override hyperparams:
     -o ppo.n_steps=512 \
     -o ppo.batch_size=32 \
     -o ppo.alpha=0.001
   ```

### Backtest Ensemble Agent
Evaluate the trained ensemble on test windows:
```bash
python scripts/backtest.py --config config.yaml --agent_type ensemble
```

### Step 1: Set Up the Data Aggregator
Use the data aggregator to preprocess financial data into a format suitable for the environment:
- Ensure your data is a `pandas.DataFrame` with required columns such as `Date`, `Ticker`, `Open`, `Close`, etc.
- Aggregate or calculate additional features like moving averages, RSI, or Bollinger Bands to enhance the model's input.

### Step 2: Configure the Environment
Modify the environment settings to fit your dataset:
- Load your preprocessed data into the environment:
  ```python
  from env import TimTradingEnv
  env = TimTradingEnv(data=your_data, initial_balance=1000)
  ```
- Adjust parameters like `lookback_window`, `transaction_cost`, and reward-related thresholds.

### Step 3: Train the PPO Model
Use the provided training pipeline to train the PPO agent:
- Import the PPO agent and start training:
  ```python
  from ppo_agent import PPOAgent
  agent = PPOAgent(env=env, n_epochs=10, learning_rate=3e-4)

  agent.train(n_episodes=500)
  ```

### Step 4: Evaluate the Model
Test the trained model on unseen data:
```python
agent.evaluate(env=test_env, n_episodes=50)
```

---

## Challenges and Next Steps

### Challenges
- **Sparse Positive Rewards**: The agent struggles to find consistent positive portfolio changes.
- **Exploration Difficulties**: Limited exploration leads to suboptimal strategies in some cases.
- **Reward Balancing**: Current rewards emphasize immediate changes, which can stagnate long-term learning.

### Next Steps
1. **Enhanced Input Features**
   - Incorporate sentiment analysis, macroeconomic indicators, and additional technical features.
2. **Advanced Reward Mechanisms**
   - Gradually transition from short-term rewards to long-term performance metrics like Sharpe Ratio.
3. **Live Data Integration**
   - Adapt the model to process live market data for real-time decision-making.
4. **Hyperparameter Optimization**
   - Use tools like Optuna for automated fine-tuning.

---

## Visualization
- In the Plot tools I have a generic system for live plotting and plotting in post which you will find aptly named haha
- And for reference here's what FinBot looks like in the metal!
![FinBot.png](images%2FFinBot.png)

---

## Contributing
Feel free to contribute by submitting issues or pull requests. Suggestions for feature enhancements or bug fixes are always welcome.

## Hyperparameter Tuning
You can optimize MLPAgent hyperparameters using Optuna:
1. Install Optuna:
   ```bash
   pip install optuna
   ```
2. Ensure you have generated the window files via the data pipeline:
   ```bash
   python Data/pipeline.py --tickers AAPL,MSFT,GOOGL --start_date 2006-01-01 --test_start_date 2020-01-01 --lookback_window 50
   ```
3. Run HPO (example with 50 trials):
   ```bash
   python experiments/hpo.py --config config.yaml --mode train --trials 50
   ```
4. Results saved to `experiments/hpo_results.csv` and best parameters printed to console.

Optional: visualize study results with Optuna's visualization API.

## Backtesting
After training, backtest a saved agent on test windows:
```bash
python scripts/backtest.py --config config.yaml --agent_type mlp
```
This will load the trained `mlp_model` (or `ensemble_model`, `legacy_model`) from your checkpoint directory,
run a single backtest episode, and save `scripts/backtest_results.csv` containing the portfolio value time series.

---
