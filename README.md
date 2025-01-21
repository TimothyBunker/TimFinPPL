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

---
