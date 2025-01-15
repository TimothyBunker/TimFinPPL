import argparse
import numpy as np
from ppo_fin_rl import Agent
from TFinEnv import CustomTradingEnv  # Import your custom environment
from utils import plot_learning_curve  # Optional, if you have a utility for plotting
import pandas as pd


def main(mode):
    if mode == "train":
        print("Training Mode")

        data = pd.read_parquet('./PPL/enriched_stock_data_with_sentiment_training.parquet')
        print(f"Loaded DataFrame shape: {data.shape}")

    elif mode == "test":
        print("Testing Mode")

        # Load test data
        data = pd.read_parquet('enriched_stock_data_with_sentiment_test.parquet')
        print(f"Loaded Test DataFrame shape: {data.shape}")

    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    # Initialize the environment
    initial_balance = 10000
    lookback_window = 50
    env = CustomTradingEnv(data=data, initial_balance=initial_balance, lookback_window=lookback_window)

    # PPO Parameters
    N = 2048  # Steps per learning iteration
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 300  # Number of episodes
    n_stocks = env.n_stocks
    n_features = env.observation_space.shape[0]
    agent = Agent(
        n_actions=n_stocks,  # Number of continuous actions (number of stocks)
        input_dims=n_features,  # Observation space shape
        alpha=alpha,
        batch_size=batch_size,
        n_epochs=n_epochs
    )

    # For plotting and tracking
    figure_file = 'plots/stock_trading.png'
    best_score = -np.inf  # Start with the lowest possible score
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    # Training Loop (if in "train" mode)
    if mode == "train":
        for i in range(n_games):
            observation, edge_index = env.reset()
            done = False
            score = 0

            while not done:
                action, prob, val = agent.choose_action(observation, edge_index)
                observation_, reward, done, edge_index, info = env.step(action)
                n_steps += 1
                score += reward

                agent.remember(observation, action, prob, val, reward, done)
                observation = observation_

                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print(f'Episode {i}, Score: {score:.1f}, Avg Score: {avg_score:.1f}, '
                  f'Timesteps: {n_steps}, Learning Steps: {learn_iters}')

        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)

    # Testing Loop
    else:
        print("Running test episodes...")
        test_episodes = 10  # Define number of test episodes
        for i in range(test_episodes):
            observation, edge_index = env.reset()
            done = False
            score = 0

            while not done:
                action, prob, val = agent.choose_action(observation, edge_index)
                observation_, reward, done, edge_index, info = env.step(action)
                score += reward
                observation = observation_

            print(f'Test Episode {i}, Final Score: {score:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock Trading PPO")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help="Specify whether to train or test the model.")
    args = parser.parse_args()


    main(mode=args.mode)
