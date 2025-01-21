import argparse
import numpy as np
from sympy import print_tree

from GruVTwo import Agent
from TimFinEnv import TimTradingEnv
from utils import plot_learning_curve
import pandas as pd


def main(mode):
    if mode == "train":
        print("Training Mode")

        data = pd.read_parquet(r'C:\\Users\\Tim\\PycharmProjects\\ppobasics\\PPL\\enriched_stock_data_with_sentiment_training.parquet')
        print(f"Loaded DataFrame shape: {data.shape}")

    elif mode == "test":
        print("Testing Mode")

        # Load test data
        data = pd.read_parquet('enriched_stock_data_with_sentiment_test.parquet')
        print(f"Loaded Test DataFrame shape: {data.shape}")

    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    # Initialize the environment
    initial_balance = 1000
    lookback_window = 50
    grace_period = 1
    env = TimTradingEnv(data=data,
                        initial_balance=initial_balance,
                        lookback_window=lookback_window,
                        initial_grace_period=grace_period
                        )

    # PPO Parameters
    N = 2048
    batch_size = 64
    n_epochs = 10
    n_games = 100000
    n_stocks = env.n_stocks
    n_features = int(env.observation_space.shape[0] / n_stocks)
    input_dims = (n_stocks, n_features)
    entropy_coefficient = 0.01
    alpha = 0.0003
    grad_norm = 0.1

    agent = Agent(
        n_actions=n_stocks,
        input_dims=input_dims,
        alpha=alpha,
        batch_size=batch_size,
        n_epochs=n_epochs,
        entropy_coef=entropy_coefficient,
        grad_norm=grad_norm,
    )

    # For plotting and tracking
    figure_file = 'C:\\Users\\Tim\\PycharmProjects\\ppobasics\\PPL\\plots'
    best_score = -np.inf
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    if mode == "train":
        # agent.load_models()
        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0

            while not done:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                print(f'observation: {observation}\naction: {action}\nprob: {prob}\nval: {val}')
                # env.render()
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

            print(f'Episode {i}, Score: {score:.8f}, Avg Score: {avg_score:.8f}, '
                  f'Timesteps: {n_steps}, Learning Steps: {learn_iters}')

        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)

    # Testing Loop
    else:
        print("Running test episodes...")
        test_episodes = 10
        for i in range(test_episodes):
            observation = env.reset()
            done = False
            score = 0

            while not done:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                observation = observation_

            print(f'Test Episode {i}, Final Score: {score:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock Trading PPO")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help="Specify whether to train or test the model.")
    args = parser.parse_args()


    main(mode=args.mode)
