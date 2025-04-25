"""
Hyperparameter Optimization for MLPAgent using Optuna.
"""
import sys, os
import argparse
# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import optuna
import numpy as np

import torch as T
from agents.mlp_agent import MLPAgent
from WindowedTradingEnv import WindowedTradingEnv

def evaluate_agent(agent, env, n_episodes=5):
    """
    Run the agent for a few episodes and return average total reward.
    """
    total_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, log_prob, value = agent.choose_action(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    return float(np.mean(total_rewards))

def objective(trial, config, mode):
    # Suggest hyperparameters
    hidden1 = trial.suggest_int('hidden1', 64, 512, log=True)
    hidden2 = trial.suggest_int('hidden2', 32, 256, log=True)
    alpha = trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    # Load environment windows
    env_cfg = config['env']
    windows_file = env_cfg['train_windows_file'] if mode=='train' else env_cfg['test_windows_file']
    # Ensure windows file exists
    if not os.path.exists(windows_file):
        raise RuntimeError(
            f"Windows file '{windows_file}' not found. Please run Data/pipeline.py"
        )
    data_windows = np.load(windows_file)
    env = WindowedTradingEnv(
        data_windows=data_windows,
        price_index=env_cfg.get('price_index', 0),
        initial_balance=env_cfg.get('initial_balance', 1000.0),
        transaction_cost=env_cfg.get('transaction_cost', 0.001),
        risk_config=env_cfg.get('risk', {})
    )
    # Instantiate agent
    obs_shape = env.observation_space.shape
    n_assets = obs_shape[0]
    lookback = obs_shape[1]
    n_features = obs_shape[2] - 2
    agent = MLPAgent(
        n_assets=n_assets,
        lookback=lookback,
        n_features=n_features,
        hidden_dims=[hidden1, hidden2],
        alpha=alpha,
        batch_size=batch_size,
        n_epochs=config['ppo']['n_epochs'],
        gamma=gamma,
        gae_lambda=config['ppo']['gae_lambda'],
        policy_clip=config['ppo']['policy_clip'],
        entropy_coef=config['ppo']['entropy_coef'],
        grad_norm=config['ppo']['grad_norm']
    )
    # Evaluate
    avg_reward = evaluate_agent(agent, env, n_episodes=3)
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, config, args.mode), n_trials=args.trials)

    print('Best trial parameters:')
    print(study.best_trial.params)
    # Save study results
    try:
        df = study.trials_dataframe()
        results_file = os.path.join(os.path.dirname(__file__), 'hpo_results.csv')
        df.to_csv(results_file, index=False)
        print(f'HPO trial results saved to {results_file}')
    except Exception as e:
        print(f'Warning: unable to save HPO results: {e}')

if __name__ == '__main__':
    main()