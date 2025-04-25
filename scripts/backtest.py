#!/usr/bin/env python3
"""
Backtesting script: evaluate a trained agent on test windows and compute performance metrics.
"""
import sys, os
import argparse
import yaml
import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WindowedTradingEnv import WindowedTradingEnv
from agents.mlp_agent import MLPAgent
from agents.ensemble import EnsembleAgent
from GruVTwo import Agent as LegacyPPOAgent

def backtest(agent, env):
    """
    Run a single backtest episode and return time series of portfolio values.
    """
    obs = env.reset()
    done = False
    values = []
    while not done:
        action, logp, val = agent.choose_action(obs)
        obs, reward, done, info = env.step(action)
        # record portfolio value
        values.append(env.portfolio_value)
    return values

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest trained agent')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config YAML')
    parser.add_argument('--agent_type', type=str, choices=['mlp','ensemble','legacy'], default='mlp')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    env_cfg = cfg['env']
    # Load test windows
    lookback = env_cfg.get('lookback_window')
    test_file = env_cfg.get('test_windows_file', f"data/processed/test_windows_{lookback}.npy")
    data_windows = np.load(test_file)
    env = WindowedTradingEnv(
        data_windows=data_windows,
        price_index=env_cfg.get('price_index',0),
        initial_balance=env_cfg.get('initial_balance',1000.0),
        transaction_cost=env_cfg.get('transaction_cost',0.001),
        risk_config=env_cfg.get('risk',{})
    )
    n_assets, loo, n_feat_plus2 = env.observation_space.shape
    n_features = n_feat_plus2 - 2
    # Initialize agent
    agent_cfg = cfg.get('agent',{})
    at = args.agent_type
    if at == 'mlp':
        agent = MLPAgent(
            n_assets=n_assets,
            lookback=lookback,
            n_features=n_features,
            hidden_dims=agent_cfg.get('hidden_dims',[256,128]),
            alpha=agent_cfg.get('alpha', cfg['ppo']['alpha']),
            batch_size=agent_cfg.get('batch_size',cfg['ppo']['batch_size']),
            n_epochs=agent_cfg.get('n_epochs',cfg['ppo']['n_epochs']),
            gamma=agent_cfg.get('gamma',cfg['ppo']['gamma']),
            gae_lambda=agent_cfg.get('gae_lambda',cfg['ppo']['gae_lambda']),
            policy_clip=agent_cfg.get('policy_clip',cfg['ppo']['policy_clip']),
            entropy_coef=agent_cfg.get('entropy_coef',cfg['ppo']['entropy_coef']),
            grad_norm=agent_cfg.get('grad_norm',cfg['ppo']['grad_norm'])
        )
        ckpt = os.path.join(cfg['logging']['checkpoint_dir'], 'mlp_model')
        try:
            agent.load_models(ckpt)
        except FileNotFoundError:
            print(f"MLP model checkpoint not found at {ckpt}. Please train the MLPAgent first.")
            return
    elif at == 'ensemble':
        agent = EnsembleAgent(
            n_assets=n_assets,
            lookback=lookback,
            n_features=n_features,
            hidden_dims=agent_cfg.get('hidden_dims',[256,128])
        )
        ckpt = os.path.join(cfg['logging']['checkpoint_dir'], 'ensemble_model')
        try:
            agent.load_models(ckpt)
        except FileNotFoundError:
            print(f"Ensemble model checkpoint not found at {ckpt}_aggregator.pt. Please train the ensemble first.")
            return
    else:
        agent = LegacyPPOAgent(
            n_actions=n_assets,
            input_dims=(n_assets, n_features),
            alpha=cfg['ppo']['alpha'],
            batch_size=cfg['ppo']['batch_size'],
            n_epochs=cfg['ppo']['n_epochs'],
            entropy_coef=cfg['ppo']['entropy_coef'],
            grad_norm=cfg['ppo']['grad_norm'],
            policy_clip=cfg['ppo']['policy_clip'],
            gamma=cfg['ppo']['gamma'],
            gae_lambda=cfg['ppo']['gae_lambda']
        )
        try:
            agent.load_models()
        except FileNotFoundError:
            print("Legacy PPO model not found. Please train first.")
            return
    # Run backtest
    values = backtest(agent, env)
    # Save results
    out_file = os.path.join('scripts','backtest_results.csv')
    import pandas as pd
    pd.DataFrame({'portfolio_value': values}).to_csv(out_file, index=False)
    print(f"Backtest results saved to {out_file}")

if __name__ == '__main__':
    main()