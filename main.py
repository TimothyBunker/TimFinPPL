import argparse
import yaml
import random
import os
import logging
import numpy as np
import torch as T
# Disable cuDNN to prevent GPU library load errors on CPU-only hosts
try:
    T.backends.cudnn.enabled = False
except Exception:
    pass
import pandas as pd

from GruVTwo import Agent as LegacyPPOAgent
from TimFinEnv import TimTradingEnv
from WindowedTradingEnv import WindowedTradingEnv
from agents.mlp_agent import MLPAgent
from agents.ensemble import EnsembleAgent
from utils import plot_learning_curve


def main(config, mode):
    # Setup seeds for reproducibility
    seed = config.get('seed')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)

    # Load data
    data_cfg = config['data']
    if mode == 'train':
        logging.info('Training mode: loading training data')
        data = pd.read_parquet(data_cfg['train_file'])
    elif mode == 'test':
        logging.info('Test mode: loading test data')
        data = pd.read_parquet(data_cfg['test_file'])
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")
    logging.info(f"Loaded DataFrame shape: {data.shape}")

    # Determine agent type early (after CLI overrides)
    agent_cfg = config.get('agent', {})
    agent_type = agent_cfg.get('type', 'legacy')
    # Initialize the environment
    env_cfg = config['env']
    env_type = env_cfg.get('type', 'legacy')
    # Multi-agent for ensemble
    if agent_type == 'ensemble' and env_type == 'multi_windowed':
        from multi_agent_env import MultiAgentWindowedEnv
        lookback = env_cfg.get('lookback_window')
        win_file = env_cfg.get(
            'train_windows_file' if mode == 'train' else 'test_windows_file',
            f"data/processed/{mode}_windows_{lookback}.npy"
        )
        logging.info(f"Loading windows from {win_file}")
        data_windows = np.load(win_file)
        agent_cfg = config.get('agent', {})
        agent_ids = agent_cfg.get('ids', []) or ['agent_0']
        env = MultiAgentWindowedEnv(
            data_windows=data_windows,
            agent_ids=agent_ids,
            price_index=env_cfg.get('price_index', 0),
            initial_balance=env_cfg.get('initial_balance', 1000.0),
            transaction_cost=env_cfg.get('transaction_cost', 0.001),
            risk_config=env_cfg.get('risk', {})
        )
    # Crypto-specific windowed environment
    elif env_type == 'crypto_windowed':
        from CryptoTradingEnv import CryptoTradingEnv
        lookback = env_cfg.get('lookback_window')
        win_file = env_cfg.get(
            'train_windows_file' if mode == 'train' else 'test_windows_file',
            f"data/processed/{mode}_windows_{lookback}.npy"
        )
        logging.info(f"Loading crypto windows from {win_file}")
        data_windows = np.load(win_file)
        feat_idx = env_cfg.get('feature_indices', {})
        env = CryptoTradingEnv(
            data_windows=data_windows,
            feature_indices=feat_idx,
            initial_balance=env_cfg.get('initial_balance', 1000.0),
            fee_spot=env_cfg.get('fee_spot', env_cfg.get('transaction_cost', 0.001)),
            fee_perp=env_cfg.get('fee_perp', env_cfg.get('transaction_cost', 0.001)),
            risk_config=env_cfg.get('risk', {})
        )
    elif env_type == 'windowed' or (agent_type != 'ensemble' and env_type == 'multi_windowed'):
        # Single-agent windowed environment (fallback for special agents)
        from WindowedTradingEnv import WindowedTradingEnv
        lookback = env_cfg.get('lookback_window')
        win_file = env_cfg.get(
            'train_windows_file' if mode == 'train' else 'test_windows_file',
            f"data/processed/{mode}_windows_{lookback}.npy"
        )
        logging.info(f"Loading windows from {win_file}")
        data_windows = np.load(win_file)
        env = WindowedTradingEnv(
            data_windows=data_windows,
            price_index=env_cfg.get('price_index', 0),
            initial_balance=env_cfg.get('initial_balance', 1000.0),
            transaction_cost=env_cfg.get('transaction_cost', 0.001),
            risk_config=env_cfg.get('risk', {})
        )
    # For crypto env, only ensemble agent_type is supported
    if env_type == 'crypto_windowed' and agent_type != 'ensemble':
        raise ValueError('crypto_windowed environment only supports agent_type=ensemble')
    else:
        # Legacy DataFrame-based environment
        env = TimTradingEnv(
            data=data,
            initial_balance=env_cfg['initial_balance'],
            lookback_window=env_cfg['lookback_window'],
            initial_grace_period=env_cfg['initial_grace_period']
        )

    # PPO hyperparameters and settings
    ppo_cfg = config['ppo']
    N = ppo_cfg['n_steps']
    batch_size = ppo_cfg['batch_size']
    n_epochs = ppo_cfg['n_epochs']
    n_games = ppo_cfg['n_games']
    alpha = ppo_cfg['alpha']
    entropy_coefficient = ppo_cfg['entropy_coef']
    grad_norm = ppo_cfg['grad_norm']

    n_stocks = env.n_stocks
    # Each observation is flattened: (n_stocks, n_features)
    n_features = int(env.observation_space.shape[0] / n_stocks)
    input_dims = (n_stocks, n_features)

    # Prepare logging and checkpoints directories
    log_cfg = config['logging']
    os.makedirs(log_cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(log_cfg['log_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(log_cfg['plot_file']), exist_ok=True)

    # Initialize the agent based on configuration
    agent_cfg = config.get('agent', {})
    if agent_type == 'mlp':
        # Determine lookback and feature dimensions
        if env_cfg.get('type') == 'windowed':
            lookback = env.lookback
            n_features = env.n_features
        else:
            lookback = env_cfg.get('lookback_window')
            # For legacy, observation is flattened: n_assets*n_features
            n_features = int(env.observation_space.shape[0] / n_stocks)
        agent = MLPAgent(
            n_assets=n_stocks,
            lookback=lookback,
            n_features=n_features,
            hidden_dims=agent_cfg.get('hidden_dims', [256, 128]),
            alpha=agent_cfg.get('alpha', alpha),
            batch_size=agent_cfg.get('batch_size', batch_size),
            n_epochs=agent_cfg.get('n_epochs', n_epochs),
            gamma=agent_cfg.get('gamma', ppo_cfg.get('gamma')),
            gae_lambda=agent_cfg.get('gae_lambda', ppo_cfg.get('gae_lambda')),
            policy_clip=agent_cfg.get('policy_clip', ppo_cfg.get('policy_clip')),
            entropy_coef=agent_cfg.get('entropy_coef', entropy_coefficient),
            grad_norm=agent_cfg.get('grad_norm', grad_norm)
        )
    elif agent_type == 'short_term_ppo':
        from agents.ppo_specialized import ShortTermPPOAgent
        agent = ShortTermPPOAgent(
            lookback=env.lookback,
            n_assets=env.n_assets,
            n_features=env.n_features,
            conv_dims=agent_cfg.get('sub_conv_dims', {}).get('short_term', [32, 32]),
            hidden_dim=agent_cfg.get('sub_hidden_dims', {}).get('short_term', 16),
            lr=ppo_cfg.get('alpha'),
            batch_size=ppo_cfg.get('batch_size'),
            n_epochs=ppo_cfg.get('n_epochs'),
            gamma=ppo_cfg.get('gamma'),
            gae_lambda=ppo_cfg.get('gae_lambda'),
            policy_clip=ppo_cfg.get('policy_clip'),
            entropy_coef=ppo_cfg.get('entropy_coef'),
            grad_norm=ppo_cfg.get('grad_norm'),
            pretrained_path=agent_cfg.get('sub_models', {}).get('short_term')
        )
    elif agent_type == 'long_term_ppo':
        from agents.ppo_specialized import LongTermPPOAgent
        agent = LongTermPPOAgent(
            lookback=env.lookback,
            n_assets=env.n_assets,
            n_features=env.n_features,
            hidden_size=agent_cfg.get('sub_hidden_dims', {}).get('long_term', 64),
            lr=ppo_cfg.get('alpha'),
            batch_size=ppo_cfg.get('batch_size'),
            n_epochs=ppo_cfg.get('n_epochs'),
            gamma=ppo_cfg.get('gamma'),
            gae_lambda=ppo_cfg.get('gae_lambda'),
            policy_clip=ppo_cfg.get('policy_clip'),
            entropy_coef=ppo_cfg.get('entropy_coef'),
            grad_norm=ppo_cfg.get('grad_norm'),
            pretrained_path=agent_cfg.get('sub_models', {}).get('long_term')
        )
    elif agent_type == 'anomaly_ppo':
        from agents.ppo_specialized import AnomalyPPOAgent
        agent = AnomalyPPOAgent(
            lookback=env.lookback,
            n_assets=env.n_assets,
            n_features=env.n_features,
            hidden_dims=agent_cfg.get('sub_hidden_dims', {}).get('anomaly', agent_cfg.get('hidden_dims', [128,64])),
            lr=ppo_cfg.get('alpha'),
            batch_size=ppo_cfg.get('batch_size'),
            n_epochs=ppo_cfg.get('n_epochs'),
            gamma=ppo_cfg.get('gamma'),
            gae_lambda=ppo_cfg.get('gae_lambda'),
            policy_clip=ppo_cfg.get('policy_clip'),
            entropy_coef=ppo_cfg.get('entropy_coef'),
            grad_norm=ppo_cfg.get('grad_norm'),
            pretrained_path=agent_cfg.get('sub_models', {}).get('anomaly')
        )
    elif agent_type == 'sentiment_ppo':
        from agents.ppo_specialized import SentimentPPOAgent
        agent = SentimentPPOAgent(
            n_assets=env.n_assets,
            lookback=env.lookback,
            n_features=env.n_features,
            hidden_dims=agent_cfg.get('hidden_dims', [256,128]),
            alpha=ppo_cfg.get('alpha'),
            batch_size=ppo_cfg.get('batch_size'),
            n_epochs=ppo_cfg.get('n_epochs'),
            gamma=ppo_cfg.get('gamma'),
            gae_lambda=ppo_cfg.get('gae_lambda'),
            policy_clip=ppo_cfg.get('policy_clip'),
            entropy_coef=ppo_cfg.get('entropy_coef'),
            grad_norm=ppo_cfg.get('grad_norm'),
            pretrained_path=agent_cfg.get('sub_models', {}).get('sentiment')
        )
    elif agent_type == 'ensemble':
        # Derive dimensions directly from the windowed environment
        n_assets = env.n_assets
        lookback = env.lookback
        n_features = env.n_features
        agent = EnsembleAgent(
            n_assets=n_assets,
            lookback=lookback,
            n_features=n_features,
            hidden_dims=agent_cfg.get('hidden_dims', [256, 128]),
            alpha=ppo_cfg.get('alpha'),
            batch_size=ppo_cfg.get('batch_size'),
            n_epochs=ppo_cfg.get('n_epochs'),
            gamma=ppo_cfg.get('gamma'),
            gae_lambda=ppo_cfg.get('gae_lambda'),
            policy_clip=ppo_cfg.get('policy_clip'),
            entropy_coef=ppo_cfg.get('entropy_coef'),
            grad_norm=ppo_cfg.get('grad_norm'),
            # per-sub-agent configuration overrides
            sub_hidden_dims=agent_cfg.get('sub_hidden_dims'),
            sub_conv_dims=agent_cfg.get('sub_conv_dims'),
            n_sent_features=agent_cfg.get('n_sent_features', 1)
        )
        # Load pretrained sub-agent weights
        sub_models = agent_cfg.get('sub_models', {})
        # Load pretrained sub-agent weights; if load fails, replace with zero-output stub
        from agents.specialized import SentimentAgent
        for idx, name in enumerate(['anomaly', 'short_term', 'long_term', 'sentiment']):
            path = sub_models.get(name)
            sub = agent.sub_agents[idx]
            if path and os.path.exists(path):
                try:
                    sub.load_models(path)
                    logging.info(f"Loaded sub-agent '{name}' from {path}")
                except Exception as e:
                    logging.warning(f"Failed to load sub-agent '{name}' from {path}: {e}")
                    # fallback stub: zero-output sentiment agent
                    agent.sub_agents[idx] = SentimentAgent(
                        n_assets=n_assets,
                        n_sent_features=agent_cfg.get('n_sent_features', 1),
                        hidden_dims=agent_cfg.get('sub_hidden_dims', {}).get(name, agent_cfg.get('hidden_dims')),
                        lr=ppo_cfg.get('alpha')
                    )
            else:
                logging.warning(f"Sub-agent model path for '{name}' not found or not provided: {path}")
                # no pretrained weights: use zero-output stub
                agent.sub_agents[idx] = SentimentAgent(
                    n_assets=n_assets,
                    n_sent_features=agent_cfg.get('n_sent_features', 1),
                    hidden_dims=agent_cfg.get('sub_hidden_dims', {}).get(name, agent_cfg.get('hidden_dims')),
                    lr=ppo_cfg.get('alpha')
                )
    else:  # legacy PPO
        agent = LegacyPPOAgent(
            n_actions=n_stocks,
            input_dims=input_dims,
            alpha=ppo_cfg.get('alpha', alpha),
            batch_size=ppo_cfg.get('batch_size', batch_size),
            n_epochs=ppo_cfg.get('n_epochs', n_epochs),
            entropy_coef=ppo_cfg.get('entropy_coef', entropy_coefficient),
            grad_norm=ppo_cfg.get('grad_norm', grad_norm),
            policy_clip=ppo_cfg.get('policy_clip'),
            gamma=ppo_cfg.get('gamma'),
            gae_lambda=ppo_cfg.get('gae_lambda')
        )

    score_history = []
    best_score = -np.inf
    learn_iters = 0
    n_steps = 0

    # Model checkpoint base path
    ckpt_base = os.path.join(log_cfg['checkpoint_dir'], f"{agent_type}_model")
    # Load existing models if any
    # For MLP, ensemble, and specialized PPO agents, load from ckpt_base
    specialized_types = ('short_term_ppo', 'long_term_ppo', 'anomaly_ppo', 'sentiment_ppo')
    if agent_type in ('mlp', 'ensemble') + specialized_types:
        try:
            agent.load_models(ckpt_base)
            logging.info(f"Loaded {agent_type} model from {ckpt_base}")
        except Exception:
            logging.info(f"No existing {agent_type} model at {ckpt_base}, starting fresh.")
    else:
        # Legacy (GruVTwo PPO)
        try:
            agent.load_models()
            logging.info("Loaded legacy PPO model")
        except Exception:
            logging.info("No existing legacy model, starting fresh.")

    # Training loop
    if mode == 'train':
        # Multi-agent training only for ensemble meta-controller
        if agent_type == 'ensemble':
            from GruVTwo import Agent as LegacyPPOAgent
            # Instantiate PPO agents for each ID
            agents = {}
            for aid in env.agent_ids:
                agents[aid] = LegacyPPOAgent(
                    n_actions=n_stocks,
                    input_dims=input_dims,
                    alpha=ppo_cfg.get('alpha', alpha),
                    batch_size=ppo_cfg.get('batch_size', batch_size),
                    n_epochs=ppo_cfg.get('n_epochs', n_epochs),
                    entropy_coef=ppo_cfg.get('entropy_coef', entropy_coefficient),
                    grad_norm=ppo_cfg.get('grad_norm', grad_norm),
                    policy_clip=ppo_cfg.get('policy_clip'),
                    gamma=ppo_cfg.get('gamma'),
                    gae_lambda=ppo_cfg.get('gae_lambda')
                )
                try:
                    agents[aid].load_models()
                    logging.info(f"Loaded model for agent '{aid}'")
                except Exception:
                    logging.info(f"No existing model for agent '{aid}', starting fresh.")
            # Histories
            score_hist = {aid: [] for aid in env.agent_ids}
            best_score = {aid: -np.inf for aid in env.agent_ids}
            learn_iters = 0
            step_count = 0
            # Episodes
            for ep in range(n_games):
                obs_dict = env.reset()
                done_all = False
                ep_scores = {aid: 0.0 for aid in env.agent_ids}
                while not done_all:
                    acts, probs, vals = {}, {}, {}
                    for aid, obs in obs_dict.items():
                        a, p, v = agents[aid].choose_action(obs)
                        acts[aid], probs[aid], vals[aid] = a, p, v
                    next_obs, rewards, dones, _ = env.step(acts)
                    step_count += 1
                    for aid in env.agent_ids:
                        agents[aid].remember(
                            obs_dict[aid], acts[aid], probs[aid], vals[aid],
                            rewards[aid], dones[aid]
                        )
                        ep_scores[aid] += rewards[aid]
                    obs_dict = next_obs
                    if step_count % N == 0:
                        for aid in env.agent_ids:
                            agents[aid].learn()
                        learn_iters += 1
                    done_all = dones.get('__all__', all(dones.values()))
                # End of episode: record and save
                for aid in env.agent_ids:
                    score_hist[aid].append(ep_scores[aid])
                    avg = np.mean(score_hist[aid][-100:])
                    if avg > best_score[aid]:
                        best_score[aid] = avg
                    agents[aid].save_models()
                    logging.info(
                        f"Agent {aid} Episode {ep}, Score: {ep_scores[aid]:.4f}, Avg: {avg:.4f}"
                    )
                # Plot individual curves
            for aid in env.agent_ids:
                x = list(range(1, len(score_hist[aid]) + 1))
                plot_learning_curve(
                    x, score_hist[aid],
                    log_cfg['plot_file'].replace('.png', f'_{aid}.png')
                )
        else:
            # Single-agent training
            for i in range(n_games):
                observation = env.reset()
                done = False
                score = 0
                while not done:
                    action, prob, val = agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
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
                if agent_type == 'legacy':
                    agent.save_models()
                    logging.info("Saved legacy PPO model")
                else:
                    agent.save_models(ckpt_base)
                    logging.info(f"Saved {agent_type} model to {ckpt_base}")
                logging.info(
                    f"Episode {i}, Score: {score:.4f}, Avg Score: {avg_score:.4f}, "
                    f"Timesteps: {n_steps}, Learning Steps: {learn_iters}"
                )
            # Plot learning curve
            x = list(range(1, len(score_history) + 1))
            plot_learning_curve(x, score_history, log_cfg['plot_file'])
    else:
        logging.info('Running test episodes...')
        test_episodes = config.get('test_episodes', 10)
        for i in range(test_episodes):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                observation = observation_
            logging.info(f'Test Episode {i}, Final Score: {score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock Trading PPO")
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test'], default='train',
        help='Specify whether to train or test the model.'
    )
    parser.add_argument(
        '--agent_type', type=str, default=None,
        help='Override the agent type specified in the config (e.g., short_term_ppo, ensemble, mlp)'
    )
    # CLI overrides for risk penalties (only effective in windowed env.mode)
    parser.add_argument(
        '--volatility_penalty', type=float, default=None,
        help='Override volatility_penalty in env.risk'
    )
    parser.add_argument(
        '--drawdown_penalty', type=float, default=None,
        help='Override drawdown_penalty in env.risk'
    )
    parser.add_argument(
        '--turnover_penalty', type=float, default=None,
        help='Override turnover_penalty in env.risk'
    )
    parser.add_argument(
        '-o', '--override', action='append', default=[], metavar='KEY=VALUE',
        help='Generic override of config entries via dot-path, e.g. ppo.n_steps=1024'
    )
    args = parser.parse_args()
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Override agent type from CLI if provided
    if args.agent_type:
        cfg_agent = config.setdefault('agent', {})
        cfg_agent['type'] = args.agent_type
    # Override risk penalties from CLI (windowed environments)
    env_cfg = config.setdefault('env', {})
    risk_cfg = env_cfg.setdefault('risk', {})
    if args.volatility_penalty is not None:
        risk_cfg['volatility_penalty'] = args.volatility_penalty
    if args.drawdown_penalty is not None:
        risk_cfg['drawdown_penalty'] = args.drawdown_penalty
    if args.turnover_penalty is not None:
        risk_cfg['turnover_penalty'] = args.turnover_penalty
    # Generic overrides: allow arbitrary config keys via dot notation
    if hasattr(args, 'override'):
        for item in args.override:
            if '=' not in item:
                raise ValueError(f"Invalid override '{item}', must be key.path=value")
            keypath, raw = item.split('=', 1)
            # parse raw value using YAML loader to support numbers, lists, dicts, bools
            try:
                val = yaml.safe_load(raw)
            except Exception:
                val = raw
            # navigate to target
            keys = keypath.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = val
    main(config, args.mode)