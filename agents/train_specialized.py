#!/usr/bin/env python3
"""
Train specialized sub-agents on sliding window data.
Supported agents: anomaly, short_term, long_term, sentiment
"""
import sys, os
import argparse
import yaml
import numpy as np
import torch as T
# Disable cuDNN to avoid GPU library loading errors on CPU-only environments
try:
    T.backends.cudnn.enabled = False
except Exception:
    pass

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.specialized import (
    AnomalyAgent,
    ShortTermPredictorAgent,
    LongTermPredictorAgent,
    SentimentAgent,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a specialized sub-agent')
    parser.add_argument('--agent', type=str, choices=['anomaly', 'short_term', 'long_term', 'sentiment'], required=True,
                        help='Which sub-agent to train')
    parser.add_argument('--windows', type=str, required=True,
                        help='Path to .npy file of shape (T, n_assets, lookback, n_features)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path prefix to save trained model (e.g. models/anomaly.pt)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='YAML config file for additional params')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer sizes for MLP (or per-agent hyperparams)')
    parser.add_argument('--sentiment_windows', type=str, default=None,
                        help='Path to .npy file of sentiment features shape (T, n_assets, n_sent_features)')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(windows, agent_type):
    """
    Prepare dataset (X, y) for supervised training of sub-agents.
    TODO: implement proper target construction for each agent.
    """
    # windows: numpy (T, n_assets, lookback, n_features)
    T_dim, n_assets, lookback, n_feat = windows.shape
    # Flatten input features per sample
    X = windows.reshape(T_dim, -1)
    # Placeholder targets
    if agent_type == 'anomaly':
        # Autoencoder: target = input
        y = X.copy()
    elif agent_type == 'short_term':
        # Next-step return per asset
        prices = windows[:, :, -1, 0]  # assume price at feature index 0
        ret = np.zeros_like(prices)
        ret[:-1] = (prices[1:] / prices[:-1]) - 1.0
        y = ret.reshape(T_dim, -1)
    elif agent_type == 'long_term':
        # Long-horizon returns, e.g., H=lookback
        H = lookback
        prices = windows[:, :, -1, 0]
        ret = np.zeros_like(prices)
        ret[:-H] = (prices[H:] / prices[:-H]) - 1.0
        y = ret.reshape(T_dim, -1)
    elif agent_type == 'sentiment':
        # Placeholder: same as short_term
        prices = windows[:, :, -1, 0]
        ret = np.zeros_like(prices)
        ret[:-1] = (prices[1:] / prices[:-1]) - 1.0
        y = ret.reshape(T_dim, -1)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return X, y

def train_loop(agent, X, y, epochs, batch_size, lr, hidden_dims, output_path):
    """
    Generic training loop for sub-agent with MLP.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Prepare dataset
    X_tensor = T.tensor(X, dtype=T.float32)
    y_tensor = T.tensor(y, dtype=T.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model: simple MLP for supervised regression
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = nn.Sequential()
    prev_dim = input_dim
    for i, h in enumerate(hidden_dims):
        model.add_module(f"linear{i}", nn.Linear(prev_dim, h))
        model.add_module(f"relu{i}", nn.ReLU())
        prev_dim = h
    model.add_module("out", nn.Linear(prev_dim, output_dim))
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = T.optim.Adam(model.parameters(), lr=lr)

    print(f"Training {agent.__class__.__name__} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    T.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")

def main():
    args = parse_args()
    config = load_config(args.config)

    # Load data windows
    if not os.path.exists(args.windows):
        raise RuntimeError(f"Windows file not found: {args.windows}")
    windows = np.load(args.windows)

    # Unpack windows dimensions
    T_dim, n_assets, lookback, n_features = windows.shape
    # Pre-train anomaly agent separately
    if args.agent == 'anomaly':
        agent = AnomalyAgent(
            lookback=lookback,
            n_assets=n_assets,
            n_features=n_features,
            hidden_dims=args.hidden_dims,
            lr=args.lr
        )
        agent.learn(windows, args.epochs, args.batch_size)
        agent.save_models(args.output)
        return
    # Instantiate and train the chosen sub-agent
    if args.agent == 'short_term':
        agent = ShortTermPredictorAgent(
            lookback=lookback,
            n_assets=n_assets,
            n_features=n_features,
            conv_dims=args.hidden_dims,
            hidden_dim=args.hidden_dims[-1],
            lr=args.lr
        )
        agent.learn(windows, args.epochs, args.batch_size)
        agent.save_models(args.output)
        return
    elif args.agent == 'long_term':
        agent = LongTermPredictorAgent(
            lookback=lookback,
            n_assets=n_assets,
            n_features=n_features,
            hidden_size=args.hidden_dims[-1],
            lr=args.lr
        )
        agent.learn(windows, args.epochs, args.batch_size)
        agent.save_models(args.output)
        return
    elif args.agent == 'sentiment':
        # Load sentiment feature windows
        if args.sentiment_windows is None or not os.path.exists(args.sentiment_windows):
            raise RuntimeError(f"Sentiment windows file not found: {args.sentiment_windows}")
        sentiment_windows = np.load(args.sentiment_windows)
        agent = SentimentAgent(
            n_assets=n_assets,
            n_sent_features=sentiment_windows.shape[2],
            hidden_dims=args.hidden_dims,
            lr=args.lr
        )
        agent.learn(windows, sentiment_windows, args.epochs, args.batch_size)
        agent.save_models(args.output)
        return
    else:
        raise ValueError(f"Invalid agent: {args.agent}")

if __name__ == '__main__':
    main()