import os
import numpy as np
import torch as T
# Disable cuDNN backend to prevent GPU library load errors
try:
    T.backends.cudnn.enabled = False
except Exception:
    pass
import torch.nn as nn
from agents.base import BaseAgent

class AnomalyAgent(BaseAgent):
    """
    Autoencoder-based anomaly detector.
    Reconstructs input windows and scores by reconstruction MSE per asset.
    """
    def __init__(self, lookback: int, n_assets: int, n_features: int, hidden_dims: list, lr: float = 1e-3):
        import torch.nn as nn
        # input dimension per sample: lookback * n_features
        self.lookback = lookback
        self.n_assets = n_assets
        self.n_features = n_features
        # Build per-asset autoencoder MLP: input_dim -> hidden_dims -> input_dim
        self.autoencoder = nn.Sequential()
        input_dim = lookback * n_features
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            self.autoencoder.add_module(f"enc_lin{i}", nn.Linear(prev, h))
            self.autoencoder.add_module(f"enc_relu{i}", nn.ReLU())
            prev = h
        for i, h in enumerate(reversed(hidden_dims)):
            self.autoencoder.add_module(f"dec_lin{i}", nn.Linear(prev, h))
            self.autoencoder.add_module(f"dec_relu{i}", nn.ReLU())
            prev = h
        self.autoencoder.add_module("dec_out", nn.Linear(prev, input_dim))
        self.optimizer = T.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.criterion = T.nn.MSELoss(reduction='none')
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.autoencoder.to(self.device)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute anomaly score per asset: MSE of reconstruction.
        Args:
            observation: np.ndarray, shape (n_assets, lookback, n_features)
        Returns:
            scores: np.ndarray, shape (n_assets,)
        """
        self.autoencoder.eval()
        # Flatten per-asset
        # Flatten and normalize each asset window
        obs = observation.reshape(self.n_assets, -1).astype(np.float32)
        x = T.tensor(obs, dtype=T.float32).to(self.device)
        # Standardize per sample
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        with T.no_grad():
            recon = self.autoencoder(x_norm)
            # Compute MSE on normalized data per asset
            mse = self.criterion(recon, x_norm).mean(dim=1)
        return mse.cpu().numpy()

    def remember(self, *args, **kwargs):
        pass

    def learn(self, windows: np.ndarray, epochs: int, batch_size: int):
        """
        Train the autoencoder on windows (T, n_assets, lookback, n_features).
        """
        from torch.utils.data import DataLoader, TensorDataset
        # Prepare dataset: each asset-window is one sample
        data = windows.reshape(-1, self.lookback * self.n_features)
        ds = TensorDataset(T.tensor(data, dtype=T.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            self.autoencoder.train()
            total_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                # normalize per sample
                mean = xb.mean(dim=1, keepdim=True)
                std = xb.std(dim=1, keepdim=True) + 1e-8
                xb_norm = (xb - mean) / std
                recon = self.autoencoder(xb_norm)
                loss = self.criterion(recon, xb_norm).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg = total_loss / len(ds)
            print(f"AnomalyAgent Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")

    def save_models(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save(self.autoencoder.state_dict(), path)

    def load_models(self, path: str):
        self.autoencoder.load_state_dict(T.load(path, map_location=self.device))
    # No-op implementations to satisfy BaseAgent interface
    def choose_action(self, observation):
        raise NotImplementedError("AnomalyAgent does not implement choose_action")

    def remember(self, *args, **kwargs):
        # Not used for supervised anomaly training
        pass



class ShortTermPredictorAgent(BaseAgent):
    """
    CNN-based short-term return predictor (next-bar forecasts).
    """
    def __init__(self, lookback: int, n_assets: int, n_features: int,
                 conv_dims: list = [32, 32], hidden_dim: int = 16, lr: float = 1e-3):
        import torch.nn as nn
        self.lookback = lookback
        self.n_assets = n_assets
        self.n_features = n_features
        # Build CNN: (batch=n_assets, in_channels=n_features, seq_len=lookback)
        layers = [
            nn.Conv1d(n_features, conv_dims[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_dims[0], conv_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output (batch, conv_dims[1], 1)
        ]
        self.cnn = nn.Sequential(*layers)
        # MLP head: conv_dims[1] -> hidden_dim -> 1
        self.mlp = nn.Sequential(
            nn.Linear(conv_dims[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.cnn.to(self.device)
        self.mlp.to(self.device)
        self.optimizer = T.optim.Adam(list(self.cnn.parameters()) + list(self.mlp.parameters()), lr=lr)
        self.criterion = nn.MSELoss()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict next-step returns for each asset.
        `observation` shape: (n_assets, lookback, n_features)
        Returns np.array shape (n_assets,) of predictions.
        """
        self.cnn.eval()
        self.mlp.eval()
        # Prepare input tensor: (n_assets, n_features, lookback)
        x = T.tensor(observation, dtype=T.float32).permute(0, 2, 1).to(self.device)
        with T.no_grad():
            conv_out = self.cnn(x).squeeze(-1)  # (n_assets, conv_dims[1])
            preds = self.mlp(conv_out).squeeze(-1)  # (n_assets,)
        return preds.cpu().numpy()

    def remember(self, *args, **kwargs):
        pass

    def learn(self, windows: np.ndarray, epochs: int, batch_size: int):
        """
        Train the CNN+MLP on next-step returns per asset.
        windows: np.ndarray of shape (T, n_assets, lookback, n_features)
        """
        from torch.utils.data import DataLoader, TensorDataset
        # Prepare per-asset samples
        T_dim, n_assets, lookback, n_feat = windows.shape
        # compute next-step returns (feature 0 = price)
        prices = windows[:, :, -1, 0]
        ret = np.zeros_like(prices, dtype=np.float32)
        ret[:-1] = (prices[1:] / prices[:-1]) - 1.0
        # flatten windows and returns: (T * n_assets)
        data = windows.reshape(-1, lookback, n_feat).astype(np.float32)
        targets = ret.reshape(-1).astype(np.float32)
        ds = TensorDataset(T.tensor(data), T.tensor(targets))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # training loop
        for epoch in range(epochs):
            self.cnn.train()
            self.mlp.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                # xb: (batch, lookback, n_feat) -> (batch, n_feat, lookback)
                x_in = xb.permute(0, 2, 1)
                conv_out = self.cnn(x_in).squeeze(-1)
                preds = self.mlp(conv_out).squeeze(-1)
                loss = self.criterion(preds, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg = total_loss / len(ds)
            print(f"ShortTermPredictorAgent Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")

    def save_models(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save({'cnn': self.cnn.state_dict(), 'mlp': self.mlp.state_dict()}, path)

    def load_models(self, path: str):
        ckpt = T.load(path, map_location=self.device)
        self.cnn.load_state_dict(ckpt['cnn'])
        self.mlp.load_state_dict(ckpt['mlp'])
    
    def choose_action(self, observation):
        """
        Not implemented for specialized predictor.
        """
        raise NotImplementedError("ShortTermPredictorAgent does not support choose_action")


class LongTermPredictorAgent(BaseAgent):
    """
    Predicts longer-horizon trends using full lookback (e.g., Transformer or deep RNN).
    """
    def __init__(self, lookback: int, n_assets: int, n_features: int,
                 hidden_size: int = 128, lr: float = 1e-3):
        import torch.nn as nn
        # GRU-based long-term predictor per asset
        self.lookback = lookback
        self.n_assets = n_assets
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        # MLP head: hidden_size -> hidden_size//2 -> 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.gru.to(self.device)
        self.mlp.to(self.device)
        self.optimizer = T.optim.Adam(list(self.gru.parameters()) + list(self.mlp.parameters()), lr=lr)
        self.criterion = nn.MSELoss()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict multi-step returns for each asset over a longer horizon.
        Returns an array of shape (n_assets,).
        Default stub returns zeros.
        """
        self.gru.eval()
        self.mlp.eval()
        # observation shape: (n_assets, lookback, n_features)
        x = T.tensor(observation, dtype=T.float32).to(self.device)
        with T.no_grad():
            out, h = self.gru(x)  # out unused
            emb = h[-1]  # shape (1, batch, hidden_size) if num_layers=1
            # if batch dimension is assets-first
            if emb.dim() == 2:
                emb = emb
            # emb shape (n_assets, hidden_size)? adapt
            preds = self.mlp(emb).squeeze(-1)
        return preds.cpu().numpy()

    def choose_action(self, observation):
        raise NotImplementedError

    def remember(self, *args, **kwargs):
        pass
    
    def learn(self, windows: np.ndarray, epochs: int, batch_size: int):
        """
        Train the GRU+MLP on long-horizon returns per asset.
        """
        from torch.utils.data import DataLoader, TensorDataset
        T_dim, n_assets, lookback, n_feat = windows.shape
        # horizon = lookback
        H = lookback
        prices = windows[:, :, -1, 0]
        ret = np.zeros_like(prices, dtype=np.float32)
        if T_dim > H:
            ret[:-H] = (prices[H:] / prices[:-H]) - 1.0
        data = windows.reshape(-1, lookback, n_feat).astype(np.float32)
        targets = ret.reshape(-1).astype(np.float32)
        ds = TensorDataset(T.tensor(data), T.tensor(targets))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            self.gru.train()
            self.mlp.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out, h = self.gru(xb)  # xb: (batch, lookback, n_feat)
                emb = h[-1]  # (batch, hidden_size)
                preds = self.mlp(emb).squeeze(-1)
                loss = self.criterion(preds, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg = total_loss / len(ds)
            print(f"LongTermPredictorAgent Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")
    
    def save_models(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save({'gru': self.gru.state_dict(), 'mlp': self.mlp.state_dict()}, path)
    
    def load_models(self, path: str):
        ckpt = T.load(path, map_location=self.device)
        self.gru.load_state_dict(ckpt['gru'])
        self.mlp.load_state_dict(ckpt['mlp'])


class SentimentAgent(BaseAgent):
    """
    Processes sentiment or macro features for additional bias.
    """
    def __init__(self, n_assets: int, n_sent_features: int,
                 hidden_dims: list = [32, 16], lr: float = 1e-3):
        import torch.nn as nn
        # MLP-based sentiment processor per asset
        self.n_assets = n_assets
        self.n_sent = n_sent_features
        # Build MLP: n_sent -> hidden_dims -> 1
        layers = []
        prev = n_sent_features
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.mlp.to(self.device)
        self.optimizer = T.optim.Adam(self.mlp.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Stub: return zeros for sentiment bias (feature not part of observation).
        Observation may be any shape starting with n_assets.
        """
        n_assets = observation.shape[0]
        return np.zeros(n_assets, dtype=float)

    def choose_action(self, observation):
        raise NotImplementedError

    def remember(self, *args, **kwargs):
        pass

    def learn(self, windows: np.ndarray, sentiment_windows: np.ndarray, epochs: int, batch_size: int):
        """
        Train the sentiment MLP on sentiment features to predict next-step returns.
        windows: np.ndarray of shape (T, n_assets, lookback, n_features)
        sentiment_windows: np.ndarray of shape (T, n_assets, n_sent_features)
        """
        from torch.utils.data import DataLoader, TensorDataset
        # Compute next-step returns from price in windows (feature idx 0)
        prices = windows[:, :, -1, 0]
        ret = np.zeros_like(prices, dtype=np.float32)
        ret[:-1] = (prices[1:] / prices[:-1]) - 1.0
        # Flatten data: (T * n_assets)
        T_dim, n_assets, n_sent = sentiment_windows.shape
        X = sentiment_windows.reshape(-1, n_sent).astype(np.float32)
        y = ret.reshape(-1).astype(np.float32)
        ds = TensorDataset(T.tensor(X), T.tensor(y))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # Training loop
        for epoch in range(epochs):
            self.mlp.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.mlp(xb).squeeze(-1)
                loss = self.criterion(preds, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg = total_loss / len(ds)
            print(f"SentimentAgent Epoch {epoch+1}/{epochs}, Loss: {avg:.6f}")

    def save_models(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        T.save(self.mlp.state_dict(), path)

    def load_models(self, path: str):
        self.mlp.load_state_dict(T.load(path, map_location=self.device))