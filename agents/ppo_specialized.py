import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from agents.base import BaseAgent
from agents.specialized import ShortTermPredictorAgent, LongTermPredictorAgent, AnomalyAgent
from agents.mlp_agent import MLPAgent
from GruVTwo import PPOMemory

class ShortTermPPOAgent(BaseAgent):
    """
    PPO-based short-term predictor agent that uses a pretrained CNN backbone.
    """
    def __init__(
        self,
        lookback: int,
        n_assets: int,
        n_features: int,
        conv_dims: list,
        hidden_dim: int,
        lr: float,
        batch_size: int,
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        policy_clip: float,
        entropy_coef: float,
        grad_norm: float,
        pretrained_path: str = None,
    ):
        # Load pretrained supervised CNN backbone
        self.cnn_agent = ShortTermPredictorAgent(
            lookback=lookback,
            n_assets=n_assets,
            n_features=n_features,
            conv_dims=conv_dims,
            hidden_dim=hidden_dim,
            lr=lr
        )
        if pretrained_path:
            try:
                self.cnn_agent.load_models(pretrained_path)
            except Exception:
                pass
        self.cnn = self.cnn_agent.cnn
        # Freeze CNN or optionally fine-tune
        # self.cnn.requires_grad_(False)
        # Feature dimension: n_assets * last conv channel
        feat_dim = n_assets * conv_dims[-1]
        self.policy_head = nn.Linear(feat_dim, n_assets + 1)
        self.value_head = nn.Linear(feat_dim, 1)
        # PPO memory and hyperparams
        self.memory = PPOMemory(batch_size)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.entropy_coef = entropy_coef
        self.grad_norm = grad_norm
        self.n_epochs = n_epochs
        # Optimizer
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.cnn.to(self.device)
        self.policy_head.to(self.device)
        self.value_head.to(self.device)
        self.optimizer = T.optim.Adam(
            list(self.cnn.parameters()) +
            list(self.policy_head.parameters()) +
            list(self.value_head.parameters()),
            lr=lr
        )

    def choose_action(self, observation):
        # observation: np.ndarray shape (n_assets, lookback, n_features+2)
        # Extract market features only
        obs = observation[:, :, :self.cnn_agent.n_features]
        x = T.tensor(obs, dtype=T.float32).permute(0, 2, 1).to(self.device)
        with T.no_grad():
            conv_out = self.cnn(x).squeeze(-1)  # (n_assets, C)
        features = conv_out.flatten().unsqueeze(0)  # (1, feat_dim)
        dist = T.distributions.Dirichlet(
            F.softplus(self.policy_head(features)) + 1e-8
        )
        action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action).sum().item()
        value = self.value_head(features).item()
        return action.cpu().numpy(), log_prob, value

    def remember(self, observation, action, log_prob, value, reward, done):
        self.memory.store_memory(observation, action, log_prob, value, reward, done)

    def learn(self):
        # Retrieve stored experiences
        state_arr, action_arr, old_log_probs, vals, rewards, dones, batches = \
            self.memory.generate_batches()
        device = self.device
        # Convert to tensors
        values = T.tensor(vals, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        dones = T.tensor(dones, dtype=T.float32).to(device)
        # Compute advantages with GAE
        advantages = T.zeros_like(rewards).to(device)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_adv * (1 - dones[t])
            last_adv = advantages[t]
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        eps = 1e-8
        # PPO policy and value updates
        for _ in range(self.n_epochs):
            for batch_idx in batches:
                # Prepare batch data
                obs_batch = state_arr[batch_idx]  # list of np arrays
                # Stack and extract market features
                obs_np = np.stack(obs_batch, axis=0)  # (B, n_assets, lookback, n_feat+2)
                market = obs_np[:, :, :, :self.cnn_agent.n_features]  # drop pf features
                # Convert to tensor and reshape for CNN
                x = T.tensor(market, dtype=T.float32).to(device)               # (B, n_assets, lookback, n_feat)
                x = x.permute(0, 1, 3, 2)                                      # (B, n_assets, n_feat, lookback)
                B, A, Ff, L = x.shape
                x = x.reshape(B * A, Ff, L)                                     # (B*A, n_feat, lookback)
                conv_out = self.cnn(x).squeeze(-1)                              # (B*A, C)
                feat = conv_out.reshape(B, -1)                                  # (B, A*C)
                # Policy distribution
                concentrations = F.softplus(self.policy_head(feat)) + eps       # (B, A+1)
                dist = T.distributions.Dirichlet(concentrations)
                # New log probs and entropy
                actions = T.tensor(action_arr[batch_idx], dtype=T.float32).to(device)
                new_log_probs = dist.log_prob(actions)
                if new_log_probs.dim() > 1:
                    new_log_probs = new_log_probs.sum(dim=-1)
                entropy = dist.entropy().mean()
                # Ratio for clipped surrogate
                old_lp = T.tensor(old_log_probs[batch_idx], dtype=T.float32).to(device)
                ratio = T.exp(new_log_probs - old_lp)
                adv_batch = advantages[batch_idx]
                surr1 = ratio * adv_batch
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv_batch
                actor_loss = -T.min(surr1, surr2).mean() - self.entropy_coef * entropy
                # Critic loss
                ret_batch = returns[batch_idx]
                value_preds = self.value_head(feat).squeeze(-1)
                critic_loss = F.mse_loss(value_preds, ret_batch)
                # Total loss
                loss = actor_loss + 0.5 * critic_loss
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(list(self.cnn.parameters()) + 
                                           list(self.policy_head.parameters()) + 
                                           list(self.value_head.parameters()),
                                           self.grad_norm)
                self.optimizer.step()
        # Clear memory after update
        self.memory.clear_memory()

    def save_models(self, path: str):
        # Save backbone and heads
        T.save({
            'cnn': self.cnn.state_dict(),
            'policy': self.policy_head.state_dict(),
            'value': self.value_head.state_dict()
        }, path)

    def load_models(self, path: str):
        ckpt = T.load(path, map_location=self.device)
        self.cnn.load_state_dict(ckpt['cnn'])
        self.policy_head.load_state_dict(ckpt['policy'])
        self.value_head.load_state_dict(ckpt['value'])
        # Optional: load CNN backbone
        try:
            self.cnn.load_state_dict(ckpt['cnn'])
        except KeyError:
            pass

class LongTermPPOAgent(BaseAgent):
    """
    PPO-based long-term predictor agent that uses a pretrained GRU backbone.
    """
    def __init__(
        self,
        lookback: int,
        n_assets: int,
        n_features: int,
        hidden_size: int,
        lr: float,
        batch_size: int,
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        policy_clip: float,
        entropy_coef: float,
        grad_norm: float,
        pretrained_path: str = None,
    ):
        # Load pretrained supervised GRU backbone
        self.long_agent = LongTermPredictorAgent(
            lookback=lookback,
            n_assets=n_assets,
            n_features=n_features,
            hidden_size=hidden_size,
            lr=lr
        )
        if pretrained_path:
            try:
                self.long_agent.load_models(pretrained_path)
            except Exception:
                pass
        self.gru = self.long_agent.gru
        # Feature dim: n_assets * hidden_size
        feat_dim = n_assets * hidden_size
        self.policy_head = nn.Linear(feat_dim, n_assets + 1)
        self.value_head = nn.Linear(feat_dim, 1)
        # PPO memory & hyperparams
        self.memory = PPOMemory(batch_size)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.entropy_coef = entropy_coef
        self.grad_norm = grad_norm
        self.n_epochs = n_epochs
        # Device and optimizer
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.gru.to(self.device)
        self.policy_head.to(self.device)
        self.value_head.to(self.device)
        self.optimizer = T.optim.Adam(
            list(self.gru.parameters()) +
            list(self.policy_head.parameters()) +
            list(self.value_head.parameters()),
            lr=lr
        )

    def choose_action(self, observation):
        # observation: (n_assets, lookback, n_features+2)
        obs = observation[:, :, :self.long_agent.n_features]
        x = T.tensor(obs, dtype=T.float32).to(self.device)
        # x shape: (n_assets, lookback, n_features)
        with T.no_grad():
            out, h = self.gru(x)
            emb = h[-1]  # shape (n_assets, hidden_size)
        feat = emb.flatten().unsqueeze(0)  # (1, n_assets*hidden_size)
        concentrations = F.softplus(self.policy_head(feat)) + 1e-8
        dist = T.distributions.Dirichlet(concentrations)
        action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action).sum().item()
        value = self.value_head(feat).item()
        return action.cpu().numpy(), log_prob, value

    def remember(self, observation, action, log_prob, value, reward, done):
        self.memory.store_memory(observation, action, log_prob, value, reward, done)

    def learn(self):
        # Similar PPO update as in ShortTermPPOAgent but using GRU features
        state_arr, action_arr, old_log_probs, vals, rewards, dones, batches = \
            self.memory.generate_batches()
        device = self.device
        values = T.tensor(vals, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        dones = T.tensor(dones, dtype=T.float32).to(device)
        # Compute advantages
        advantages = T.zeros_like(rewards).to(device)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_adv * (1 - dones[t])
            last_adv = advantages[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        eps = 1e-8
        for _ in range(self.n_epochs):
            for batch_idx in batches:
                obs_batch = state_arr[batch_idx]
                obs_np = np.stack([o[:, :, :self.long_agent.n_features] for o in obs_batch], axis=0)
                # Process all assets together
                x = T.tensor(obs_np, dtype=T.float32).to(device)  # (B, A, L, F)
                B, A, L, Ff = x.shape
                x = x.reshape(B*A, L, Ff)
                out, h = self.gru(x)
                emb = h[-1]  # (B*A, hidden_size)
                feat = emb.reshape(B, -1)  # (B, A*hidden_size)
                dist = T.distributions.Dirichlet(F.softplus(self.policy_head(feat)) + eps)
                actions = T.tensor(action_arr[batch_idx], dtype=T.float32).to(device)
                new_log_probs = dist.log_prob(actions)
                if new_log_probs.dim() > old_log_probs[batch_idx].ndim:
                    new_log_probs = new_log_probs.sum(dim=-1)
                entropy = dist.entropy().mean()
                old_lp = T.tensor(old_log_probs[batch_idx], dtype=T.float32).to(device)
                ratio = T.exp(new_log_probs - old_lp)
                adv_batch = advantages[batch_idx]
                surr1 = ratio * adv_batch
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv_batch
                actor_loss = -T.min(surr1, surr2).mean() - self.entropy_coef * entropy
                value_preds = self.value_head(feat).squeeze(-1)
                critic_loss = F.mse_loss(value_preds, returns[batch_idx])
                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(list(self.gru.parameters()) + 
                                           list(self.policy_head.parameters()) + 
                                           list(self.value_head.parameters()),
                                           self.grad_norm)
                self.optimizer.step()
        self.memory.clear_memory()

class AnomalyPPOAgent(BaseAgent):
    """
    PPO-based anomaly gating agent that uses a pretrained autoencoder encoder.
    """
    def __init__(
        self,
        lookback: int,
        n_assets: int,
        n_features: int,
        hidden_dims: list,
        lr: float,
        batch_size: int,
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        policy_clip: float,
        entropy_coef: float,
        grad_norm: float,
        pretrained_path: str = None,
    ):
        # Load pretrained anomaly autoencoder
        self.anom_agent = AnomalyAgent(
            lookback=lookback,
            n_assets=n_assets,
            n_features=n_features,
            hidden_dims=hidden_dims,
            lr=lr
        )
        if pretrained_path:
            try:
                self.anom_agent.load_models(pretrained_path)
            except Exception:
                pass
        # Extract encoder part
        n_enc = len(hidden_dims) * 2
        enc_modules = list(self.anom_agent.autoencoder.children())[:n_enc]
        self.encoder = nn.Sequential(*enc_modules)
        # Feature dim: n_assets * last hidden_dim
        feat_dim = n_assets * hidden_dims[-1]
        self.policy_head = nn.Linear(feat_dim, n_assets + 1)
        self.value_head = nn.Linear(feat_dim, 1)
        # PPO memory & hyperparams
        self.memory = PPOMemory(batch_size)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.entropy_coef = entropy_coef
        self.grad_norm = grad_norm
        self.n_epochs = n_epochs
        # Device and optimizer
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.policy_head.to(self.device)
        self.value_head.to(self.device)
        self.optimizer = T.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.policy_head.parameters()) +
            list(self.value_head.parameters()),
            lr=lr
        )

    def choose_action(self, observation):
        # observation: (n_assets, lookback, n_features+2)
        obs = observation[:, :, :self.anom_agent.n_features]
        # Flatten and normalize per asset
        x = obs.reshape(self.anom_agent.n_assets, -1).astype(np.float32)
        t_x = T.tensor(x, dtype=T.float32).to(self.device)
        mean = t_x.mean(dim=1, keepdim=True)
        std = t_x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (t_x - mean) / std
        with T.no_grad():
            emb = self.encoder(x_norm)
        # Flatten per-asset embeddings and prepare model input
        feat = emb.flatten().unsqueeze(0)
        # Compute Dirichlet concentration parameters and sanitize
        concentrations = F.softplus(self.policy_head(feat)) + 1e-8
        concentrations = T.nan_to_num(
            concentrations,
            nan=1e-8,
            neginf=1e-8,
            posinf=1e8
        )
        dist = T.distributions.Dirichlet(concentrations)
        # Sample action and evaluate log-prob and value
        action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action).sum().item()
        value = self.value_head(feat).item()
        return action.cpu().numpy(), log_prob, value

    def remember(self, observation, action, log_prob, value, reward, done):
        self.memory.store_memory(observation, action, log_prob, value, reward, done)

    def learn(self):
        # PPO update as in other agents
        state_arr, action_arr, old_log_probs, vals, rewards, dones, batches = \
            self.memory.generate_batches()
        device = self.device
        values = T.tensor(vals, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        dones = T.tensor(dones, dtype=T.float32).to(device)
        # Compute advantages
        adv = T.zeros_like(rewards).to(device)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            adv[t] = delta + self.gamma * self.gae_lambda * last_adv * (1 - dones[t])
            last_adv = adv[t]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = adv + values
        eps = 1e-8
        for _ in range(self.n_epochs):
            for batch_idx in batches:
                obs_batch = state_arr[batch_idx]
                obs_np = np.stack([o[:, :, :self.anom_agent.n_features] for o in obs_batch], axis=0)
                B, A, L, Ff = obs_np.shape
                x = T.tensor(obs_np.reshape(B*A, L*Ff), dtype=T.float32).to(device)
                # embeddings
                emb = self.encoder(x)
                feat = emb.reshape(B, -1)
                # Sanitize Dirichlet concentration parameters
                conc = F.softplus(self.policy_head(feat)) + eps
                conc = T.nan_to_num(conc, nan=eps, neginf=eps, posinf=1e8)
                dist = T.distributions.Dirichlet(conc)
                actions = T.tensor(action_arr[batch_idx], dtype=T.float32).to(device)
                new_lp = dist.log_prob(actions)
                if new_lp.dim() > old_log_probs[batch_idx].ndim:
                    new_lp = new_lp.sum(dim=-1)
                ent = dist.entropy().mean()
                old_lp = T.tensor(old_log_probs[batch_idx], dtype=T.float32).to(device)
                ratio = T.exp(new_lp - old_lp)
                adv_b = adv[batch_idx]
                s1 = ratio * adv_b
                s2 = T.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip) * adv_b
                actor_loss = -T.min(s1, s2).mean() - self.entropy_coef * ent
                values_pred = self.value_head(feat).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, returns[batch_idx])
                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + 
                                           list(self.policy_head.parameters()) + 
                                           list(self.value_head.parameters()),
                                           self.grad_norm)
                self.optimizer.step()
        self.memory.clear_memory()
    
    def save_models(self, path: str):
        """Save encoder, policy head, and value head state to a single checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            'encoder': self.encoder.state_dict(),
            'policy_head': self.policy_head.state_dict(),
            'value_head': self.value_head.state_dict()
        }
        T.save(ckpt, path)
    
    def load_models(self, path: str):
        """Load encoder, policy head, and value head state from checkpoint."""
        ckpt = T.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.policy_head.load_state_dict(ckpt['policy_head'])
        self.value_head.load_state_dict(ckpt['value_head'])

class SentimentPPOAgent(BaseAgent):
    """
    PPO-based sentiment agent wrapper that delegates to MLPAgent.
    """
    def __init__(
        self,
        n_assets: int,
        lookback: int,
        n_features: int,
        hidden_dims: list,
        alpha: float,
        batch_size: int,
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        policy_clip: float,
        entropy_coef: float,
        grad_norm: float,
        pretrained_path: str = None,
    ):
        from agents.mlp_agent import MLPAgent
        # Initialize MLPAgent with full observation flattening
        self.agent = MLPAgent(
            n_assets=n_assets,
            lookback=lookback,
            n_features=n_features,
            hidden_dims=hidden_dims,
            alpha=alpha,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            policy_clip=policy_clip,
            entropy_coef=entropy_coef,
            grad_norm=grad_norm
        )
        # Optional: load pretrained sentiment model (MLP checkpoint)
        if pretrained_path:
            try:
                self.agent.load_models(pretrained_path)
            except Exception:
                pass

    def choose_action(self, observation):
        return self.agent.choose_action(observation)

    def remember(self, observation, action, log_prob, value, reward, done):
        self.agent.remember(observation, action, log_prob, value, reward, done)

    def learn(self):
        self.agent.learn()

    def save_models(self, path: str):
        self.agent.save_models(path)

    def load_models(self, path: str):
        self.agent.load_models(path)