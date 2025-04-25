import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from agents.base import BaseAgent
from agents.specialized import (
    AnomalyAgent,
    ShortTermPredictorAgent,
    LongTermPredictorAgent,
    SentimentAgent,
)
from GruVTwo import PPOMemory

class AggregatorNetwork(nn.Module):
    """
    Neural network to aggregate sub-agent signals and output a policy distribution and value.
    """
    def __init__(self, input_dim: int, hidden_dims: list, n_experts: int, lr: float=1e-3):
        super(AggregatorNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.mlp = nn.Sequential(*layers)
        # Meta-policy: Gaussian logits per expert (will be softmax-ed in the agent)
        self.mu_head = nn.Linear(prev_dim, n_experts)
        # Learned log-std for exploration
        self.log_std = nn.Parameter(T.zeros(n_experts))
        # Value head for baseline
        self.value_head = nn.Linear(prev_dim, 1)
        # Optimizer for the aggregator network
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: T.Tensor):
        """
        Forward pass: returns (dist, value)
        where dist is a Normal distribution over expert logits.
        The meta-agent will softmax samples for gating weights.
        """
        h = self.mlp(x)
        mu = self.mu_head(h)
        std = T.exp(self.log_std).unsqueeze(0).expand_as(mu)
        dist = T.distributions.Normal(mu, std)
        value = self.value_head(h).squeeze(-1)
        return dist, value

class EnsembleAgent(BaseAgent):
    """
    Ensemble agent combining specialized sub-agents and an aggregator policy.
    """
    def __init__(
        self,
        n_assets: int,
        lookback: int,
        n_features: int,
        hidden_dims: list = [256, 128],
        alpha: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        entropy_coef: float = 0.01,
        grad_norm: float = 0.5,
        sub_hidden_dims: dict = None,
        sub_conv_dims: dict = None,
        n_sent_features: int = 1
    ):
        # Initialize specialized sub-agents
        # AnomalyAgent requires lookback, n_assets, n_features, hidden_dims, lr
        # Instantiate sub-agents with default dimensions
        # Sub-agent hyperparameters
        sub_hd = sub_hidden_dims or {}
        sub_cd = sub_conv_dims or {}
        n_sent = n_sent_features
        # Anomaly autoencoder dims
        a_hd = sub_hd.get('anomaly', hidden_dims)
        # Short-term predictor dims
        st_cd = sub_cd.get('short_term', [32, 32])
        st_hd = sub_hd.get('short_term', 16)
        # Long-term predictor dims
        lt_hd = sub_hd.get('long_term', 64)
        # Sentiment dims
        sn_hd = sub_hd.get('sentiment', [32, 16])
        self.sub_agents = [
            AnomalyAgent(
                lookback=lookback,
                n_assets=n_assets,
                n_features=n_features,
                hidden_dims=a_hd,
                lr=alpha
            ),
            ShortTermPredictorAgent(
                lookback=lookback,
                n_assets=n_assets,
                n_features=n_features,
                conv_dims=st_cd,
                hidden_dim=st_hd,
                lr=alpha
            ),
            LongTermPredictorAgent(
                lookback=lookback,
                n_assets=n_assets,
                n_features=n_features,
                hidden_size=lt_hd,
                lr=alpha
            ),
            SentimentAgent(
                n_assets=n_assets,
                n_sent_features=n_sent,
                hidden_dims=sn_hd,
                lr=alpha
            ),
        ]
        n_sub = len(self.sub_agents)
        # Store original market feature count for sub-agent slicing
        self.market_features = n_features
        # The meta-controller takes raw observations only
        raw_dim = n_assets * lookback * (n_features + 2)
        input_dim = raw_dim
        # Aggregator (meta-policy) network with gating over sub-agents
        self.aggregator = AggregatorNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_experts=n_sub,
            lr=alpha,
        )
        # Value head included in aggregator
        # PPO memory
        self.memory = PPOMemory(batch_size)
        # PPO hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.grad_norm = grad_norm

    def choose_action(self, observation: T.Tensor):
        """
        Given an observation, compute sub-agent features, then aggregator policy.
        Returns action (np.ndarray), log_prob (float), value (float).
        """
        # 1) Get each sub-agent's allocation action (n_assets+1 vector)
        sub_actions = []
        for sub in self.sub_agents:
            a, lp, val = sub.choose_action(observation)
            sub_actions.append(a)
        sub_actions = np.stack(sub_actions, axis=0)  # shape (n_experts, n_assets+1)
        # 2) Meta-policy: intake raw observation (flattened) and output gating weights
        if isinstance(observation, np.ndarray):
            raw_flat = observation.flatten()
        else:
            raw_flat = observation.detach().cpu().numpy().flatten()
        device = next(self.aggregator.parameters()).device
        x = T.tensor(raw_flat, dtype=T.float32).unsqueeze(0).to(device)
        dist, meta_value = self.aggregator(x)
        # Sample gating weights and compute log-prob
        # Sample raw gating logits via the reparameterization trick
        gating_raw = dist.rsample().squeeze(0)
        log_prob = dist.log_prob(gating_raw).sum(-1).item()
        # Normalize into weights via softmax
        gating = T.softmax(gating_raw, dim=-1)
        # 3) Blend sub-agent actions by gating weights
        gating_np = gating.cpu().numpy()
        final_action = np.tensordot(gating_np, sub_actions, axes=(0, 0))
        # 4) Return blended action, meta log-prob, and value estimate
        return final_action, log_prob, meta_value.item()

    def remember(self, observation, action, log_prob, value, reward, done):
        self.memory.store_memory(observation, action, log_prob, value, reward, done)

    def learn(self):
        """
        Perform PPO updates on the aggregator network using stored experiences.
        Sub-agents are frozen (no gradient updates).
        """
        # Retrieve batches from memory
        state_arr, action_arr, old_log_probs_arr, vals_arr, reward_arr, dones_arr, batches = \
            self.memory.generate_batches()
        device = next(self.aggregator.parameters()).device
        # Prepare returns and advantages via GAE
        values = T.tensor(vals_arr, dtype=T.float32).to(device)
        rewards = T.tensor(reward_arr, dtype=T.float32).to(device)
        dones = T.tensor(dones_arr, dtype=T.float32).to(device)
        advantage = T.zeros_like(rewards).to(device)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantage[t] = delta + self.gamma * self.gae_lambda * last_adv * (1 - dones[t])
            last_adv = advantage[t]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        returns = advantage + values

        # PPO update loop
        for epoch in range(self.n_epochs):
            for batch in batches:
                # Prepare batch data
                state_batch = state_arr[batch]  # shape: (b, n_assets, lookback, n_feat+2)
                # Build input tensor: raw features
                b = state_batch.shape[0]
                raw_flat = state_batch.reshape(b, -1)
                # Sub-agent predictions (use only market features)
                sub_feats = []
                # slice off portfolio channels; keep first self.market_features
                market_batch = state_batch[:, :, :, :self.market_features]
                for sub in self.sub_agents:
                    # predict returns numpy of shape (n_assets,)
                    feats = np.stack([sub.predict(obs) for obs in market_batch], axis=0)
                    sub_feats.append(feats)
                sub_flat = np.concatenate(sub_feats, axis=1)  # shape: (b, n_sub*n_assets)
                # Combine
                inp = np.concatenate([raw_flat, sub_flat], axis=1)
                x = T.tensor(inp, dtype=T.float32).to(device)
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(device)
                old_log_probs = T.tensor(old_log_probs_arr[batch], dtype=T.float32).to(device)
                # Forward
                dist, critic_value = self.aggregator(x)
                critic_value = critic_value.squeeze()
                entropy = dist.entropy().mean()
                # Log probs
                new_log_probs = dist.log_prob(actions)
                if new_log_probs.dim() > old_log_probs.dim():
                    new_log_probs = new_log_probs.sum(dim=-1)
                # Ratios and surrogate loss
                ratio = T.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantage[batch]
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(surr1, surr2).mean() - self.entropy_coef * entropy
                critic_loss = F.mse_loss(critic_value, returns[batch])
                loss = actor_loss + 0.5 * critic_loss
                # Optimize
                self.aggregator.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(self.aggregator.parameters(), self.grad_norm)
                self.aggregator.optimizer.step()
        # Clear memory
        self.memory.clear_memory()

    def save_models(self, path: str):
        T.save(self.aggregator.state_dict(), path + '_aggregator.pt')
        # Sub-agents not yet implemented

    def load_models(self, path: str):
        self.aggregator.load_state_dict(T.load(path + '_aggregator.pt'))