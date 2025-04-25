import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from agents.base import BaseAgent
from GruVTwo import PPOMemory

class MLPActorCritic(nn.Module):
    """
    MLP-based actor-critic network with Dirichlet policy.
    """
    def __init__(self, input_dim, hidden_dims, n_assets, lr):
        super(MLPActorCritic, self).__init__()
        # Build MLP for shared features
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.shared = nn.Sequential(*layers)
        # Policy head: Dirichlet concentrations
        self.policy_head = nn.Linear(prev, n_assets + 1)
        # Value head: scalar
        self.value_head = nn.Linear(prev, 1)
        # Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.shared(x)
        conc = F.softplus(self.policy_head(x)) + 1e-8
        dist = T.distributions.Dirichlet(conc)
        value = self.value_head(x)
        return dist, value.squeeze(-1)

class MLPAgent(BaseAgent):
    """
    PPO agent using an MLP-based actor-critic.
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
        grad_norm: float
    ):
        # Dimensions
        input_dim = n_assets * lookback * (n_features + 2)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.entropy_coef = entropy_coef
        self.grad_norm = grad_norm
        self.n_epochs = n_epochs
        # Actor-Critic network
        self.ac = MLPActorCritic(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_assets=n_assets,
            lr=alpha
        )
        # Memory
        self.memory = PPOMemory(batch_size)

    def choose_action(self, observation):
        # observation: numpy array, convert to tensor
        x = T.tensor(observation.flatten(), dtype=T.float32).unsqueeze(0).to(self.ac.device)
        dist, value = self.ac(x)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def remember(self, observation, action, log_prob, value, reward, done):
        self.memory.store_memory(observation, action, log_prob, value, reward, done)

    def learn(self):
        """
        Perform PPO update for MLP agent using collected memory.
        """
        # Retrieve batches
        state_arr, action_arr, old_log_probs_arr, vals_arr, reward_arr, dones_arr, batches = \
            self.memory.generate_batches()
        # Convert to tensors
        device = self.ac.device
        values = T.tensor(vals_arr, dtype=T.float32).to(device)
        rewards = T.tensor(reward_arr, dtype=T.float32).to(device)
        dones = T.tensor(dones_arr, dtype=T.float32).to(device)
        # Compute advantages via GAE
        advantage = T.zeros_like(rewards).to(device)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage[t] = delta + self.gamma * self.gae_lambda * last_adv * (1 - dones[t])
            last_adv = advantage[t]
        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        returns = advantage + values

        # PPO updates
        for epoch in range(self.n_epochs):
            for batch in batches:
                # Prepare batch
                states = T.tensor(state_arr[batch], dtype=T.float32).to(device)
                # Flatten states for MLP input
                states = states.view(states.shape[0], -1)
                old_log_probs = T.tensor(old_log_probs_arr[batch], dtype=T.float32).to(device)
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(device)
                # Forward pass
                dist, critic_value = self.ac(states)
                critic_value = critic_value.squeeze()
                entropy = dist.entropy().mean()
                # New log probs
                new_log_probs = dist.log_prob(actions)
                # Sum across dims if needed
                if new_log_probs.dim() > old_log_probs.dim():
                    new_log_probs = new_log_probs.sum(dim=-1)
                # PPO ratio
                ratio = T.exp(new_log_probs - old_log_probs)
                # Surrogate losses
                surr1 = ratio * advantage[batch]
                surr2 = T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(surr1, surr2).mean() - self.entropy_coef * entropy
                critic_loss = F.mse_loss(critic_value, returns[batch])
                total_loss = actor_loss + 0.5 * critic_loss
                # Optimize
                self.ac.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.ac.parameters(), self.grad_norm)
                self.ac.optimizer.step()
        # Clear memory
        self.memory.clear_memory()

    def save_models(self, path: str):
        T.save(self.ac.state_dict(), path + '_mlp.pt')

    def load_models(self, path: str):
        self.ac.load_state_dict(T.load(path + '_mlp.pt'))