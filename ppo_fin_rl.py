import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.edge_indices = []  # New

        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done, edge_index):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.edge_indices.append(edge_index)  # Store edge_index

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (np.array(self.states), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), np.array(self.edge_indices), batches)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.edge_indices = []


class ActorNetwork(nn.Module):
    def __init__(self, n_stocks, n_features, hidden_size, num_heads, gru_hidden_size, alpha):
        """
        Args:
            n_stocks (int): Number of stocks (nodes).
            features (list): List of feature names (e.g., ["Open", "Close", ...]).
            hidden_size (int): Hidden size for GATv2 output per head.
            num_heads (int): Number of attention heads.
            gru_hidden_size (int): Hidden size for GRU.
            alpha (float): Learning rate.
        """
        super(ActorNetwork, self).__init__()
        self.n_stocks = n_stocks  # Number of stocks (nodes)
        self.n_features = n_features  # Number of features per stock

        # GATv2 layer (learn relationships between stocks)
        self.gatv2 = GATv2Conv(
            in_channels=self.n_features,  # Number of input features per stock
            out_channels=hidden_size,  # Output size for each attention head
            heads=num_heads,  # Number of attention heads
            concat=True,  # Concatenate the heads' outputs
            dropout=0.1
        )

        # GRU for sequential feature extraction
        self.gru = nn.GRU(
            input_size=hidden_size * num_heads,  # GATv2 output dimension (after concatenation)
            hidden_size=gru_hidden_size,  # GRU hidden size
            batch_first=True  # GRU expects batch as the first dimension
        )

        # Fully connected layers for policy output (actor)
        self.policy_fc = nn.Linear(gru_hidden_size, n_stocks)  # One action for each stock
        self.log_std = nn.Parameter(T.zeros(n_stocks))  # Learnable log standard deviation

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, edge_index):
        """
        Forward pass of the Actor network.
        Args:
            x: Node features (shape: [n_stocks, n_features]).
            edge_index: Graph connections (shape: [2, num_edges]).
        Returns:
            mean: Mean of the action distribution (shape: [n_stocks]).
            std: Standard deviation of the action distribution.
        """
        # Apply GATv2 to node features (shape: [n_stocks, hidden_size * num_heads])
        print(x.shape, edge_index.shape)
        print(x)
        # x = x.reshape(self.n_stocks, self.n_features)

        gat_output = self.gatv2(x, edge_index) # .unsqueeze(0)  # Add batch dimension [1, n_stocks, features]
        print(gat_output)

        # GRU for time-series modeling
        gru_output, _ = self.gru(gat_output)  # Shape: [1, seq_len (n_stocks), gru_hidden_size]
        last_gru_output = gru_output[:, -1, :]  # Take last time step [1, gru_hidden_size]

        # Policy output (mean of the normal distribution)
        mean = self.policy_fc(last_gru_output)
        log_std = torch.clamp(self.log_std, min=-10, max=2)  # Prevent too large/small values
        std = log_std.exp()  # Convert log standard deviation to std deviation

        return mean.squeeze(), std.squeeze()


class CriticNetwork(nn.Module):
    def __init__(self, n_stocks, n_features, hidden_size, num_heads, gru_hidden_size, alpha):
        """
        Args:
            n_stocks (int): Number of stocks (nodes).
            features (list): List of feature names (e.g., ["Open", "Close", ...]).
            hidden_size (int): Hidden size for GATv2 output per head.
            num_heads (int): Number of attention heads.
            gru_hidden_size (int): Hidden size for GRU.
            alpha (float): Learning rate.
        """
        super(CriticNetwork, self).__init__()
        self.n_stocks = n_stocks
        self.n_features = n_features

        # GATv2 layer for feature learning
        self.gatv2 = GATv2Conv(
            in_channels=self.n_features,
            out_channels=hidden_size,
            heads=num_heads,
            concat=True,
            dropout=0.1
        )

        # GRU for sequential feature extraction
        self.gru = nn.GRU(
            input_size=hidden_size * num_heads,
            hidden_size=gru_hidden_size,
            batch_first=True
        )

        # Fully connected layer for value estimation
        self.value_fc = nn.Linear(gru_hidden_size, 1)  # Outputs a single value (V(s))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, edge_index):
        """
        Forward pass of the Critic network.
        Args:
            x: Node features (shape: [n_stocks, n_features]).
            edge_index: Graph connections (shape: [2, num_edges]).
        Returns:
            value: Estimated state value.
        """
        # GATv2 layer
        gat_output = self.gatv2(x, edge_index).unsqueeze(0)  # Add batch dimension

        # GRU layer
        gru_output, _ = self.gru(gat_output)  # Shape: [1, n_stocks, gru_hidden_size]
        last_gru_output = gru_output[:, -1, :]  # Last time step

        # Value estimation
        value = self.value_fc(last_gru_output).squeeze()  # Single value output

        return value


# Chooses actions
# Saves data to PPOMemory
# Iterates over the available batches each game
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):

        # Discount factor
        self.gamma = gamma

        # allows for explanation without going too far off policy
        self.policy_clip = policy_clip

        self.n_stocks = n_actions
        self.n_features = input_dims

        # number of iteration over batches
        self.n_epochs = n_epochs

        # used to calculate the discount factor
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(
            n_stocks=self.n_stocks,
            n_features=self.n_features,
            hidden_size=128,
            num_heads=4,
            gru_hidden_size=256,
            alpha=0.0003
        )

        self.critic = CriticNetwork(
            n_stocks=self.n_stocks,
            n_features=self.n_features,
            hidden_size=128,
            num_heads=4,
            gru_hidden_size=256,
            alpha=0.0003
        )

        # initialize your memory to handle a full batch of data
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, rewards, done):
        self.memory.store_memory(state, action, probs, vals, rewards, done)

    def save_models(self):
        print('--saving models--')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('--loading models--')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, edge_index):
        if observation is None or len(observation) == 0:
            raise ValueError(f"Invalid observation: {observation}. Did you forget to reset the environment?")

        # Convert observation to tensor and move to device (no reshaping)
        state = T.tensor(observation, dtype=T.float32).to(self.actor.device)
        edge_index = edge_index.to(self.actor.device)


        # Forward pass through actor and critic
        mean, std = self.actor(state, edge_index)  # Actor's output for action distribution
        dist = T.distributions.Normal(mean, std)

        # Sample action from normal distribution
        raw_action = dist.sample()
        action = T.sigmoid(raw_action)  # Normalize to [0, 1]

        # Log probability of sampled action
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # Critic's value estimation
        value = self.critic(state, edge_index)

        return action.cpu().detach().numpy(), log_prob.cpu().detach().item(), value.cpu().detach().item()

    def learn(self):
        for epoch in range(self.n_epochs):
            (
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, dones_arr, edge_indices_arr,  # Add edge_indices
            batches) = self.memory.generate_batches()

            # Convert arrays to tensors
            values = T.tensor(vals_arr).to(self.actor.device)
            rewards = T.tensor(reward_arr).to(self.actor.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            dones = T.tensor(dones_arr).to(self.actor.device)
            advantage = T.zeros_like(rewards).to(self.actor.device)

            # Generalized Advantage Estimation (GAE)
            last_advantage = 0
            for t in reversed(range(len(reward_arr))):
                if t == len(reward_arr) - 1:
                    next_value = 0  # Last step has no next state value
                else:
                    next_value = values[t + 1]

                delta = reward_arr[t] + self.gamma * next_value * (1 - dones_arr[t]) - values[t]
                advantage[t] = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones_arr[t])
                last_advantage = advantage[t]

            # Normalize the advantage for stability
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            returns = advantage + values

            for batch_idx, batch in enumerate(batches):
                # Get batch tensors
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # Add edge_index for the current batch
                edge_index_batch = T.tensor(edge_indices_arr[batch], dtype=T.long).to(self.actor.device)

                # Actor forward pass (with edge_index)
                mean, std = self.actor(states, edge_index_batch)
                dist = T.distributions.Normal(mean, std)

                # Critic forward pass (with edge_index)
                critic_value = self.critic(states, edge_index_batch).squeeze()

                # Calculate new log probabilities
                new_probs = dist.log_prob(actions).sum(dim=-1)

                # PPO clipped objective
                prob_ratio = T.exp(new_probs - old_probs)
                weighted_probs = advantage[batch] * prob_ratio
                clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, clipped_probs).mean()

                # Critic loss (mean squared error)
                critic_loss = ((returns[batch] - critic_value) ** 2).mean()

                # Entropy regularization
                entropy_loss = dist.entropy().mean()
                actor_loss -= 0.01 * entropy_loss  # Encourage exploration

                # Regularization on std
                log_std_reg_penalty = 0.01 * (dist.scale.log() ** 2).mean()  # Penalize large log_std
                total_loss = actor_loss + 0.5 * critic_loss + log_std_reg_penalty

                # Logging
                logger.debug(f"Epoch: {epoch}, Batch: {batch_idx}, Actor Loss: {actor_loss.item():.4f}, "
                             f"Critic Loss: {critic_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}, "
                             f"Total Loss: {total_loss.item():.4f}")

                # Optimize
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # Clear memory after learning
        self.memory.clear_memory()




