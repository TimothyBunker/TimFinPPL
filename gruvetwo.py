import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PPOMemory:
    # Class made to contain all of the data of our model
    # such as our batches of data
    # our agent data
    # and methods to clear and store it too
    # accessed by the agent
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (np.array(self.states), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):

    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='/PPL/tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        # Initialize your optimizer and allocate it to your model
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='/PPL/tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # normal linear forward pass, no need for categorical only one output
    # your value estimation
    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):

        # Discount factor
        self.gamma = gamma

        # allows for explanation without going too far off policy
        self.policy_clip = policy_clip

        # number of iteration over batches
        self.n_epochs = n_epochs

        # used to calculate the discount factor
        self.gae_lambda = gae_lambda

        # Your policy gradient
        self.actor = ActorNetwork(n_actions=n_actions, input_dims=input_dims, alpha=alpha)

        # Your value estimate gradient
        self.critic = CriticNetwork(input_dims=input_dims, alpha=alpha)

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

    def choose_action(self, observation):
        if observation is None or len(observation) == 0:
            raise ValueError(f"Invalid observation: {observation}. Did you forget to reset the environment?")

        # Ensure observation is converted to a PyTorch tensor and batched
        state = T.tensor(observation, dtype=T.float32).flatten().unsqueeze(0).to(self.actor.device)  # Add batch dimension
        print(f'state shape: {state.shape}')

        # Pass the state through the actor and critic networks
        dist = self.actor(state)  # Get action distribution
        value = self.critic(state)  # Get state value
        print(f'state: {state}')
        print(f'value: {value}')

        # Sample an action from the action distribution
        action = dist.sample()
        print(f'Action: {action}')
        # Log probability of the chosen action
        log_prob = dist.log_prob(action)
        print(f'log prob: {log_prob}')

        # Remove extra dimensions for scalar values
        action = action.squeeze().cpu().numpy()  # Convert to NumPy array if needed
        log_prob = log_prob.sum().cpu().item()
        value = value.squeeze().cpu().item()

        return action, log_prob, value

    def learn(self):
        for epoch in range(self.n_epochs):
            (state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, dones_arr,
             batches) = self.memory.generate_batches()

            # Convert arrays to tensors
            values = T.tensor(vals_arr).to(self.actor.device)
            rewards = T.tensor(reward_arr).to(self.actor.device)
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

            # Convert advantage and values to tensors
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # Normalize for stability
            returns = advantage + values  # Calculate returns

            for batch_idx, batch in enumerate(batches):
                # Get batch tensors
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)  # Forward pass through the actor
                critic_value = self.critic(states).squeeze()  # Forward pass through the critic

                # Calculate new log probabilities
                new_probs = dist.log_prob(actions)

                # PPO clipped objective
                prob_ratio = T.exp(new_probs - old_probs)
                weighted_probs = advantage[batch] * prob_ratio
                clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, clipped_probs).mean()

                # Critic loss (mean squared error)
                critic_loss = ((returns[batch] - critic_value) ** 2).mean()

                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss

                # Optimize
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)  # Gradient clipping for stability
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # Clear memory after learning
        self.memory.clear_memory()

