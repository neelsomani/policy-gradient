# Reference: https://colab.research.google.com/github/MrSyee/pg-is-all-you-need/blob/master/01.A2C.ipynb

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ReinforcePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.use_baseline = False
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.logstd_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # Returns mean and the log of the std_dev - a common trick to ensure non-negativity
        h = self.net(state)
        # The mean is allowed to vary from -2 to 2
        mu = torch.tanh(self.mu_head(h)) * 2
        log_std_dev = F.softplus(self.logstd_head(h))
        return mu, log_std_dev
    
    def sample(self, state):
        """Get action from policy"""
        mu, log_std_dev = self.forward(state)
        std_dev = torch.exp(log_std_dev)
        dist = Normal(mu, std_dev)
        action = dist.sample()
        return action, dist


class A2CPolicy(ReinforcePolicy):
     def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.use_baseline = True


class Value(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.out(x)


class PendulumRL:
    """
    Main RL class for training on Pendulum-v1 environment
    """
    def __init__(self, policy_cls, lr=3e-4, gamma=.9):
        self.env = gym.make("Pendulum-v1")
        self.gamma = gamma

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.actor = policy_cls(state_dim, action_dim)
        self.value = Value(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    def select_action(self, state, is_training):
        """Select an action from the input state."""
        state = torch.FloatTensor(state)
        action, dist = self.actor.sample(state)
        selected_action = dist.mean if not is_training else action

        if is_training:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            return selected_action.clamp(-2.0, 2.0).detach().numpy(), state, log_prob

        return selected_action.clamp(-2.0, 2.0).detach().numpy()

    def update_policy(self, state, log_prob, next_state, reward, done):
        mask = 1 - done
        next_state = torch.FloatTensor(next_state)
        # Need the value model for TD(0) even with no baseline
        targ_value = reward + self.gamma * self.value(next_state) * mask
        pred_value = self.value(state)

        # Optimize value model
        value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        returns = targ_value.detach() if not self.actor.use_baseline else (targ_value - pred_value).detach()

        policy_loss = -returns * log_prob

        # Optimize actor model
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def evaluate(self, num_episodes=5):
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            score = 0.0
            while not done:
                action = self.select_action(state, is_training=False)
                state, reward, terminated, truncated, _ = self.env.step(action.astype(np.float32))
                done = bool(terminated or truncated)
                score += reward
            print(f"Test {ep+1}: return {score}")

    def train(self, num_episodes, log_interval=200):
        state, _ = self.env.reset()
        score = 0.0

        for step in range(1, num_episodes + 1):
            action, state_tensor, log_prob = self.select_action(state, is_training=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action.astype(np.float32))
            done = bool(terminated or truncated)
            self.update_policy(state_tensor, log_prob, next_state, reward, done)
            state = next_state
            score += reward

            if done:
                state, _ = self.env.reset()
                if step % log_interval == 0:
                    print(f"Step {step}: return {score}")
                score = 0.0

        self.env.close()


if __name__ == "__main__":
    num_episodes = 100000
    agent = PendulumRL(A2CPolicy)
    agent.train(num_episodes, log_interval=2000)
    agent.evaluate(num_episodes=3)
