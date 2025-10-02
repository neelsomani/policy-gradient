"""
Reinforcement Learning Implementation for Pendulum-v1 Environment
"""
from collections import deque
import random
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


LOG_STD_MIN, LOG_STD_MAX = -3.0, 3.0  # Clamp for numerical stability


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
    
    def forward(self, state):
        raise NotImplementedError
    
    def sample(self, state):
        raise NotImplementedError


class DummyPolicy(Policy):
    """
    A simple dummy policy network for testing purposes.
    This policy takes random actions within the action space bounds.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DummyPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple neural network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output raw action values (will be scaled to [-2, 2] for Pendulum)
        action = torch.tanh(self.fc3(x))
        return action
    
    def sample(self, state):
        """Get action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.forward(state_tensor)
            # Scale to Pendulum action space [-2, 2]
            action = action * 2.0
            return action.squeeze(0).numpy()


class GaussianPolicy(Policy):
    """
    This policy outputs a distribution pi_{theta}(a | s), so we can compute the gradient.
    Normal(mean, std) -> tanh -> scale to env bounds
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.logstd_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        # Returns mean and the log of the std_dev - a common trick to ensure non-negativity
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.logstd_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    
    def sample(self, state):
        """Get action from policy"""
        if not torch.is_tensor(state):  # Boilerplate wrapping to make sure the type is right
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        action_raw = torch.tanh(z)
        # Scale to Pendulum action space [-2, 2]
        action = action_raw * 2.0
        
        # Solve for the correct log(pi(a | s)) using the change-of-variables
        log_p_z = -0.5 * (((z - mu) / std)**2 + 2*log_std + math.log(2*math.pi))
        log_p_z = log_p_z.sum(dim=-1, keepdim=True)
        log_det_tanh = torch.log(1 - action_raw.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_det_scale = -math.log(2.0) * mu.shape[-1]
        log_prob = log_p_z - log_det_tanh + log_det_scale
        return action.squeeze(0), log_prob.squeeze(0)


class PendulumRL:
    """
    Main RL class for training on Pendulum-v1 environment
    """
    
    def __init__(self, policy_cls, env_name='Pendulum-v1', lr=3e-4, gamma=0.99, render=False):
        render_params = dict(render_mode="human") if render else {}
        self.env = gym.make(env_name, **render_params)
        self.policy = policy_cls(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0]
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        
        # Debugging info
        self._episode_rewards = []

        # Episode storage for training
        self._states = []
        self._actions = []
        self._log_probs = []
        self._rewards = []

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        action_t, logp_t = self.policy.sample(state_t)
        return action_t.detach().numpy(), logp_t

    def _compute_returns_to_go(self, rewards):
        # G_t = r_t + gamma * r_{t+1} + ...
        # Save all of them so we can leverage the causality argument
        G = []
        running = 0.0
        for r in reversed(rewards):
            running = r + self.gamma * running
            G.append(running)
        G.reverse()
        return torch.as_tensor(G, dtype=torch.float32)
    
    def update_policy(self):
        returns = self._compute_returns_to_go(self._rewards)
        logps = torch.stack(self._log_probs)
        policy_loss = -(logps * returns).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # clear storage
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()

        return float(policy_loss.item())
    
    def run_episode(self, is_training=True):
        """Train for one episode"""
        state, info = self.env.reset()
        
        episode_reward = 0
        while True:
            # TODO: Possibly make action deterministic if is_training=False
            action, logp = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if is_training:
                self._states.append(state)
                self._actions.append(action)
                self._log_probs.append(logp if logp.dim() == 0 else logp.squeeze())
                self._rewards.append(reward)

            state = next_state
            if done:
                break

        
        if is_training:
            self.update_policy()
            self._episode_rewards.append(episode_reward)
        return episode_reward
    
    def evaluate(self, num_episodes=5):
        """Evaluate current policy - useful for debugging"""
        total_rewards = [self.run_episode(is_training=False) for _ in range(num_episodes)]
        return np.mean(total_rewards), np.std(total_rewards)
    
    def train(self, num_episodes=1000, eval_interval=100):
        """Main training loop"""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Environment: {self.env.spec.id}")
        print("-" * 50)
        
        eval_rewards = []
        
        for episode in range(num_episodes):
            episode_reward = self.run_episode()
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self._episode_rewards[-10:])
                print(f"Episode {episode:4d} | Reward: {episode_reward:8.2f} | "
                      f"Avg Reward (last 10): {avg_reward:8.2f}")
            
            # Evaluate policy
            if episode % eval_interval == 0 and episode > 0:
                eval_reward, eval_std = self.evaluate(num_episodes=5)
                eval_rewards.append(eval_reward)
                print(f"Evaluation at episode {episode}: {eval_reward:.2f} ± {eval_std:.2f}")
        
        print("\nTraining completed!")
        return self._episode_rewards, eval_rewards


def plot_training_results(episode_rewards, eval_rewards):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards During Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot evaluation rewards
    if eval_rewards:
        eval_episodes = np.arange(0, len(eval_rewards)) * 100
        ax2.plot(eval_episodes, eval_rewards, 'r-', marker='o')
        ax2.set_title('Evaluation Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the RL training"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create RL agent
    agent = PendulumRL(policy_cls=GaussianPolicy)
    # Train the agent
    episode_rewards, eval_rewards = agent.train(
        num_episodes=500,
        eval_interval=50
    )
    plot_training_results(episode_rewards, eval_rewards)
    agent.env.close()


if __name__ == "__main__":
    main()
