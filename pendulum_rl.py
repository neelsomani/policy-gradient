"""
Reinforcement Learning Implementation for Pendulum-v1 Environment
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
    
    def forward(self, state):
        raise NotImplementedError
    
    def get_action(self, state):
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
    
    def get_action(self, state):
        """Get action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.forward(state_tensor)
            # Scale to Pendulum action space [-2, 2]
            action = action * 2.0
            return action.squeeze(0).numpy()


class PendulumRL:
    """
    Main RL class for training on Pendulum-v1 environment
    """
    
    def __init__(self, env_name='Pendulum-v1', lr=3e-4, gamma=0.99, batch_size=64, render=False):
        render_params = dict(render_mode="human") if render else {}
        self.env = gym.make(env_name, **render_params)
        self.policy = DummyPolicy(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0]
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self._current_batch = []
        
        # Debugging info
        self._episode_rewards = []
        
    def select_action(self, state):
        """Select action using current policy"""
        action = self.policy.get_action(state)
        return action
    
    def update_policy(self):
        """Update policy using stored experiences"""
        states, actions, rewards, next_states, dones = map(np.stack, zip(*self._current_batch))
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # FIXME: Compute policy loss
        # F.mse_loss(...)
        # Update policy
        self.optimizer.zero_grad()
        # action_loss.backward()
        self.optimizer.step()

        self._current_batch = []
    
    def run_episode(self, is_training=True):
        """Train for one episode"""
        state, info = self.env.reset()
        
        episode_reward = 0
        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self._current_batch.append((state, action, reward, next_state, terminated or truncated))
            
            state = next_state
            episode_reward += reward
            
            if terminated or truncated:
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
    agent = PendulumRL()
    # Train the agent
    episode_rewards, eval_rewards = agent.train(
        num_episodes=500,
        eval_interval=50
    )
    plot_training_results(episode_rewards, eval_rewards)
    agent.env.close()


if __name__ == "__main__":
    main()
