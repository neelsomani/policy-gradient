# Pendulum-v1 Reinforcement Learning Implementation

This project implements a reinforcement learning algorithm for the Pendulum-v1 environment using PyTorch, featuring a dummy policy for testing and demonstration purposes.

## Features

- **Dummy Policy Network**: A simple neural network that learns to control the pendulum
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Target Network**: Uses target networks for more stable training
- **Evaluation Metrics**: Tracks training progress and evaluates policy performance
- **Visualization**: Generates training plots and supports environment rendering

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training
Run the main training script:
```bash
python3 pendulum_rl.py
```

## Environment Details

- **Environment**: Pendulum-v1 (Gymnasium)
- **State Space**: 3D continuous (cos(theta), sin(theta), angular velocity)
- **Action Space**: 1D continuous [-2, 2] (torque applied to pendulum)
- **Goal**: Balance the pendulum upright with minimal energy
