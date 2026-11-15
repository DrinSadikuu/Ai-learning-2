
import gymnasium as gym
import torch
from torch import nn

# Create environment with a visible window
env = gym.make("CartPole-v1", render_mode="human")

# Recreate the same neural network
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Load the trained weights
model.load_state_dict(
    torch.load("cartpole_dqn.pth", weights_only=True)
)

model.eval()

# Start simulation
state, info = env.reset()
total_reward = 0

for step in range(500):
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action = model(state_tensor).argmax().item()

    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print("Steps survived:", total_reward)

env.close()
