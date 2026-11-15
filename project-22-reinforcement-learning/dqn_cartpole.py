
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn

# 1. Create environment
env = gym.make("CartPole-v1")

# 2. Create neural network
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# 3. Create target network
target_model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
target_model.load_state_dict(model.state_dict())

# 4. Configure training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

memory = deque(maxlen=10000)
batch_size = 64
episodes = 500

# 5. Training
for episode in range(episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(500):

        # Choose action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        # Move cart
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Save experience
        memory.append((state, action, reward, next_state, terminated))

        state = next_state
        total_reward += reward

        # Train neural network
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)

            states, actions, rewards, next_states, terminals = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            terminals = torch.tensor(terminals, dtype=torch.float32)

            # Current Q-values
            predictions = model(states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                future_q = target_model(next_states).max(dim=1).values
                targets = rewards + gamma * future_q * (1 - terminals)

            # Update weights
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Reduce random exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network every 10 episodes
    if (episode + 1) % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    if (episode + 1) % 20 == 0:
        print(f"Episode {episode + 1}: Reward = {total_reward}, Epsilon = {epsilon:.2f}")

# 6. Save trained model
torch.save(model.state_dict(), "cartpole_dqn.pth")

env.close()

# 7. Test trained model
test_env = gym.make("CartPole-v1")
state, info = test_env.reset()

total_reward = 0

for step in range(500):
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action = model(state_tensor).argmax().item()

    state, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print("\nFinal test reward:", total_reward)

test_env.close()
