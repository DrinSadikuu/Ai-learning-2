
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO.load("cartpole_ppo")

state, info = env.reset()
total_reward = 0

for step in range(500):
    action, _ = model.predict(state, deterministic=True)

    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

print("Steps survived:", total_reward)

env.close()
