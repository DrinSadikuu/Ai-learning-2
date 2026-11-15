
import numpy as np

positions = 5
actions = 2
goal = 4

q_table = np.zeros((positions, actions))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2

episodes = 1000

for episode in range(episodes):
    state = 0

    for step in range(20):

        # Choose an action
        if np.random.random() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(q_table[state])

        # Move the robot
        if action == 0:
            next_state = max(0, state - 1)
        else:
            next_state = min(positions - 1, state + 1)

        # Calculate reward
        if next_state == goal:
            reward = 10
        else:
            reward = -1

        # Check whether the goal was reached
        done = next_state == goal

        # Calculate future value
        if done:
            future_value = 0
        else:
            future_value = np.max(q_table[next_state])

        # Update Q-table
        q_table[state, action] += learning_rate * (
            reward + discount_factor * future_value
            - q_table[state, action]
        )

        state = next_state

        if done:
            break

print("Learned Q-table:")
print(np.round(q_table, 2))

print("\nTesting the trained AI:")

state = 0

while state != goal:
    action = np.argmax(q_table[state])

    if action == 0:
        next_state = max(0, state - 1)
        direction = "LEFT"
    else:
        next_state = min(positions - 1, state + 1)
        direction = "RIGHT"

    print(f"Position {state} -> {direction} -> Position {next_state}")

    state = next_state

print("Goal reached!")

