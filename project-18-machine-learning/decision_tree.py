
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Features: [temperature (°C), vibration level]
X = np.array([
    [40, 2],
    [45, 3],
    [50, 2],
    [55, 4],
    [65, 5],
    [75, 7],
    [85, 8],
    [95, 9]
])

# Target: 0 = Normal, 1 = Maintenance needed
y = np.array([0, 0, 0, 0, 0, 1, 1, 1])

# Create and train the Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Predict the condition of a new machine
prediction = model.predict([[80, 8]])

if prediction[0] == 1:
    print("Maintenance needed")
else:
    print("Machine operating normally")

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 6))

plot_tree(
    model,
    feature_names=["Temperature", "Vibration"],
    class_names=["Normal", "Maintenance"],
    filled=True,
    rounded=True
)

plt.title("Machine Maintenance Decision Tree")
plt.show()
