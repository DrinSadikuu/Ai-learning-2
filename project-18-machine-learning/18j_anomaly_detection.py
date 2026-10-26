
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Machine data: [temperature, vibration]
machines = [
    [65, 0.20],
    [68, 0.25],
    [63, 0.18],
    [66, 0.22],
    [70, 0.28],
    [64, 0.19],
    [67, 0.24],
    [69, 0.26],
    [62, 0.17],
    [71, 0.30],
    [65, 0.21],
    [68, 0.23],
    [105, 0.90],
    [40, 0.05]
]

# 2. Create model
model = IsolationForest(
    contamination=0.14,
    random_state=42
)

# 3. Train and detect anomalies
predictions = model.fit_predict(machines)

# 4. Get anomaly scores
scores = model.decision_function(machines)

# 5. Print results
for machine, prediction, score in zip(machines, predictions, scores):
    status = "ANOMALY" if prediction == -1 else "NORMAL"

    print(
        f"Temperature: {machine[0]}°C, "
        f"Vibration: {machine[1]} -> "
        f"{status}, Score: {score:.4f}"
    )

# 6. Visualize results
temperatures = [machine[0] for machine in machines]
vibrations = [machine[1] for machine in machines]

colors = [
    "red" if prediction == -1 else "blue"
    for prediction in predictions
]

plt.figure(figsize=(8, 5))

plt.scatter(
    temperatures,
    vibrations,
    c=colors,
    s=100
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration")
plt.title("Machine Anomaly Detection - Isolation Forest")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Visualize anomaly scores
plt.figure(figsize=(9, 5))

plt.bar(
    range(1, len(machines) + 1),
    scores,
    color=colors
)

plt.axhline(0, color="black", linestyle="--")

plt.xlabel("Machine Number")
plt.ylabel("Anomaly Score")
plt.title("Isolation Forest - Anomaly Scores")
plt.xticks(range(1, len(machines) + 1))
plt.grid(axis="y")
plt.tight_layout()
plt.show()
