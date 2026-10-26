
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("data/ai4i2020.csv")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32
).reshape(-1, 1)

y_test = torch.tensor(
    y_test.to_numpy(), dtype=torch.float32
).reshape(-1, 1)

# 5. Build neural network
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 6. Calculate failure weight using training data
failures = y_train.sum().item()
healthy = len(y_train) - failures

weight = healthy / failures

print(f"Failure weight: {weight:.2f}")

# 7. Configure weighted loss and optimizer
loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([weight])
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. Train neural network
for epoch in range(100):
    model.train()

    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# 9. Evaluate
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.7).int()

print("\nClassification Report:")
print(classification_report(
    y_test.numpy(), predictions.numpy(), zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test.numpy(), predictions.numpy()))
