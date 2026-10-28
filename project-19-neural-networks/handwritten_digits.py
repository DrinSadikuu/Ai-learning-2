
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Download and prepare MNIST
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Use small subsets to keep training fast
train_data = Subset(train_data, range(5000))
test_data = Subset(test_data, range(1000))

train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=64
)

# 2. Build neural network
torch.manual_seed(42)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 3. Configure training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# 4. Train for 3 epochs
for epoch in range(3):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: Loss = {average_loss:.4f}")

# 5. Evaluate
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_digits = predictions.argmax(dim=1)

        correct += (predicted_digits == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\nTest Accuracy: {accuracy:.2%}")

# 6. Show example predictions
images, labels = next(iter(test_loader))

with torch.no_grad():
    predictions = model(images)
    predicted_digits = predictions.argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i in range(10):
    ax = axes[i // 5, i % 5]

    ax.imshow(images[i].squeeze(), cmap="gray")

    actual = labels[i].item()
    predicted = predicted_digits[i].item()

    ax.set_title(f"Actual: {actual}\nPredicted: {predicted}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# 7. Inspect a wrong prediction
image_index = 8

with torch.no_grad():
    scores = model(images[image_index].unsqueeze(0))
    probabilities = torch.softmax(scores, dim=1)

print("\nDigit scores for image 8:")

for digit in range(10):
    probability = probabilities[0][digit].item()
    print(f"Digit {digit}: {probability:.2%}")

print("Actual digit:", labels[image_index].item())
print("Predicted digit:", probabilities.argmax(dim=1).item())
