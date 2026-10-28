
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 1. Load MNIST dataset
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

train_data = Subset(train_data, range(5000))
test_data = Subset(test_data, range(1000))

train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True
)

test_loader = DataLoader(
    test_data, batch_size=64
)

# 2. Build CNN
torch.manual_seed(42)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(16, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(32 * 5 * 5, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 3. Configure training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train CNN
for epoch in range(10):
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

# 5. Evaluate CNN
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
print(f"\nCNN Test Accuracy: {accuracy:.2%}")

import matplotlib.pyplot as plt

wrong_images = []
wrong_labels = []
wrong_predictions = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images).argmax(dim=1)

        for i in range(len(labels)):
            if predictions[i] != labels[i]:
                wrong_images.append(images[i])
                wrong_labels.append(labels[i].item())
                wrong_predictions.append(predictions[i].item())

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i in range(min(10, len(wrong_images))):
    ax = axes[i // 5, i % 5]
    ax.imshow(wrong_images[i].squeeze(), cmap="gray")
    ax.set_title(
        f"Actual: {wrong_labels[i]}\n"
        f"Predicted: {wrong_predictions[i]}"
    )
    ax.axis("off")

for i in range(min(10, len(wrong_images)), 10):
    axes[i // 5, i % 5].axis("off")

plt.tight_layout()
plt.show()
