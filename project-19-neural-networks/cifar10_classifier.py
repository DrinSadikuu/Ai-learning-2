
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Load CIFAR-10 dataset
transform = transforms.ToTensor()

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

classes = train_data.classes

train_data = Subset(train_data, range(5000))
test_data = Subset(test_data, range(1000))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 2. Build CNN
torch.manual_seed(42)

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(16, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(32 * 6 * 6, 64),
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
        predicted_classes = predictions.argmax(dim=1)

        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\nCIFAR-10 Test Accuracy: {accuracy:.2%}")

# 6. Visualize predictions
images, labels = next(iter(test_loader))

with torch.no_grad():
    predictions = model(images).argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i in range(10):
    ax = axes[i // 5, i % 5]

    image = images[i].permute(1, 2, 0)
    ax.imshow(image)

    actual = classes[labels[i].item()]
    predicted = classes[predictions[i].item()]

    ax.set_title(f"Actual: {actual}\nPredicted: {predicted}")
    ax.axis("off")

plt.tight_layout()
plt.show()
