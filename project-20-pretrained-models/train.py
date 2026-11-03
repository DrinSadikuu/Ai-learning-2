
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from datasets import load_dataset


# 1. Load cats and dogs dataset
dataset = load_dataset("Melisa13/Animals_dataset")

train_data = dataset["train"].filter(lambda x: x["label"] != 2)
test_data = dataset["test"].filter(lambda x: x["label"] != 2)


# 2. Prepare images for ResNet18
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()


class AnimalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        image = preprocess(example["image"].convert("RGB"))
        label = example["label"]
        return image, label


train_loader = DataLoader(AnimalDataset(train_data), batch_size=16, shuffle=True)
test_loader = DataLoader(AnimalDataset(test_data), batch_size=16)


# 3. Load pretrained ResNet18
model = resnet18(weights=weights)


# 4. Freeze pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# 5. Replace final classification layer
model.fc = nn.Linear(model.fc.in_features, 2)


# 6. Set up training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 5

# 7. Train the new classification layer
model.eval()

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")


# 8. Evaluate on test images
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100

print(f"Test accuracy: {accuracy:.2f}%")


# 9. Save trained model
torch.save(model.state_dict(), "cat_dog_resnet18.pth")
