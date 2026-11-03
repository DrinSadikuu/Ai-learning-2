
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from datasets import load_dataset


# 1. Load dataset
dataset = load_dataset("Melisa13/Animals_dataset")

train_data = dataset["train"].filter(lambda x: x["label"] != 2)
test_data = dataset["test"].filter(lambda x: x["label"] != 2)


# 2. Prepare images
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


# 3. Load previous trained model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cat_dog_resnet18.pth", weights_only=True))


# 4. Freeze all layers
for parameter in model.parameters():
    parameter.requires_grad = False


# 5. Unfreeze layer4 and final classifier
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True


# 6. Set up fine-tuning
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    [
        {"params": model.layer4.parameters(), "lr": 0.00001},
        {"params": model.fc.parameters(), "lr": 0.0001},
    ]
)

epochs = 3


# 7. Fine-tune model
for epoch in range(epochs):
    model.eval()
    model.layer4.train()

    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")


# 8. Evaluate model
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100

print(f"Fine-tuning accuracy: {accuracy:.2f}%")


# 9. Save fine-tuned model
torch.save(model.state_dict(), "cat_dog_finetuned.pth")
