
import torch
from torch import nn
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# 1. Load our fine-tuned model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("cat_dog_finetuned.pth", weights_only=True)
)
model.eval()

# 2. Prepare the image
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

image = Image.open("dog.png").convert("RGB")
image_tensor = preprocess(image).unsqueeze(0)

# 3. Make prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

# 4. Display result
class_id = probabilities.argmax(dim=1).item()
categories = ["Cat", "Dog"]

print("Prediction:", categories[class_id])
print(f"Confidence: {probabilities[0, class_id].item():.2%}")
