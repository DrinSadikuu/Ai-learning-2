
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# Load pretrained model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# Prepare image
image = Image.open("dog.png").convert("RGB")
preprocess = weights.transforms()
image_tensor = preprocess(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

# Get result
class_id = probabilities.argmax(dim=1).item()
category = weights.meta["categories"][class_id]
confidence = probabilities[0, class_id].item()

print(f"Prediction: {category}")
print(f"Confidence: {confidence:.2%}")
