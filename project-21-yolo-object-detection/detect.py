
from collections import Counter
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Detect objects
results = model.predict(source="img.png", save=True)

# Count detected objects
counts = Counter()

for box in results[0].boxes:
    class_id = int(box.cls[0])
    name = results[0].names[class_id]
    counts[name] += 1

# Display counts
for name, count in counts.items():
    print(f"{name}: {count}")
