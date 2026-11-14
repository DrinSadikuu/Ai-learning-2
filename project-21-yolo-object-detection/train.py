
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Fine-tune the model using COCO8
model.train(
    data="coco8.yaml",
    epochs=5,
    imgsz=320,
    batch=2,
    device="cpu"
)
