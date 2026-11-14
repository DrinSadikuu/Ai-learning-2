
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Track objects using webcam
for result in model.track(
    source=0,
    show=True,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml"
):
    pass
