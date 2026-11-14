
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

for result in model.predict(source=0, show=True, stream=True):
    pass
