
import cv2
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

# Track objects that have already crossed the line
counted_ids = set()
previous_positions = {}
count = 0

for result in model.track(
    source=0,
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
):
    frame = result.orig_img
    height, width = frame.shape[:2]

    # Horizontal line in the middle of the camera
    line_y = height // 2

    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

    if result.boxes is not None and result.boxes.id is not None:
        for box in result.boxes:
            object_id = int(box.id.item())
            class_id = int(box.cls.item())

            # Count only people and common handheld objects
            if result.names[class_id] not in ["person", "bottle", "cell phone"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_y = (y1 + y2) // 2

            # Count when object crosses the horizontal line
            if object_id in previous_positions and object_id not in counted_ids:
                old_y = previous_positions[object_id]

                if (old_y < line_y <= center_y) or (old_y > line_y >= center_y):
                    count += 1
                    counted_ids.add(object_id)

            previous_positions[object_id] = center_y

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {object_id}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Line Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
