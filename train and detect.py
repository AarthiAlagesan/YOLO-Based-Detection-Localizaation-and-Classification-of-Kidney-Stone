import os
import cv2
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_YAML = "stone.yaml"
MODEL_NAME = "yolov10n.pt"     # lightweight & fast
EPOCHS = 50
IMG_SIZE = 640
TEST_IMAGE = "dataset/images/val/sample.png"  # change this

# -------------------------------
# TRAIN YOLOv10
# -------------------------------
def train_model():
    model = YOLO(MODEL_NAME)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project="runs",
        name="kidney_stone_yolov10"
    )

# -------------------------------
# DETECT & COUNT STONES
# -------------------------------
def detect_and_count(image_path):
    model = YOLO("runs/kidney_stone_yolov10/weights/best.pt")

    results = model(image_path, conf=0.3)

    img = cv2.imread(image_path)

    for r in results:
        boxes = r.boxes
        stone_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"Stone {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    print(f"🟢 Number of kidney stones detected: {stone_count}")

    cv2.imshow("Kidney Stone Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Step 1: Train (run once)
    train_model()

    # Step 2: Detect & count stones
    detect_and_count(TEST_IMAGE)
