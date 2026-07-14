from ultralytics import YOLO # type: ignore
import numpy as np # type: ignore

MODEL_PATH = "runs/detect/train/weights/best.pt"

model = YOLO(MODEL_PATH)

metrics = model.val()

print("\n========= VALIDATION RESULTS =========")

precision = metrics.box.p.mean()
recall = metrics.box.r.mean()
map50 = metrics.box.map50
map5095 = metrics.box.map

print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"mAP@0.5        : {map50:.4f}")
print(f"mAP@0.5:0.95   : {map5095:.4f}")
