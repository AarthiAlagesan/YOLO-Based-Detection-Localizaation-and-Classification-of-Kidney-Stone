from ultralytics import YOLO # type: ignore
import cv2 # type: ignore
import os
import numpy as np # type: ignore
import math

# ================= CONFIG =================
MODEL_PATH = "runs/detect/train8/weights/best.pt"
IMAGE_FOLDER = "test_images"
RESULTS_FOLDER = "results"
CROP_FOLDER = os.path.join(RESULTS_FOLDER, "stone_crops")

# ---------------------------------------------------------
# CALIBRATION:
# MM_PER_PIXEL must be computed experimentally.
# Example:
# If a 6 mm stone measures 12 pixels in PNG:
# MM_PER_PIXEL = 6 / 12 = 0.5
# ---------------------------------------------------------
MM_PER_PIXEL = 0.5   # Replace with experimentally calculated value

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

print("Loaded YOLO model")

# ================= SIZE CLASSIFICATION =================
# Clinical guideline:
# <5mm = Small
# 5-10mm = Medium
# >10mm = Large
def classify_size(diameter_mm):
    if diameter_mm < 5:
        return "Small"
    elif diameter_mm <= 10:
        return "Medium"
    else:
        return "Large"

# ================= SAFE NMS =================
def apply_nms(r, conf_thres=0.05, nms_thres=0.4):

    if r.boxes is None or len(r.boxes) == 0:
        return [], []

    boxes = np.array([b.xyxy[0].cpu().numpy() for b in r.boxes])
    confs = np.array([b.conf.item() for b in r.boxes])

    boxes_xywh = []
    for x1, y1, x2, y2 in boxes:
        boxes_xywh.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])

    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh,
        confs.tolist(),
        score_threshold=conf_thres,
        nms_threshold=nms_thres
    )

    if idxs is None or len(idxs) == 0:
        return [], []

    idxs = np.array(idxs).flatten()
    return [boxes[i] for i in idxs], [confs[i] for i in idxs]

# ================= MAIN LOOP =================
for img_name in os.listdir(IMAGE_FOLDER):

    if not img_name.lower().endswith(".png"):
        continue

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    original = img.copy()
    img_width = img.shape[1]

    results = model(img_path, conf=0.05)

    print("\n==============================")
    print("Image:", img_name)

    for r in results:

        boxes, confs = apply_nms(r)

        if not boxes:
            print("Stone Present: False")
            continue

        print("Stone Present: True")
        print("Number of Stones:", len(boxes))

        for i, (x1, y1, x2, y2) in enumerate(boxes):

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            side = "Left Kidney" if (x1 + x2)/2 < img_width/2 else "Right Kidney"
            confidence = confs[i]

            # ROI extraction
            roi_original = original[y1:y2, x1:x2]
            roi = roi_original.copy()

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # ---------------------------------------------------------
            # SCIENTIFIC SEGMENTATION:
            # Otsu automatically selects optimal threshold
            # based on intra-class variance minimization
            # ---------------------------------------------------------
            _, mask = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area_pixels = cv2.contourArea(largest_contour)

                # Remove tiny noise regions
                if area_pixels < 10:
                    continue

                # ---------------------------------------------------------
                # AREA-BASED EQUIVALENT DIAMETER
                # Formula:
                # D = 2 * sqrt(A / π)
                # Assumes equivalent circular approximation
                # ---------------------------------------------------------
                diameter_pixels = 2 * math.sqrt(area_pixels / math.pi)
                diameter_mm = diameter_pixels * MM_PER_PIXEL
            else:
                continue

            size_class = classify_size(diameter_mm)

            print(f"\nStone {i+1}")
            print(f"  Confidence : {confidence:.3f}")
            print(f"  Location   : {side}")
            print(f"  Diameter   : {diameter_mm:.2f} mm")
            print(f"  Category   : {size_class}")

            # Visualization (blue highlight only on mask)
            blue_overlay = np.zeros_like(roi)
            blue_overlay[:] = (255, 0, 0)

            roi[mask == 255] = cv2.addWeighted(
                roi[mask == 255], 0.3,
                blue_overlay[mask == 255], 0.7,
                0
            )

            img[y1:y2, x1:x2] = roi

            # Green bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Red text
            cv2.putText(img, side,
                        (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255), 2)

            cv2.putText(img, size_class,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255), 2)

            # Save original crop
            crop_name = f"{os.path.splitext(img_name)[0]}_stone_{i+1}.png"
            cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), roi_original)

    cv2.imwrite(os.path.join(RESULTS_FOLDER, img_name), img)

print("\n✅ DONE")
