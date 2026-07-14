import os
import json
import cv2  # type: ignore

# Paths
labels_path = "dataset/labels/train"
images_path = "dataset/images/train"
output_path = "dataset/labels/train"

# Supported image extensions
extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

def find_image(image_name):
    """Search for image with any supported extension"""
    base_name = os.path.splitext(image_name)[0]

    for ext in extensions:
        possible_path = os.path.join(images_path, base_name + ext)
        if os.path.exists(possible_path):
            return possible_path

    return None


for file in os.listdir(labels_path):
    if file.endswith(".json"):
        json_path = os.path.join(labels_path, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        image_name = data[0]["image"]
        image_path = find_image(image_name)

        if image_path is None:
            print(f"❌ Image not found for {file}")
            print(f"   Tried base name: {os.path.splitext(image_name)[0]}")
            continue

        img = cv2.imread(image_path)
        h, w, _ = img.shape

        txt_filename = os.path.splitext(file)[0] + ".txt"
        txt_path = os.path.join(output_path, txt_filename)

        with open(txt_path, "w") as txt_file:
            for obj in data[0]["annotations"]:
                x = obj["coordinates"]["x"]
                y = obj["coordinates"]["y"]
                width = obj["coordinates"]["width"]
                height = obj["coordinates"]["height"]

                # Convert to YOLO format
                x_center = x / w
                y_center = y / h
                width /= w
                height /= h

                class_id = 0  # change if multiple classes

                txt_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"✅ Converted: {file} → {txt_filename}")

print("\n🎉 Conversion completed!")