import os
import shutil
import random

# --------------------------
# Paths
# --------------------------
images_folder = "dataset/images/train"  # your images
labels_folder = "dataset/labels/train"  # your labeled txt files

val_images = "dataset/images/val"
val_labels = "dataset/labels/val"

# --------------------------
# Make val folders if they don't exist
# --------------------------
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# --------------------------
# List all images that have labels
# --------------------------
all_images = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images_with_labels = []

for img in all_images:
    label_file = os.path.splitext(img)[0] + ".txt"
    if os.path.exists(os.path.join(labels_folder, label_file)):
        images_with_labels.append(img)
    else:
        print(f"⚠ Warning: Missing label for {img}")

if len(images_with_labels) == 0:
    print("❌ No images with labels found! Exiting...")
    exit()

# --------------------------
# Shuffle and split 80/20
# --------------------------
random.seed(42)
random.shuffle(images_with_labels)

split_index = int(0.8 * len(images_with_labels))
train_files = images_with_labels[:split_index]
val_files   = images_with_labels[split_index:]

# --------------------------
# Move 20% to val
# --------------------------
for img in val_files:
    # Move image
    shutil.move(os.path.join(images_folder, img), os.path.join(val_images, img))
    
    # Move corresponding label
    label_file = os.path.splitext(img)[0] + ".txt"
    shutil.move(os.path.join(labels_folder, label_file), os.path.join(val_labels, label_file))

print("\n✅ Dataset split completed successfully!")
print("Training images/labels (remain in train):", len(train_files))
print("Validation images/labels (moved to val):", len(val_files))