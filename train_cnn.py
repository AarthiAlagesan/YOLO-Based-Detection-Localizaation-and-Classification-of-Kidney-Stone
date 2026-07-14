import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from collections import Counter

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

DATASET_PATH = DATASET_PATH = r"D:\ks\Expelled-Kidney-Stones-Processed-Dataset\Processed-Dataset"
CNN_MODEL_PATH = r"D:\ks\stone_cnn.pth"
METRICS_PATH = r"D:\ks\training_metrics.pth"

EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0001
NUM_WORKERS = 0

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= LOAD DATASET =================
full_dataset = ImageFolder(DATASET_PATH, transform=transform)

NUM_CLASSES = len(full_dataset.classes)
print("Classes:", full_dataset.classes)
print("Total images:", len(full_dataset))

# ================= TRAIN / VAL SPLIT =================
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# ================= CLASS WEIGHTS =================
targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = Counter(targets)
total_samples = sum(class_counts.values())

class_weights = [total_samples / class_counts[i] for i in range(NUM_CLASSES)]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print("Class weights:", class_weights)

# ================= MODEL =================
cnn_model = resnet18(weights=ResNet18_Weights.DEFAULT)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, NUM_CLASSES)
cnn_model = cnn_model.to(DEVICE)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(cnn_model.parameters(), lr=LR)

# ================= METRIC STORAGE =================
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# ================= TRAINING FUNCTION =================
def train_model():
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        cnn_model.train()

        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # -------- VALIDATION --------
        cnn_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = cnn_model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"🧪 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # Save model
    torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)
    print(f"\n💾 Model saved at: {CNN_MODEL_PATH}")

    # Save metrics
    torch.save({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }, METRICS_PATH)

    print("📊 Training metrics saved!")

# ================= PLOT FUNCTION =================
def plot_graphs():

    # ----- LOSS GRAPH -----
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_loss_graph.png")
    plt.show()

    # ----- ACCURACY GRAPH -----
    plt.figure(figsize=(8,6))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_accuracy_graph.png")
    plt.show()

    print("📈 Graphs saved successfully!")

# ================= MAIN =================
if __name__ == "__main__":
    train_model()
    plot_graphs()