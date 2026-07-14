import torch
import matplotlib.pyplot as plt
import csv
import torch.nn as nn
from torchvision.models import resnet18

# ================= PATHS =================
MODEL_PATH = "stone_cnn.pth"
METRICS_PATH = "training_metrics.pth"
CSV_FILE = "training_metrics.csv"

# ================= LOAD METRICS =================
metrics = torch.load(METRICS_PATH)

train_losses = metrics["train_losses"]
val_losses = metrics["val_losses"]
train_acc = metrics["train_accuracies"]
val_acc = metrics["val_accuracies"]

# ================= SAVE CSV =================
with open(CSV_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch","Train Loss","Validation Loss","Train Accuracy","Validation Accuracy"])

    for i in range(len(train_losses)):
        writer.writerow([
            i+1,
            train_losses[i],
            val_losses[i],
            train_acc[i],
            val_acc[i]
        ])

print("✅ CSV file created")

# ================= LOSS GRAPH =================
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

# ================= ACCURACY GRAPH =================
plt.figure(figsize=(8,6))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training & Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("cnn_accuracy_graph.png")
plt.show()

# ================= LOAD CNN MODEL =================
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 8)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

print("✅ CNN model loaded successfully\n")
print(model)