from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset_loader import RadioMLDataset
from models.st_hgnn import STHGNN

# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Using Device :", device)
print("=" * 60)

# ==========================================================
# Test Dataset
# ==========================================================
test_dataset = RadioMLDataset(
    "data/processed_dataset.pt",
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Test Samples :", len(test_dataset))

# ==========================================================
# Load Model
# ==========================================================
model = STHGNN().to(device)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device
)

# Load only the model weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\n✓ Model Loaded Successfully")
print("=" * 60)

# ==========================================================
# Testing
# ==========================================================
correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        _, predicted = torch.max(outputs, dim=1)

        total += y.size(0)
        correct += (predicted == y).sum().item()

        all_labels.extend(y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# ==========================================================
# Accuracy
# ==========================================================
accuracy = 100 * correct / total

print("\n" + "=" * 60)
print(f"Test Accuracy : {accuracy:.2f}%")
print("=" * 60)

# ==========================================================
# Classification Report
# ==========================================================
print("\nClassification Report")
print("-" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================
cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("=" * 60)
print("Testing Completed Successfully")
print("=" * 60)