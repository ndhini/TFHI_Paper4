import os
import random
import numpy as np
from configs.config import *

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils.dataset_loader import RadioMLDataset

from models.st_hgnn import STHGNN
from training.ssl_trainer import SSLTrainer

from utils.logger import TrainingLogger
from utils.checkpoint import CheckpointManager
from utils.early_stopping import EarlyStopping
from utils.plots import TrainingPlotter

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

##############################################################
# Reproducibility
##############################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##############################################################
# Create Required Directories
##############################################################

os.makedirs("logs", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

##############################################################
# Device
##############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ST-HGNN Research Framework")
print("=" * 60)
print("Using Device :", device)

##############################################################
# Dataset
##############################################################

train_dataset = RadioMLDataset(
    "data/processed_dataset.pt",
    split="train"
)

val_dataset = RadioMLDataset(
    "data/processed_dataset.pt",
    split="valid"
)

from torch.utils.data import Subset

# ==========================================================
# FAST RESEARCH MODE
# ==========================================================

if USE_SUBSET:

    train_dataset = Subset(
        train_dataset,
        range(min(TRAIN_SUBSET, len(train_dataset)))
    )

    val_dataset = Subset(
        val_dataset,
        range(min(VAL_SUBSET, len(val_dataset)))
    )

    print("FAST RESEARCH MODE ENABLED")

# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Training Samples  :", len(train_dataset))
print("Validation Samples:", len(val_dataset))

##############################################################
# Model
##############################################################

model = STHGNN().to(device)

ssl_model = SSLTrainer(
    model.encoder
).to(device)

##############################################################
# Loss Function
##############################################################

criterion = nn.CrossEntropyLoss()

##############################################################
# Optimizer
##############################################################

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

ssl_optimizer = torch.optim.Adam(
    ssl_model.parameters(),
    lr=1e-3
)

##############################################################
# Scheduler
##############################################################

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=TRAIN_EPOCHS,
    eta_min=1e-6
)

##############################################################
# Utilities
##############################################################

logger = TrainingLogger()

checkpoint = CheckpointManager()

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001
)

plotter = TrainingPlotter()

##############################################################
# History
##############################################################

train_losses = []
val_accuracies = []
ssl_losses = []

best_accuracy = 0.0

best_y_true = []
best_y_pred = []

start_epoch = 0

# Resume training if checkpoint exists
if os.path.exists("checkpoints/latest_model.pth"):

    checkpoint_data = torch.load(
        "checkpoints/latest_model.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint_data["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint_data["optimizer_state_dict"]
    )

    start_epoch = checkpoint_data["epoch"]

    best_accuracy = checkpoint_data["accuracy"]

    print("=" * 60)
    print(f"Resuming Training from Epoch {start_epoch}")
    print(f"Best Accuracy : {best_accuracy:.2f}%")
    print("=" * 60)

print("\nInitialization Completed Successfully.")
print("=" * 60)

##############################################################
# SELF-SUPERVISED PRETRAINING
##############################################################

if start_epoch == 0:

    ssl_epochs = SSL_EPOCHS

    print("\n" + "=" * 60)
    print("SELF-SUPERVISED PRETRAINING")
    print("=" * 60)

    ssl_model.train()

    for epoch in range(ssl_epochs):

        running_ssl_loss = 0.0
        total_batches = 0

        for batch_idx, (x, _) in enumerate(train_loader):

            if batch_idx >= 100:
                break

            x = x.to(device)

            ssl_optimizer.zero_grad()

            loss, _, _ = ssl_model(x)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                ssl_model.parameters(),
                max_norm=1.0
            )

            ssl_optimizer.step()

            running_ssl_loss += loss.item()
            total_batches += 1

            if batch_idx % 20 == 0:

                print(
                    f"SSL Epoch [{epoch+1}/{ssl_epochs}] "
                    f"Batch [{batch_idx:03d}/100] "
                    f"Loss : {loss.item():.6f}"
                )

        avg_ssl_loss = running_ssl_loss / total_batches

        ssl_losses.append(avg_ssl_loss)

        print("-" * 60)
        print(f"Epoch {epoch+1} Completed")
        print(f"Average SSL Loss : {avg_ssl_loss:.6f}")
        print("-" * 60)

    print("\nSSL PRETRAINING COMPLETED")
    print("=" * 60)

else:

    print("\nCheckpoint found.")
    print("Skipping SSL Pretraining.")

##############################################################
# SUPERVISED TRAINING
##############################################################

epochs = TRAIN_EPOCHS          # Development
# epochs = 100       # Final Paper

for epoch in range(start_epoch, epochs):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{epochs}")
    print("=" * 60)

    ##########################################################
    # TRAINING
    ##########################################################

    model.train()

    running_loss = 0.0
    total_batches = 0

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate : {current_lr:.8f}")

    for batch_idx, (x, y) in enumerate(train_loader):

        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        ######################################################
        # Gradient Clipping
        ######################################################

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        if batch_idx % 20 == 0:

            print(
                f"Batch [{batch_idx:03d}/{len(train_loader)-1:03d}] "
                f"Loss : {loss.item():.6f}"
            )

    ##########################################################
    # Average Training Loss
    ##########################################################

    avg_loss = running_loss / total_batches

    train_losses.append(avg_loss)

    ##########################################################
    # VALIDATION
    ##########################################################

    model.eval()

    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch_idx, (x_val, y_val) in enumerate(val_loader):

            
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)

            outputs = model(x_val)

            predictions = outputs.argmax(dim=1)

            total += y_val.size(0)

            correct += (predictions == y_val).sum().item()

            y_true.extend(
                y_val.cpu().numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    ##########################################################
    # Validation Accuracy
    ##########################################################

    accuracy = 100.0 * correct / total

    val_accuracies.append(accuracy)

    ##########################################################
    # Scheduler
    ##########################################################

    scheduler.step()

    ##########################################################
    # Logger
    ##########################################################

    logger.log(
        epoch=epoch + 1,
        train_loss=avg_loss,
        val_accuracy=accuracy,
        lr=optimizer.param_groups[0]["lr"]
    )

    ##########################################################
    # Save Latest Model
    ##########################################################

    checkpoint.save_latest(
        model,
        optimizer,
        epoch + 1,
        accuracy
    )

    ##########################################################
    # Save Best Model
    ##########################################################

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_y_true = y_true.copy()
        best_y_pred = y_pred.copy()

        checkpoint.save_best(
            model,
            optimizer,
            epoch + 1,
            accuracy
        )

        print("\n★★★★★ BEST MODEL UPDATED ★★★★★")

    ##########################################################
    # Epoch Summary
    ##########################################################

    print("\n" + "-" * 60)

    print(f"Epoch               : {epoch + 1}")
    print(f"Training Loss       : {avg_loss:.6f}")
    print(f"Validation Accuracy : {accuracy:.2f}%")
    print(f"Best Accuracy       : {best_accuracy:.2f}%")
    print(f"Learning Rate       : {optimizer.param_groups[0]['lr']:.8f}")

    print("-" * 60)

    ##########################################################
    # Early Stopping
    ##########################################################

    early_stopping(accuracy)

    if early_stopping.stop:

        print("\nEarly stopping triggered.")

        break
##############################################################
# TRAINING COMPLETED
##############################################################

print("\n" + "=" * 70)
print("Generating Training Curves...")
print("=" * 70)

##############################################################
# Plot Loss Curve
##############################################################

try:
    plotter.plot_loss(train_losses)
    print("✓ Loss curve saved.")
except Exception as e:
    print("Loss plot skipped :", e)

##############################################################
# Plot Accuracy Curve
##############################################################

try:
    plotter.plot_accuracy(val_accuracies)
    print("✓ Accuracy curve saved.")
except Exception as e:
    print("Accuracy plot skipped :", e)

##############################################################
# Save Final Model
##############################################################

torch.save(
    model.state_dict(),
    "checkpoints/final_model.pth"
)
##############################################################
# FINAL EVALUATION METRICS
##############################################################

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

precision = precision_score(
    best_y_true,
    best_y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    best_y_true,
    best_y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    best_y_true,
    best_y_pred,
    average="weighted",
    zero_division=0
)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        best_y_true,
        best_y_pred,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        best_y_true,
        best_y_pred
    )
)
##############################################################
# FINAL SUMMARY
##############################################################

print("\n" + "=" * 70)
print("ST-HGNN TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Best Validation Accuracy : {best_accuracy:.2f}%")
if len(train_losses) > 0:
    print(f"Final Training Loss      : {train_losses[-1]:.6f}")
else:
    print("Final Training Loss      : N/A")
print(f"Total Epochs Completed   : {len(train_losses)}")

print("=" * 70)

print("\nGenerated Files")
print("-" * 70)

print("✓ checkpoints/best_model.pth")
print("✓ checkpoints/latest_model.pth")
print("✓ checkpoints/final_model.pth")
print("✓ logs/training_log.csv")
print("✓ figures/loss_curve.png")
print("✓ figures/accuracy_curve.png")

print("-" * 70)

print("\nTraining Finished Successfully.")
print("=" * 70)