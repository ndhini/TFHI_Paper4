import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------
# Dataset Path
# ---------------------------------------------------
DATA_PATH = "data/RML2016.10a_dict.pkl"

print("=" * 60)
print("Loading RadioML2016.10A Dataset")
print("=" * 60)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f, encoding="latin1")

print("Dataset Loaded Successfully!")

# ---------------------------------------------------
# Extract Modulation Types
# ---------------------------------------------------
mods = sorted(list(set(k[0] for k in dataset.keys())))

# ---------------------------------------------------
# Extract SNR Levels
# ---------------------------------------------------
snrs = sorted(list(set(k[1] for k in dataset.keys())))

print("\nModulation Types")
print(mods)

print("\nSNR Levels")
print(snrs)

# ---------------------------------------------------
# Prepare Data
# ---------------------------------------------------
X = []
Y = []
SNR = []

for mod in mods:
    for snr in snrs:

        samples = dataset[(mod, snr)]

        X.append(samples)

        Y.extend([mod] * samples.shape[0])

        SNR.extend([snr] * samples.shape[0])

# ---------------------------------------------------
# Convert to Arrays
# ---------------------------------------------------
X = np.vstack(X)

Y = np.array(Y)

SNR = np.array(SNR)

print("\nDataset Statistics")
print("----------------------")
print("Samples :", X.shape)
print("Labels  :", Y.shape)
print("SNR     :", SNR.shape)

# ---------------------------------------------------
# Encode Labels
# ---------------------------------------------------
encoder = LabelEncoder()

Y_encoded = encoder.fit_transform(Y)

print("\nEncoded Labels")

for i, m in enumerate(encoder.classes_):
    print(i, "->", m)

# ---------------------------------------------------
# Train Validation Test Split
# ---------------------------------------------------
X_train, X_temp, y_train, y_temp, snr_train, snr_temp = train_test_split(
    X,
    Y_encoded,
    SNR,
    test_size=0.30,
    random_state=42,
    stratify=Y_encoded
)

X_valid, X_test, y_valid, y_test, snr_valid, snr_test = train_test_split(
    X_temp,
    y_temp,
    snr_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split")
print("--------------------")

print("Train :", X_train.shape)

print("Validation :", X_valid.shape)

print("Test :", X_test.shape)

# ---------------------------------------------------
# Convert to Torch
# ---------------------------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_valid = torch.tensor(y_valid, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

snr_train = torch.tensor(snr_train)
snr_valid = torch.tensor(snr_valid)
snr_test = torch.tensor(snr_test)

# ---------------------------------------------------
# Save Processed Dataset
# ---------------------------------------------------
torch.save({
    "X_train": X_train,
    "X_valid": X_valid,
    "X_test": X_test,

    "y_train": y_train,
    "y_valid": y_valid,
    "y_test": y_test,

    "snr_train": snr_train,
    "snr_valid": snr_valid,
    "snr_test": snr_test,

    "classes": encoder.classes_

}, "data/processed_dataset.pt")

print("\nProcessed dataset saved!")

print("=" * 60)
print("Completed Successfully")
print("=" * 60)