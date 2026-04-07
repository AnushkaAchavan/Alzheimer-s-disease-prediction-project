print("TRAINING STARTED")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm   

from dataset.loader import load_dataset
from model.ensemble_model import FeatureConcatEnsemble

# ── GPU Setup ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ───────────────────────────────────────────
CSV_PATH = "processed_data/metadata/final_metadata.csv"
IMAGE_PATH = "processed_data/slices"
SAVE_DIR = "outputs"

SAMPLE_SIZE = 15000
BATCH_SIZE = 12
IMG_SIZE = 256

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 15

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────
train_loader, val_loader, test_loader, class_weights, y_test = load_dataset(
    CSV_PATH,
    IMAGE_PATH,
    sample_size=SAMPLE_SIZE,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE
)

class_weights = class_weights.to(device)

# ── Model ─────────────────────────────────────────────
model = FeatureConcatEnsemble(num_classes=3, freeze_backbones=True)
model = model.to(device)

# ── Training Functions ───────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(loader, desc="Training", leave=False)  # 🔥 progress bar

    for imgs, labels in loop:

        if imgs.size(0) == 1:
            continue

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        
        loop.set_postfix(
            loss=loss.item(),
            acc=correct / total
        )

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(loader, desc="Validation", leave=False)  

    with torch.no_grad():
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

            loop.set_postfix(
                val_loss=loss.item(),
                val_acc=correct / total
            )

    return total_loss / total, correct / total


# ── Phase 1 ──────────────────────────────────────────
print("Phase 1 Training")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

for epoch in range(PHASE1_EPOCHS):
    print(f"\nEpoch {epoch+1}/{PHASE1_EPOCHS}")

    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc = evaluate(model, val_loader, criterion)

    print(f" Train Acc: {tr_acc:.4f} | Val Acc: {vl_acc:.4f}")

# ── Phase 2 ──────────────────────────────────────────
print("\n Phase 2 Fine-tuning")

model.unfreeze_top_layers(20)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

for epoch in range(PHASE2_EPOCHS):
    print(f"\nEpoch {epoch+1}/{PHASE2_EPOCHS}")

    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc = evaluate(model, val_loader, criterion)

    print(f"Train Acc: {tr_acc:.4f} | Val Acc: {vl_acc:.4f}")

# ── Test ─────────────────────────────────────────────
print("\nTesting...")

_, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.4f}")

# Save
torch.save(model, os.path.join(SAVE_DIR, "model.pth"))

print("\nDONE")