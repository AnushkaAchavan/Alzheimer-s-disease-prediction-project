print("🔥 TRAINING STARTED 🔥")

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm   # ✅ ADDED

from dataset.loader import load_dataset
from model.ensemble_model import FeatureConcatEnsemble

# ── GPU Setup ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "processed_data/metadata/final_metadata.csv"
IMAGE_PATH  = "processed_data/slices"
SAVE_DIR    = "outputs"
SAMPLE_SIZE = 11000
BATCH_SIZE  = 8
IMG_SIZE    = 256
NUM_CLASSES = 3
LABELS      = ["CN", "MCI", "AD"]
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 20

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader, class_weights, y_test = load_dataset(
    CSV_PATH, IMAGE_PATH,
    sample_size=SAMPLE_SIZE,
    balance=True,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=0
)
class_weights = class_weights.to(device)

# ── Model ─────────────────────────────────────────────────────────────────────
model = FeatureConcatEnsemble(num_classes=NUM_CLASSES, freeze_backbones=True)
model = model.to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Training Function ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(loader, desc="Training", dynamic_ncols=True)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct/total)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs = [], []

    loop = tqdm(loader, desc="Validation", dynamic_ncols=True)

    with torch.no_grad():
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(1)

            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            loop.set_postfix(loss=loss.item(), acc=correct/total)

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_probs)


def plot_history(history, phase_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_acc'], label='Train Acc')
    ax1.plot(history['val_acc'],   label='Val Acc')
    ax1.set_title(f'Accuracy — {phase_name}')
    ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'],   label='Val Loss')
    ax2.set_title(f'Loss — {phase_name}')
    ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "plots", f"curves_{phase_name}.png"), dpi=150)
    plt.show()

# ── Phase 1 ──────────────────────────────────────────────────────────────────
print("\n📌 Phase 1: Training fusion head (backbones frozen)...")
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

best_val_acc = 0
patience_counter = 0
history1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(PHASE1_EPOCHS):
    print(f"\nEpoch {epoch+1}/{PHASE1_EPOCHS}")
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step(vl_loss)

    history1['train_loss'].append(tr_loss)
    history1['train_acc'].append(tr_acc)
    history1['val_loss'].append(vl_loss)
    history1['val_acc'].append(vl_acc)

    print(f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_phase1.pth"))
        print(f"💾 Saved best model")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 7:
            print("⏹ Early stopping")
            break

plot_history(history1, "Phase1")

# ── Phase 2 ──────────────────────────────────────────────────────────────────
print("\n📌 Phase 2: Fine-tuning...")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_phase1.pth")))
model.unfreeze_top_layers(n=30)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

best_val_acc = 0
patience_counter = 0
history2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(PHASE2_EPOCHS):
    print(f"\nEpoch {epoch+1}/{PHASE2_EPOCHS}")
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step(vl_loss)

    history2['train_loss'].append(tr_loss)
    history2['train_acc'].append(tr_acc)
    history2['val_loss'].append(vl_loss)
    history2['val_acc'].append(vl_acc)

    print(f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_phase2.pth"))
        print(f"💾 Saved best model")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 7:
            print("⏹ Early stopping")
            break

plot_history(history2, "Phase2")

# ── Final Evaluation ─────────────────────────────────────────
print("\n📊 Final Evaluation...")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_phase2.pth")))
_, test_acc, y_pred, y_probs = evaluate(model, test_loader, criterion)

print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
print(classification_report(y_test, y_pred, target_names=LABELS))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABELS, yticklabels=LABELS)
plt.show()

# ROC
y_test_bin = label_binarize(y_test, classes=[0,1,2])
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=f"{LABELS[i]} (AUC={auc(fpr,tpr):.3f})")
plt.legend()
plt.show()

torch.save(model, os.path.join(SAVE_DIR, "ensemble_model_final.pth"))
print("🔥 TRAINING COMPLETE!")