import pandas as pd
import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.utils import resample

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}

def load_dataset(csv_path, image_folder, sample_size=None, balance=True, img_size=224):
    print("🔥 LOADER RUNNING 🔥")

    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")

    df["diagnosis"] = (
        df["diagnosis"].astype(str).str.strip().str.replace(r'\s+', '', regex=True)
    )
    df = df[df["diagnosis"].isin(LABEL_MAP)].reset_index(drop=True)
    print(f"After label filter: {len(df)} rows | classes: {df['diagnosis'].unique()}")

    if sample_size:
        per_class = sample_size // len(LABEL_MAP)
        parts = []
        for cls in LABEL_MAP:
            subset = df[df["diagnosis"] == cls]
            parts.append(subset.sample(min(len(subset), per_class), random_state=42))
        df = pd.concat(parts).reset_index(drop=True)
        print(f"After capping ({sample_size} total): {len(df)} rows")

    images, labels, missing = [], [], 0

    for i in range(len(df)):
        row = df.iloc[i]
        img = cv2.imread(os.path.join(image_folder, row["image_name"]))
        if img is None:
            missing += 1
            continue
        # ✅ No /255.0 — EfficientNetB0 handles normalization internally
        img = cv2.resize(img, (img_size, img_size)).astype(np.float32)
        images.append(img)
        labels.append(LABEL_MAP[row["diagnosis"]])

    if missing:
        print(f"⚠️  Skipped {missing} missing images")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels)

    if balance:
        max_count = max(np.bincount(y))
        X_bal, y_bal = [], []
        for cls in range(len(LABEL_MAP)):
            idx = np.where(y == cls)[0]
            X_cls = resample(X[idx], replace=True, n_samples=max_count, random_state=42)
            X_bal.append(X_cls)
            y_bal.append(np.full(max_count, cls))
        X = np.concatenate(X_bal)
        y = np.concatenate(y_bal)
        perm = np.random.RandomState(42).permutation(len(X))
        X, y = X[perm], y[perm]
        print(f"After balancing: {X.shape} | class counts: {np.bincount(y)}")

    y_onehot = to_categorical(y, num_classes=len(LABEL_MAP))
    print(f"Final dataset: {X.shape}  {y_onehot.shape}")
    return X, y_onehot