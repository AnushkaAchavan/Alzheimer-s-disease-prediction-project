import pandas as pd
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}


# ─────────────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, dataframe, image_folder, img_size=256, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_folder, row["image_name"])
        label = LABEL_MAP[row["diagnosis"]]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32)

        # Normalize
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = img / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.augment:
            import torchvision.transforms.functional as TF
            import random

            if random.random() > 0.5:
                img = TF.hflip(img)

            img = TF.rotate(img, random.uniform(-18, 18))
            img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
            img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))

        return img, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# Loader Function
# ─────────────────────────────────────────────────────────────
def load_dataset(csv_path, image_folder, sample_size=None,
                 balance=True, img_size=256,
                 batch_size=16, num_workers=0):

    print("🔥 LOADER RUNNING 🔥")

    df = pd.read_csv(csv_path)

    df["diagnosis"] = (
        df["diagnosis"].astype(str)
        .str.strip()
        .str.replace(r'\s+', '', regex=True)
    )

    df = df[df["diagnosis"].isin(LABEL_MAP)].reset_index(drop=True)

    # Sampling
    if sample_size:
        per_class = sample_size // len(LABEL_MAP)
        parts = []
        for cls in LABEL_MAP:
            subset = df[df["diagnosis"] == cls]
            parts.append(subset.sample(min(len(subset), per_class), random_state=42))
        df = pd.concat(parts).reset_index(drop=True)

    # Remove missing images
    df = df[df["image_name"].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))]
    df = df.reset_index(drop=True)

    # Balance
    if balance:
        max_count = df["diagnosis"].value_counts().max()
        balanced = []
        for cls in LABEL_MAP:
            cls_df = df[df["diagnosis"] == cls]
            cls_resampled = resample(cls_df, replace=True, n_samples=max_count, random_state=42)
            balanced.append(cls_resampled)
        df = pd.concat(balanced).sample(frac=1).reset_index(drop=True)

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["diagnosis"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=2/3, stratify=temp_df["diagnosis"], random_state=42)

    # Class weights
    y_train = train_df["diagnosis"].map(LABEL_MAP).values
    counts = np.bincount(y_train)
    class_weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(counts)

    y_test = test_df["diagnosis"].map(LABEL_MAP).values

    # Datasets
    train_ds = MRIDataset(train_df, image_folder, img_size, augment=True)
    val_ds   = MRIDataset(val_df, image_folder, img_size, augment=False)
    test_ds  = MRIDataset(test_df, image_folder, img_size, augment=False)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights, y_test