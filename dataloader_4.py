# dataloader.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np

class DamageDataset(Dataset):
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.attr_cols = [col for col in self.df.columns if col.endswith("_id")]
        self.image_paths = self.df["pre_image_path"].tolist()
        self.labels = self.df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_image_cv2(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Image not found: {path}")
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        return torch.tensor(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        img = self._load_image_cv2(self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        attr_tensor = torch.tensor(self.df.iloc[idx][self.attr_cols].values, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, attr_tensor, label
