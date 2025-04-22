# === resnet_shape_fusion.py ===
# 統合構成: 画像 (ResNet) + 形状特徴 (MLP) → 被害分類
# cropしたものを利用

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import random
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# === Dataset: 画像 + 数値特徴 ===
class ShapeFusionDataset(Dataset):
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].tolist()
        self.shape_cols = ["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity"]
        self.shape_feats = self.df[self.shape_cols].fillna(0).values.astype(np.float32)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
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
        image = self._load_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        shape_feat = torch.tensor(self.shape_feats[idx], dtype=torch.float)
        return image, shape_feat, label

# === Model: ResNet + Shape Feature Fusion ===
class ResNetWithShape(nn.Module):
    def __init__(self, shape_feat_dim, num_classes=4):
        super().__init__()
        base = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512 + shape_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_shape):
        feat_img = self.resnet(x_img).squeeze()
        x = torch.cat([feat_img, x_shape], dim=1)
        return self.classifier(x)

# === Training ===
def train(model, train_loader, val_loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for img, shape, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img, shape, label = img.to(device), shape.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img, shape)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        train_losses.append(total_loss)
        print(f"✅ Epoch {epoch+1}: Loss={total_loss:.4f}, Train Acc={train_acc:.4f}")

        # Validation
        acc, preds, gts = evaluate(model, val_loader, device)
        val_accuracies.append(acc)

    # Plot & Save
    torch.save(model.state_dict(), "crop_model_shape_fusion.pt")
    print("✅ Model saved: crop_model_shape_fusion.pt")
    plot_results(train_losses, val_accuracies)
    save_report(gts, preds)

# === Evaluation ===
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for img, shape, label in loader:
            img, shape = img.to(device), shape.to(device)
            output = model(img, shape)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.numpy())
            correct += (pred.cpu() == label).sum().item()
            total += label.size(0)
    acc = correct / total
    print(f"🧪 Val Accuracy: {acc:.4f}")
    return acc, all_preds, all_labels

# === Utilities ===
def plot_results(losses, accs):
    plt.figure()
    plt.plot(losses, label="Train Loss")
    plt.plot(accs, label="Val Acc")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig("crop_training_curve.png")
    print("📈 Saved: crop_training_curve.png")

def save_report(y_true, y_pred):
    labels = ["no", "minor", "major", "destroyed"]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.savefig("crop_confusion_matrix.png")
    print("📊 Saved: crop_confusion_matrix.png")
    with open("crop_val_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=labels))
    print("📄 Saved: crop_val_report.txt")

# === Entry ===
if __name__ == "__main__":
    csv = "cropped_dataset_with_shapes.csv"
    df = pd.read_csv(csv)
    # df["disaster"] = df["file"].apply(lambda x: x.split("_")[0])  # 災害名を抽出 file -> image path
    df["disaster"] = df["image_path"].apply(lambda x: os.path.basename(x).split("_")[0])
    all_events = sorted(df["disaster"].unique())
    random.seed(42)
    random.shuffle(all_events)

    split_idx = int(0.8 * len(all_events))
    train_events = all_events[:split_idx]
    val_events = all_events[split_idx:]

    train_df = df[df["disaster"].isin(train_events)].reset_index(drop=True)
    val_df = df[df["disaster"].isin(val_events)].reset_index(drop=True)

    train_df.to_csv("crop_train_event_split.csv", index=False)
    val_df.to_csv("crop_val_event_split.csv", index=False)

    train_loader = DataLoader(ShapeFusionDataset("crop_train_event_split.csv"), batch_size=32, shuffle=True)
    val_loader = DataLoader(ShapeFusionDataset("crop_val_event_split.csv"), batch_size=32)

    model = ResNetWithShape(shape_feat_dim=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, val_loader, device, epochs=10)
