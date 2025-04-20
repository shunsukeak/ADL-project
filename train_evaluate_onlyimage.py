import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import pandas as pd
from dataloader import DamageDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# === 画像のみモデル定義 ===
class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base_model = models.resnet18(weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, _):  # 属性は使わない
        img_feat = self.resnet(image).squeeze()
        out = self.classifier(img_feat)
        return out

# === トレーニングループ ===
def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, log_file="train_log_image_only.txt"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with open(log_file, "w") as log:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for images, attrs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images, attrs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = correct / total
            log.write(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Acc = {acc:.4f}\n")
            print(f"✅ Epoch {epoch+1} | Loss: {total_loss:.4f} | Train Acc: {acc:.4f}")

            if val_loader:
                val_acc = evaluate(model, val_loader, device)
                log.write(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}\n")
                log.flush()

    torch.save(model.state_dict(), "best_model_image_only.pt")
    print("💾 Model saved to best_model_image_only.pt")

# === 評価関数（分類レポート付き） ===
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, attrs, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, attrs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"🧪 Validation Accuracy: {acc:.4f}")

    print("\n✅ Classification Report:")
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("🧾 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return acc

# === 実行 ===
if __name__ == "__main__":
    csv_path = "./training_dataset_with_labels_and_features.csv"
    dataset = DamageDataset(csv_path)

    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageOnlyModel()

    train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4)
