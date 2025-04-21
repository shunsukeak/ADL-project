import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# === Dataset: Image only ===
class ImageOnlyDataset(Dataset):
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["pre_image_path"].tolist()
        self.labels = self.df["label"].tolist()
        self.image_size = image_size
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
        image = self._load_image_cv2(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# === Model: ResNet only ===
class ResNetOnlyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base_model = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image):
        feat = self.resnet(image).squeeze()
        return self.classifier(feat)

# === Training Loop ===
def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_accuracies = [], []
    all_preds, all_labels = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_losses.append(total_loss)
        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={train_acc:.4f}")

        val_acc, preds, labels = evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)
        all_preds, all_labels = preds, labels

    torch.save(model.state_dict(), "resnet_only_model.pt")
    print("✅ Model saved: resnet_only_model.pt")

    # === Training Curve ===
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig("training_curve.png")
    print("📉 Saved training_curve.png")

    # === Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no", "minor", "major", "destroyed"],
                yticklabels=["no", "minor", "major", "destroyed"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("📊 Saved confusion_matrix.png")

    # === Classification Report ===
    report = classification_report(all_labels, all_preds,
                                   target_names=["no", "minor", "major", "destroyed"])
    with open("val_report.txt", "w") as f:
        f.write(report)
    print("📄 Saved val_report.txt")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"🧪 Val Accuracy: {acc:.4f}")
    return acc, all_preds, all_labels

# === Main ===
if __name__ == "__main__":
    train_csv = "train_event_image_only.csv"
    val_csv = "val_event_image_only.csv"

    train_dataset = ImageOnlyDataset(train_csv)
    val_dataset = ImageOnlyDataset(val_csv)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetOnlyClassifier()
    train_model(model, train_loader, val_loader, device)
