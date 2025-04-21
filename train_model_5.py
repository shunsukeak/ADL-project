# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import pandas as pd
from dataloader_4 import DamageDataset  # Part 4 で定義したクラス

class MultimodalDamageClassifier(nn.Module):
    def __init__(self, num_attr_classes, embedding_dims=None, num_classes=4):
        super().__init__()
        base_model = models.resnet34(weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(base_model.children())[:-1])  # 出力: (B, 512, 1, 1)
        if embedding_dims is None:
            embedding_dims = [8 for _ in num_attr_classes]
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cls, emb_dim)
            for num_cls, emb_dim in zip(num_attr_classes, embedding_dims)
        ])
        emb_total_dim = sum(embedding_dims)
        self.classifier = nn.Sequential(
            nn.Linear(512 + emb_total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, attr_ids):
        img_feat = self.resnet(image).squeeze()
        emb_list = [emb(attr_ids[:, i]) for i, emb in enumerate(self.embeddings)]
        attr_feat = torch.cat(emb_list, dim=1)
        fused = torch.cat([img_feat, attr_feat], dim=1)
        return self.classifier(fused)

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, log_file="train_log.txt"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with open(log_file, "w") as log:
        for epoch in range(num_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for images, attrs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, attrs, labels = images.to(device), attrs.to(device), labels.to(device)
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
            print(f"✅ Epoch {epoch+1}: Loss={total_loss:.4f} | Train Acc={acc:.4f}")
            if val_loader:
                val_acc = evaluate(model, val_loader, device)
                log.write(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}\n")
                log.flush()
    torch.save(model.state_dict(), "new_best_model.pt")
    print("💾 Model saved to new_best_model.pt")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, attrs, labels in loader:
            images, attrs, labels = images.to(device), attrs.to(device), labels.to(device)
            outputs = model(images, attrs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"🧪 Validation Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    csv_path = "training_dataset_with_labels_and_features_new.csv"
    df = pd.read_csv(csv_path)
    attr_cols = [col for col in df.columns if col.endswith("_id")]
    num_attr_classes = [df[col].nunique() for col in attr_cols]

    dataset = DamageDataset(csv_path)
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalDamageClassifier(num_attr_classes=num_attr_classes)
    train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4)
