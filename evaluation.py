import torch
from torch.utils.data import DataLoader
from dataloader import DamageDataset
from train_model import MultimodalDamageClassifier  
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_with_metrics(model, loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, attrs, labels in loader:
            images, attrs = images.to(device), attrs.to(device)
            outputs = model(images, attrs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("✅ Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("🧾 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# データ準備
csv_path = "./training_dataset_with_labels_and_features.csv"
dataset = DamageDataset(csv_path)

# 分割（同じように）
from torch.utils.data import random_split
val_size = int(len(dataset) * 0.2)
_, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=32)

# モデル準備
df = pd.read_csv(csv_path)
num_attr_classes = [df[col].nunique() for col in df.columns if col.endswith("_id")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalDamageClassifier(num_attr_classes)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)

# 評価
class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
evaluate_with_metrics(model, val_loader, device, class_names)
