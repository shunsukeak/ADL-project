# run_all.py
import os
from generate_training_data_2 import create_training_dataset_from_enriched
from encode_attributes_3 import encode_attributes_and_save
from dataloader_4 import DamageDataset
from train_model_5 import train_model, MultimodalDamageClassifier
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch

# === ファイルパス設定 ===
tier1_label_dir = "/mnt/bigdisk/xbd/geotiffs/tier1/labels"
tier3_label_dir = "/mnt/bigdisk/xbd/geotiffs/tier3/labels"
osm_root = "./output"
enriched_dir = "./enriched_all_buildings"
image_root = "/mnt/bigdisk/xbd/geotiffs"
raw_csv = "./training_dataset_raw.csv"
encoded_csv = "./training_dataset_with_labels_and_features_new.csv"
log_file = "train_report.txt"

# === ステップ 1: xBD + OSM + 属性補完（Part 1） ===
print("🛠️ Step 1: Preprocessing xBD + OSM buildings ...")
if not os.path.exists(enriched_dir):
    from preprocess_buildings_1 import process_all_disasters
    process_all_disasters(tier1_label_dir, tier3_label_dir, osm_root, enriched_dir)
print("✅ Step 1 complete: enriched building CSVs generated.\n")

# === ステップ 2: 学習用データ作成（Part 2） ===
print("🛠️ Step 2: Creating training dataset with pre_image path and labels ...")
create_training_dataset_from_enriched(enriched_dir, image_root, raw_csv)
print("✅ Step 2 complete: training_dataset_raw.csv created.\n")

# === ステップ 3: 属性エンコード & embedding次元調整（Part 3） ===
print("🛠️ Step 3: Encoding categorical attributes and computing embedding dims ...")
df, num_attr_classes, embedding_dims = encode_attributes_and_save(raw_csv, encoded_csv)
print("✅ Step 3 complete: encoded CSV and embedding dims ready.\n")

# === Step 4: 災害単位で train/val を分割（event-level split） ===
print("🛠️ Step 4: Splitting data by disaster (event-level split) ...")
df = pd.read_csv(encoded_csv)
df["disaster"] = df["file"].apply(lambda x: x.split("_")[0])
all_disasters = sorted(df["disaster"].unique())
random.seed(42)
random.shuffle(all_disasters)
split_idx = int(0.8 * len(all_disasters))
train_disasters = all_disasters[:split_idx]
val_disasters = all_disasters[split_idx:]

train_df = df[df["disaster"].isin(train_disasters)].reset_index(drop=True)
val_df = df[df["disaster"].isin(val_disasters)].reset_index(drop=True)

train_df.to_csv("train_event_split.csv", index=False)
val_df.to_csv("val_event_split.csv", index=False)

print(f"✅ Step 4 complete: {len(train_df)} train / {len(val_df)} val samples\n")

# === Step 5: 学習・評価（ResNet + 属性） ===
print("🛠️ Step 5: Training and evaluating the multimodal model ...")
from dataloader import DamageDataset

train_dataset = DamageDataset("train_event_split.csv")
val_dataset = DamageDataset("val_event_split.csv")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalDamageClassifier(num_attr_classes=num_attr_classes, embedding_dims=embedding_dims)
train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, log_file=log_file)

print("🎉 All steps complete. Event-based evaluation is ready!")