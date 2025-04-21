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
log_file = "train_log.txt"

# === ステップ 1: xBD + OSM + 属性補完（Part 1） ===
if not os.path.exists(enriched_dir):
    from preprocess_buildings_1 import process_all_disasters
    process_all_disasters(tier1_label_dir, tier3_label_dir, osm_root, enriched_dir)

# === ステップ 2: 学習用データ作成（Part 2） ===
create_training_dataset_from_enriched(enriched_dir, image_root, raw_csv)

# === ステップ 3: 属性エンコード & embedding次元調整（Part 3） ===
df, num_attr_classes, embedding_dims = encode_attributes_and_save(raw_csv, encoded_csv)

# === ステップ 4 & 5: データセットロード & モデル学習（Part 4+5） ===
dataset = DamageDataset(encoded_csv)
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalDamageClassifier(num_attr_classes, embedding_dims)
train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, log_file=log_file)
