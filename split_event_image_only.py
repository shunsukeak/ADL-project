import pandas as pd
import random

df = pd.read_csv("./training_dataset_image_only.csv")
df["disaster"] = df["file"].apply(lambda x: x.split("_")[0])

all_disasters = sorted(df["disaster"].unique())
random.seed(42)
random.shuffle(all_disasters)

split_idx = int(0.8 * len(all_disasters))
train_disasters = all_disasters[:split_idx]
val_disasters = all_disasters[split_idx:]

train_df = df[df["disaster"].isin(train_disasters)].reset_index(drop=True)
val_df = df[df["disaster"].isin(val_disasters)].reset_index(drop=True)

train_df.to_csv("train_event_image_only.csv", index=False)
val_df.to_csv("val_event_image_only.csv", index=False)

print(f"✅ train: {len(train_df)} samples, val: {len(val_df)} samples")
