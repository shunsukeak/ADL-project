import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

class DamageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform if transform else self._default_transform()

        # 属性ID列とラベル列を抽出
        self.attr_cols = [col for col in self.df.columns if col.endswith("_id")]
        self.image_paths = self.df["pre_image_path"].tolist()
        self.labels = self.df["label"].tolist()

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Optional: normalize
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                                 [0.229, 0.224, 0.225]) # ImageNet std
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 画像読み込み
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # 属性IDのTensor
        attr_values = self.df.iloc[idx][self.attr_cols].values.astype("int64")
        attr_tensor = torch.tensor(attr_values, dtype=torch.long)

        # ラベル
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, attr_tensor, label


csv_path = "./training_dataset_with_labels_and_features.csv"
dataset = DamageDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 1バッチ取得して確認
for images, attrs, labels in dataloader:
    print(images.shape)     # (B, 3, 224, 224)
    print(attrs.shape)      # (B, num_attributes)
    print(labels.shape)     # (B,)
    break
