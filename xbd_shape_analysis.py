import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
df = pd.read_csv(".xbd_shape_features.csv")

# 利用する特徴量
feature_cols = ["area", "perimeter", "aspect_ratio", "extent_ratio", "convexity"]

# subtype の順序を明示
subtype_order = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Boxplot の描画
for col in feature_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="subtype", y=col, data=df, order=subtype_order)
    plt.title(f"Distribution of {col} by subtype")
    plt.xlabel("Subtype")
    plt.ylabel(col)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"./resnet_baseline/boxplot_{col}.png")
    plt.close()

# subtype ごとの平均値を表示
grouped_stats = df.groupby("subtype")[feature_cols].mean().loc[subtype_order]