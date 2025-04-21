# encode_attributes.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_attributes_and_save(input_csv, output_csv,
                               categorical_cols=["building", "building:material", "building:levels"],
                               max_embed_dim=16, min_embed_dim=4):
    df = pd.read_csv(input_csv)
    embed_dims = {}
    print("🔁 Encoding categorical attributes and estimating embedding dims:")
    for col in categorical_cols:
        col_clean = col.replace(":", "_") + "_id"
        values = df[col].fillna("unknown").astype(str)
        if col == "building":
            values = values.replace("yes", "unknown")
        encoder = LabelEncoder()
        df[col_clean] = encoder.fit_transform(values)
        unknown_ratio = (values == "unknown").sum() / len(values)
        dim = int(max_embed_dim - (max_embed_dim - min_embed_dim) * unknown_ratio)
        dim = max(min_embed_dim, dim)
        embed_dims[col] = dim
        print(f" - {col}: {unknown_ratio:.2%} unknown → emb_dim={dim}")
    df.to_csv(output_csv, index=False)
    return df, [df[col.replace(":", "_") + "_id"].nunique() for col in categorical_cols], \
           [embed_dims[col] for col in categorical_cols]

if __name__ == "__main__":
    encode_attributes_and_save(
        input_csv="./training_dataset_raw.csv",
        output_csv="training_dataset_with_labels_and_features_new.csv"
    )
