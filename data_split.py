import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

label_csv_path = "dataset_csv/slide_labels.csv"
save_dir = "dataset_csv/splits/task1"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(label_csv_path)
assert "slide_id" in df.columns and "label" in df.columns, "slide_labels.csv에 'slide_id', 'label' 컬럼 있어함요"

X = df["slide_id"].values
y = df["label"].values 

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=220)

for i, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
    skf_inner = StratifiedKFold(n_splits=9, shuffle=True, random_state=220)
    train_idx_inner, val_idx_inner = next(skf_inner.split(X_train_val, y_train_val))
    
    train_idx = train_val_idx[train_idx_inner]
    val_idx = train_val_idx[val_idx_inner]

    bool_df = pd.DataFrame(index=X, columns=["train", "val", "test"], data=False)
    bool_df.loc[X[train_idx], "train"] = True
    bool_df.loc[X[val_idx], "val"] = True
    bool_df.loc[X[test_idx], "test"] = True
    bool_df.index.name = "slide_id"
    bool_df.to_csv(f"{save_dir}/splits_{i}_bool.csv")

    def count_class(indices):
        y_sub = y[np.isin(X, X[indices])]
        normal = np.sum(y_sub == 0)
        tumor = np.sum(y_sub == 1)
        return normal, tumor

    train_n, train_t = count_class(train_idx)
    val_n, val_t = count_class(val_idx)
    test_n, test_t = count_class(test_idx)

    desc_df = pd.DataFrame({
        "train": [train_n, train_t],
        "val": [val_n, val_t],
        "test": [test_n, test_t],
    }, index=["normal_tissue", "tumor_tissue"])
    desc_df.to_csv(f"{save_dir}/splits_{i}_descriptor.csv")

    max_len = max(len(train_idx), len(val_idx), len(test_idx))
    pad = lambda arr: np.pad(arr, (0, max_len - len(arr)), constant_values="")
    split_df = pd.DataFrame({
        "train": pad(X[train_idx]),
        "val": pad(X[val_idx]),
        "test": pad(X[test_idx]),
    })
    split_df.to_csv(f"{save_dir}/splits_{i}.csv", index_label="index")

print("10 split finish")