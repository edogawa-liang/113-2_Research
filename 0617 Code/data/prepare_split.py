import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# prepare_splits.py
import os
import csv
import argparse
import numpy as np
from data.dataset_loader import GraphDatasetLoader
from utils.device import DEVICE

def split_and_save(data, dataset_name, repeat_id, train_ratio=0.2, val_ratio=0.05, test_ratio=0.1, seed=42):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)

    np.random.seed(seed)
    np.random.shuffle(indices)

    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)
    num_test = int(test_ratio * num_nodes)

    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train+num_val]
    test_idx = indices[num_train+num_val:num_train+num_val+num_test]

    # init mask → 全部先是 unknown_mask = 1
    train_mask = np.zeros(num_nodes, dtype=int)
    val_mask = np.zeros(num_nodes, dtype=int)
    test_mask = np.zeros(num_nodes, dtype=int)
    unknown_mask = np.ones(num_nodes, dtype=int)

    # set mask
    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1

    # update unknown
    unknown_mask[train_idx] = 0
    unknown_mask[val_idx] = 0
    unknown_mask[test_idx] = 0

    # Save to CSV
    os.makedirs(f"saved/splits/{dataset_name}", exist_ok=True)
    save_path = f"saved/splits/{dataset_name}/split_{repeat_id}.csv"

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "train_mask", "val_mask", "test_mask", "unknown_mask"])
        for i in range(num_nodes):
            writer.writerow([i, train_mask[i], val_mask[i], test_mask[i], unknown_mask[i]])
    
    print(f"[Split {repeat_id}] Saved to {save_path}")

    # Print statistics
    train_ratio_real = train_mask.sum() / num_nodes
    val_ratio_real = val_mask.sum() / num_nodes
    test_ratio_real = test_mask.sum() / num_nodes
    unknown_ratio_real = unknown_mask.sum() / num_nodes

    print(f"    → Train %: {train_ratio_real*100:.2f}%")
    print(f"    → Val %:   {val_ratio_real*100:.2f}%")
    print(f"    → Test %:  {test_ratio_real*100:.2f}%")
    print(f"    → Unknown %: {unknown_ratio_real*100:.2f}%")
    print("")



# python data/prepare_split.py --dataset Cora --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data splits and save as CSV.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--num_repeat", type=int, default=10, help="Number of repeats")
    parser.add_argument("--train_ratio", type=float, default=0.2, help="Train ratio")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio")
    args = parser.parse_args()

    # Load data
    loader = GraphDatasetLoader() # 只是要分split node，不需要 normalize
    data, _, _, _, _ = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)

    # Repeat
    for repeat in range(args.num_repeat):
        seed = 100 + repeat
        split_and_save(
            data, args.dataset, repeat_id=repeat,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            seed=seed
        )


import torch
import pandas as pd

def load_split_csv(dataset_name, repeat_id, device):
    path = f"saved/splits/{dataset_name}/split_{repeat_id}.csv"
    df = pd.read_csv(path)

    # 自動決定 num_nodes
    num_nodes = df["node_id"].max() + 1

    # 初始化 mask
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    unknown_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    # 填 mask
    for _, row in df.iterrows():
        node_id = int(row["node_id"])
        if row["train_mask"] == 1:
            train_mask[node_id] = True
        elif row["val_mask"] == 1:
            val_mask[node_id] = True
        elif row["test_mask"] == 1:
            test_mask[node_id] = True
        elif row["unknown_mask"] == 1:
            unknown_mask[node_id] = True

    return train_mask, val_mask, test_mask, unknown_mask
