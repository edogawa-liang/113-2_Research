# prepare_splits.py
import os
import csv
import numpy as np
import torch
from data.dataset_loader import GraphDatasetLoader
from utils.device import DEVICE

# 先存下repeat_id，方便後續分割
def split_and_save(data, dataset_name, repeat_id, train_ratio=0.2, val_ratio=0.1, test_ratio=0.1, seed=42):
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

    # Save to CSV
    os.makedirs(f"saved/splits/{dataset_name}", exist_ok=True)
    save_path = f"saved/splits/{dataset_name}/split_{repeat_id}.csv"

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "split"])
        for i in train_idx:
            writer.writerow([i, "train"])
        for i in val_idx:
            writer.writerow([i, "val"])
        for i in test_idx:
            writer.writerow([i, "test"])
    
    print(f"[Split {repeat_id}] Saved to {save_path}")


if __name__ == "__main__":
    loader = GraphDatasetLoader(args.normalize)
    data, _, _, _, _ = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)

    for repeat in range(10):
        seed = 100 + repeat
        split_and_save(data, dataset, repeat_id=repeat, seed=seed)
