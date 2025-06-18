import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import numpy as np
import os
import torch


def split_unknown_to_test(dataset, use_id=0, seed=42, num_repeat=10, fraction=0.1):
    split_csv_path = f"saved/splits/{dataset}/split_{use_id}.csv"
    output_csv_path = f"saved/splits/{dataset}/split_{use_id}_test.csv"

    print(f"Loading {split_csv_path}...")
    df = pd.read_csv(split_csv_path) 

    # 取得原始 test_mask=1 的 nodes
    test_mask_nodes = df[df['test_mask'] == 1]['node_id'].values
    print(f"Original test_mask=1 nodes: {len(test_mask_nodes)} nodes")

    # 取得 candidate nodes = test_mask=1 or unknown_mask=1
    candidate_nodes = df[(df['test_mask'] == 1) | (df['unknown_mask'] == 1)]['node_id'].values
    print(f"Candidate nodes for test_mask_1~{num_repeat-1}: {len(candidate_nodes)} nodes")

    # 設定 random seed
    rng = np.random.default_rng(seed)

    # 準備結果 dataframe
    result_df = pd.DataFrame({'node_id': df['node_id']})

    # 生成 test_mask_0 → 就是原本的 test_mask
    result_df['test_mask_0'] = df['test_mask']

    # 生成 test_mask_1 ~ test_mask_N
    for repeat in range(1, num_repeat):
        num_select = len(test_mask_nodes)
        selected_nodes = rng.choice(candidate_nodes, size=num_select, replace=False)

        mask_col = df['node_id'].isin(selected_nodes).astype(int)
        result_df[f'test_mask_{repeat}'] = mask_col

        print(f"Repeat {repeat}: selected {num_select} nodes")

    # 確保 output 資料夾存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 存檔
    print(f"Saving to {output_csv_path}...")
    result_df.to_csv(output_csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split unknown to test masks")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--use_id", type=int, default=0, help="Repeat ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_repeat", type=int, default=10, help="Number of test masks to generate (including test_mask_0)")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of candidate nodes to sample for test_mask_1~N")

    args = parser.parse_args()

    split_unknown_to_test(
        dataset=args.dataset,
        use_id=args.use_id,
        seed=args.seed,
        num_repeat=args.num_repeat,
        fraction=args.fraction
    )


def load_split_test(dataset_name, use_id, repeat_id, device=None):
    path = f"saved/splits/{dataset_name}/split_{use_id}_test.csv"

    # Load csv
    df = pd.read_csv(path)

    # 欄位名
    col_name = f"test_mask_{repeat_id}"

    if col_name not in df.columns:
        raise ValueError(f"Column {col_name} not found in {path}!")

    # 取出該欄 → 轉成 tensor
    mask_tensor = torch.tensor(df[col_name].values, dtype=torch.bool)

    # 搬 device
    if device is not None:
        mask_tensor = mask_tensor.to(device)

    print(f"Loaded test mask {col_name}: {mask_tensor.shape} → (num_nodes,)")

    return mask_tensor
