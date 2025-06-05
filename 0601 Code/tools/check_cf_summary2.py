import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import os
from data.dataset_loader import GraphDatasetLoader
from subgraph_selector.utils.choose_node import ChooseNodeSelector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_dir", type=str, required=True, help="Stage folder, e.g. stage2_y_edge_0.3")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Actor")
    parser.add_argument("--node_ratio", type=str, default="auto", help="'auto' for automatic calculation or a numeric value to manually set node selection ratio")
    parser.add_argument("--edge_ratio", type=float, default=0.3, help="Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'")
    parser.add_argument("--mask_type", type=str, default="train", choices=["train", "test", "all"], help="Mask type for node selection")
    args = parser.parse_args()

    # 讀取 csv
    csv_path = os.path.join("saved", args.stage2_dir, "CFExplainer", args.dataset, "1_GCN2Classifier", "node_record.csv")
    df = pd.read_csv(csv_path)

    # 取得方法欄位
    method_columns = [col for col in df.columns if col not in ["Dataset", "Model", "Node", "Explainer"]]

    # Load dataset
    loader = GraphDatasetLoader()
    data, _, _ = loader.load_dataset(args.dataset)

    for method in method_columns:
        # Select nodes to explain
        node_selector = ChooseNodeSelector(data, node_ratio=args.node_ratio, edge_ratio=args.edge_ratio, strategy=method, mask_type=args.mask_type)
        node_indices = node_selector.select_nodes()

        print(f"Selected {len(node_indices)} nodes using strategy: {method}")

        cf_num =  (df[method] == 1).sum()
        not_cf_num = len(node_indices) - cf_num

        print(f"Method: {method}")
        print("  成功生成CF的節點數量:", cf_num)
        print("  沒有生成CF的節點數量:", not_cf_num)
        print()

if __name__ == "__main__":
    main()
