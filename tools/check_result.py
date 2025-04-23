import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.dataset_loader import GraphDatasetLoader
from models.basic_GCN import GCN2Classifier
import argparse


class DistanceChecker:
    def __init__(self, model_path, config_path, dataset, distance_csv, model_name,run_mode, device="cpu"):
        self.device = torch.device(device)
        self.data, self.num_features, self.num_classes = self.load_data(dataset)
        self.model = self.load_model(model_path, config_path)
        self.csv_path = distance_csv
        self.dataset = dataset
        self.model_name = model_name
        self.run_mode = run_mode

    def load_model(self, model_path, config_path):
        config = torch.load(config_path)
        print(f"Model config: {config}")

        import inspect
        allowed_params = inspect.signature(GCN2Classifier.__init__).parameters
        allowed_keys = set(allowed_params.keys()) - {"self"}

        model_config = {}
        for key in allowed_keys:
            if key == 'in_channels':
                model_config[key] = config.get(key, self.num_features)
            elif key == 'out_channels':
                model_config[key] = config.get(key, self.num_classes)
            else:
                model_config[key] = config.get(key, 64)

        model = GCN2Classifier(**model_config).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_data(self, dataset):
        loader = GraphDatasetLoader()
        data, num_features, num_classes = loader.load_dataset(dataset)
        return data.to(self.device), num_features, num_classes

    def get_unchanged_nodes(self):
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            preds = out.argmax(dim=1)
        test_idx = self.data.test_mask.nonzero(as_tuple=True)[0]
        unchanged_idx = test_idx[(preds[test_idx] == self.data.y[test_idx])]
        changed_idx = test_idx[(preds[test_idx] != self.data.y[test_idx])]
        print(f"Unchanged nodes: {len(unchanged_idx)}, Changed nodes: {len(changed_idx)}")
        return unchanged_idx.tolist(), changed_idx.tolist()

    def summarize_distances(self):
        df = pd.read_csv(self.csv_path)
        if "TestNode" not in df.columns:
            raise ValueError("CSV must contain 'TestNode'")

        unchanged_nodes, changed_nodes = self.get_unchanged_nodes()
        df_unchanged = df[df["TestNode"].isin(unchanged_nodes)]
        df_changed = df[df["TestNode"].isin(changed_nodes)]

        print("Distance summary for unchanged prediction nodes:")
        print(df_unchanged.describe())
        print("Distance summary for changed prediction nodes:")
        print(df_changed.describe())

        save_dir = os.path.join("saved", self.run_mode, "check_result", self.dataset)
        os.makedirs(save_dir, exist_ok=True)

        df_unchanged.to_csv(os.path.join(save_dir, f"unchanged_distances_{self.model_name}.csv"), index=False)
        df_changed.to_csv(os.path.join(save_dir, f"changed_distances_{self.model_name}.csv"), index=False)

        self.plot_distance_summary(df_unchanged, df_changed, save_dir)
        return df_unchanged, df_changed

    def plot_distance_summary(self, df1, df2, save_dir):
        k_vals = []
        for col in df1.columns:
            if col.startswith("top") and col.endswith("_avg_dist"):
                try:
                    k_str = col[len("top"):col.index("_avg_dist")]
                    k_vals.append(int(k_str))
                except ValueError:
                    continue
        k = k_vals[0] if k_vals else 3

        metrics = [
            ("min_dist", "Minimum"),
            ("mean_dist", "Mean"),
            (f"top{k}_avg_dist", f"Top-{k} Avg")
        ]

        strategies = sorted({
            col[: -len(metric_key) - 1]
            for col in df1.columns
            for metric_key, _ in metrics
            if col.endswith(metric_key)
        })

        data1 = {metric_label: [] for _, metric_label in metrics}  # unchanged
        data2 = {metric_label: [] for _, metric_label in metrics}  # changed
        valid_strategies = []

        for strategy in strategies:
            has_any = False
            for metric_key, metric_label in metrics:
                col = f"{strategy}_{metric_key}"
                v1 = df1.get(col, pd.Series(dtype=float)).mean()
                v2 = df2.get(col, pd.Series(dtype=float)).mean()
                data1[metric_label].append(v1)
                data2[metric_label].append(v2)
                if not np.isnan(v1) or not np.isnan(v2):
                    has_any = True
            if has_any:
                valid_strategies.append(strategy)

        x = np.arange(len(valid_strategies))
        width = 0.35

        fig, axs = plt.subplots(1, 3, figsize=(max(12, len(valid_strategies) * 2), 4))
        fig.suptitle("Distance from Test Nodes to Selected Nodes: Unchanged vs Changed Predictions", fontsize=14)
        colors = ['#1f77b4', '#ff7f0e']
        labels = ['Unchanged', 'Changed']

        for i, metric_label in enumerate(data1.keys()):
            ax = axs[i]
            values1 = [data1[metric_label][j] for j in range(len(valid_strategies))]
            values2 = [data2[metric_label][j] for j in range(len(valid_strategies))]

            ax.bar(x - width/2, values1, width=width, label=labels[0], color=colors[0])
            ax.bar(x + width/2, values2, width=width, label=labels[1], color=colors[1])

            ax.set_title(metric_label)
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace("_", " ").title() for s in valid_strategies], rotation=30)
            ax.set_ylabel("Average Distance")
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(save_dir, f"distance_summary_{self.model_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved bar chart to: {save_path}")




# python tools/check_result.py --dataset Actor --run_mode remove_from_GNNExplainer --distance_csv_dir stage2_y_edge_0.3/GNNExplainer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--base_dir", type=str, default="saved")
    parser.add_argument("--run_mode", type=str, default="remove_from_GNNExplainer")
    parser.add_argument("--model_name", type=str, default="1_GCN2Classifier")
    parser.add_argument("--distance_csv_dir", type=str, default="stage2_y_edge_0.3/GNNExplainer", help="Path to distance_from_test_to_selected.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_path = os.path.join(args.base_dir, args.run_mode, "model", args.dataset, f"{args.model_name}.pth")
    config_path = os.path.join(args.base_dir, args.run_mode, "model", args.dataset, f"{args.model_name}_config.pth")
    # 只是要算測試節點離被選中節點的距離，計算1_GCN2Classifier內就夠
    distance_csv_path = os.path.join(args.base_dir, args.distance_csv_dir, args.dataset, "1_GCN2Classifier", "distance_from_test_to_selected.csv")

    checker = DistanceChecker(
        model_path=model_path,
        config_path=config_path,
        dataset=args.dataset,
        distance_csv=distance_csv_path,
        model_name=args.model_name,
        run_mode=args.run_mode,
        device=args.device
    )
    checker.summarize_distances()