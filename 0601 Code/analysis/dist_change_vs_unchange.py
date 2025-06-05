# 經過模型有改變預測與沒改變預測的節點，並且計算這些節點與測試節點的距離
# Mann-Whitney U（又稱 Wilcoxon rank-sum test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from data.dataset_loader import GraphDatasetLoader
from models.basic_GCN import GCN2Classifier
import argparse
from utils.device import DEVICE
from subgraph_selector.remaining_graph import RemainingGraphConstructor
from subgraph_selector.explainer_selector import ExplainerEdgeSelector



class DistanceChecker:
    def __init__(self, dataset, distance_csv, remain_model, original_model, remain_run_mode, original_run_mode, orig_data, remain_data):
        
        self.csv_path = distance_csv
        self.dataset = dataset
        self.remain_model = remain_model
        self.original_model_name = original_model
        self.remain_run_mode = remain_run_mode
        self.original_run_mode = original_run_mode
        self.orig_data = orig_data
        self.remain_data = remain_data
        self.model = self.load_model(self.remain_run_mode, self.remain_model)
        self.original_model = self.load_model(self.original_run_mode, self.original_model_name)



    def load_model(self, run_mode, model_name):
        config_path = os.path.join("saved", run_mode, "model", self.dataset, f"{model_name}_config.pth")
        model_path = os.path.join("saved", run_mode, "model", self.dataset, f"{model_name}.pth")
        config = torch.load(config_path)

        import inspect
        allowed_params = inspect.signature(GCN2Classifier.__init__).parameters
        allowed_keys = set(allowed_params.keys()) - {"self"}

        model_config = {
            key: config.get(key, self.orig_data.x.size(1) if key == "in_channels" else int(self.orig_data.y.max()) + 1 if key == "out_channels" else 64)
            for key in allowed_keys
        }

        model = GCN2Classifier(**model_config).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model


    # 改! unchange 跟 change 不是跟 y 比, 是 original 跟 remain 比
    def get_unchanged_nodes(self):
        with torch.no_grad():
            orig_out = self.original_model(self.orig_data.x, self.orig_data.edge_index)
            remain_out = self.model(self.remain_data.x, self.remain_data.edge_index)

            orig_preds = orig_out.argmax(dim=1)
            remain_preds = remain_out.argmax(dim=1)

        test_idx = self.orig_data.test_mask.nonzero(as_tuple=True)[0]
        unchanged_idx = test_idx[(orig_preds[test_idx] == remain_preds[test_idx])]
        changed_idx = test_idx[(orig_preds[test_idx] != remain_preds[test_idx])]
        print(f"Unchanged nodes: {len(unchanged_idx)}, Changed nodes: {len(changed_idx)}")
        return unchanged_idx.tolist(), changed_idx.tolist()


    def summarize_distances(self, target_strategy=None):
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

        save_dir = os.path.join("saved", "analysis", "dist_change_vs_unchange")
        os.makedirs(save_dir, exist_ok=True)

        csv_path_unchanged = os.path.join(save_dir, f"{self.dataset}_{self.remain_model}_unchanged.csv")
        csv_path_changed = os.path.join(save_dir, f"{self.dataset}_{self.remain_model}_changed.csv")
        df_unchanged.to_csv(csv_path_unchanged, index=False)
        df_changed.to_csv(csv_path_changed, index=False)
        print(f"Saved: {csv_path_unchanged}")
        print(f"Saved: {csv_path_changed}")

        self.plot_distance_summary(df_unchanged, df_changed, save_dir, target_strategy)
        self.plot_distance_histogram(df_unchanged, df_changed, save_dir, target_strategy)

        return df_unchanged, df_changed

    def plot_distance_summary(self, df1, df2, save_dir, strategy):
        topk_metrics = sorted([
            col for col in df1.columns
            if col.startswith(f"{strategy}_top") and col.endswith("_avg_dist")
        ])

        metrics = [(f"{strategy}_min_dist", "Minimum")]
        metrics += [(col, f"Top-{col.split('_')[1][3:]} Avg") for col in topk_metrics]

        data_to_plot, labels, colors = [], [], []
        p_values, stat_results = [], []

        for col, label in metrics:
            if col not in df1.columns or col not in df2.columns:
                continue

            unchanged_data = df1[col].dropna().values
            changed_data = df2[col].dropna().values

            data_to_plot.append(unchanged_data)
            labels.append(f"{label}\nUnchanged")
            colors.append("#1f77b4")

            data_to_plot.append(changed_data)
            labels.append(f"{label}\nChanged")
            colors.append("#ff7f0e")

            # Mann-Whitney U Test
            u_stat, p_val = mannwhitneyu(unchanged_data, changed_data, alternative="two-sided")
            p_values.append(p_val)

            stat_results.append({
                "Metric": label,
                "Strategy": strategy,
                "Unchanged_Mean": np.mean(unchanged_data),
                "Changed_Mean": np.mean(changed_data),
                "Unchanged_Median": np.median(unchanged_data),
                "Changed_Median": np.median(changed_data),
                "MannWhitneyU_p": p_val
            })

        fig, ax = plt.subplots(figsize=(10, 6))
        box = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("none")

        for median in box['medians']:
            median.set_color("#444444")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1f77b4", label="Unchanged"),
            Patch(facecolor="#ff7f0e", label="Changed")
        ]
        ax.legend(handles=legend_elements, loc="upper left")


        # 顯示 p-value 標註
        for i, p in enumerate(p_values):
            x = 2 * i + 1.5
            y = max(np.max(data_to_plot[2 * i]), np.max(data_to_plot[2 * i + 1])) * 1.05
            label = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"p={p:.2f}"
            ax.text(x, y, label, ha="center", va="bottom", fontsize=10, color="red")

        ax.set_title(f"{strategy.replace('_', ' ').title()} – Distance Comparison", fontsize=12)
        ax.set_ylabel("Distance")
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f"{self.dataset}_{self.remain_model}.png")
        stat_path = os.path.join(save_dir, f"{self.dataset}_{self.remain_model}_stat.csv")


        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to: {plot_path}")

        pd.DataFrame(stat_results).to_csv(stat_path, index=False)
        print(f"Saved statistical results to: {stat_path}")


    def plot_distance_histogram(self, df_unchanged, df_changed, save_dir, strategy):
        import matplotlib.pyplot as plt

        dist_col = f"{strategy}_min_dist"
        if dist_col not in df_unchanged.columns or dist_col not in df_changed.columns:
            print(f"Column {dist_col} not found.")
            return

        # Round distance to nearest int for histogram
        changed = df_changed[dist_col].dropna().round().astype(int)
        unchanged = df_unchanged[dist_col].dropna().round().astype(int)

        # Count frequency
        bins = sorted(set(changed.tolist() + unchanged.tolist()))
        changed_counts = changed.value_counts().reindex(bins, fill_value=0)
        unchanged_counts = unchanged.value_counts().reindex(bins, fill_value=0)

        # Plot side-by-side bars
        x = np.arange(len(bins))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, unchanged_counts.values, width, label="Unchanged", color="#1f77b4")
        ax.bar(x + width/2, changed_counts.values, width, label="Changed", color="#ff7f0e")

        ax.set_xlabel("Minimum Distance")
        ax.set_ylabel("Node Count")
        ax.set_title(f"Changed vs Unchanged (by Min Distance)")
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{self.dataset}_{self.remain_model}_bar_by_min_dist.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved histogram to: {save_path}")


# python analysis/dist_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --original_run_mode original --remain_model 1_GCN2Classifier --original_model 1_GCN2Classifier --strategy random  --explainer_name GNNExplainer --fraction 0.1 --fraction_feat 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--remain_run_mode", type=str, default="remove_from_GNNExplainer")  # remain model 的 run_mode
    parser.add_argument("--original_run_mode", type=str, default="original")         # original model 的 run_mode
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument("--remain_model", type=str, default="1_GCN2Classifier")
    parser.add_argument("--original_model", type=str, default="original_GCN2Classifier")

    # for remain data
    parser.add_argument("--base_dir", type=str, default="saved/stage2_y_edge_0.3")
    parser.add_argument("--explainer_name", type=str, required=True)
    # parser.add_argument("--node_choose", type=str, default="top_pagerank")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--fraction_feat", type=float, default=0.0)
    parser.add_argument("--feature_to_node", action="store_true")
    parser.add_argument("--same_feat", action="store_true")
    parser.add_argument("--feature_type", type=str, default="categorical")

    args = parser.parse_args()

    loader = GraphDatasetLoader()
    orig_data, num_features, num_classes, _, _ = loader.load_dataset(args.dataset)
    
    # remain data
    selector = ExplainerEdgeSelector(
    base_dir=args.base_dir,
    explainer_name=args.explainer_name,
    dataset_name=args.dataset,
    node_choose=args.strategy,
    top_k_percent=args.fraction,
    feature_type=args.feature_type,
    device=DEVICE,
    top_k_percent_feat=args.fraction_feat,
    use_feature_to_node=args.feature_to_node
    )
    selector.load_data()
    selected_edges, selected_feat_ids = selector.select_edges(
        num_ori_edges=orig_data.num_edges,
        num_ori_nodes=orig_data.num_nodes,
        ori_num_features=orig_data.x.size(1),
        return_feat_ids=True
    )
    selected_feat = None
    if not args.feature_to_node and args.fraction_feat > 0:
        selected_feat = selector.select_node_features(
            num_ori_nodes=orig_data.num_nodes,
            same_feat=args.same_feat
        )

    remaining_graph_constructor = RemainingGraphConstructor(orig_data, selected_edges, selected_feat_mask=selected_feat, device=DEVICE)
    remain_data = remaining_graph_constructor.get_remaining_graph()



    checker = DistanceChecker(
        dataset=args.dataset,
        distance_csv=os.path.join("saved", "analysis", "dist_from_test_to_selected", f"{args.dataset}.csv"),
        remain_model=args.remain_model,
        original_model=args.original_model,
        remain_run_mode=args.remain_run_mode,
        original_run_mode=args.original_run_mode,
        orig_data=orig_data,
        remain_data=remain_data
    )
    checker.summarize_distances(target_strategy=args.strategy)