# 測試節點的degree 數量
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.utils import k_hop_subgraph, degree
from data.dataset_loader import GraphDatasetLoader
from models.basic_GCN import GCN2Classifier
from utils.device import DEVICE
from scipy.stats import mannwhitneyu
from subgraph_selector.remaining_graph import RemainingGraphConstructor
from subgraph_selector.explainer_selector import ExplainerEdgeSelector



class TestDegreeAnalyzer:
    def __init__(self, dataset, base_dir, remain_model, original_model, remain_run_mode, original_run_mode, orig_data, remain_data):
        self.dataset = dataset
        self.remain_model = remain_model
        self.original_model_name = original_model
        self.remain_run_mode = remain_run_mode
        self.original_run_mode = original_run_mode
        self.orig_data = orig_data
        self.remain_data = remain_data

        self.csv_dir = os.path.join(base_dir, "GNNExplainer", dataset, original_model)
        self.record_path = os.path.join(self.csv_dir, "node_record.csv")

        self.model = self.load_model(self.remain_run_mode, self.remain_model)
        self.original_model = self.load_model(self.original_run_mode, self.original_model_name)

        self.save_dir = os.path.join("saved", "analysis", "test_degree")
        os.makedirs(self.save_dir, exist_ok=True)

        self.node_degrees = degree(self.orig_data.edge_index[0], self.orig_data.num_nodes).cpu().numpy()


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
    

    def analyze_test_node_degree(self, changed, unchanged):
        test_degrees = {i: self.node_degrees[i] for i in self.orig_data.test_mask.nonzero(as_tuple=True)[0].tolist()}
        changed_vals = [test_degrees[n] for n in changed if n in test_degrees]
        unchanged_vals = [test_degrees[n] for n in unchanged if n in test_degrees]

        # Boxplot
        plt.figure(figsize=(8, 6))
        box = plt.boxplot([unchanged_vals, changed_vals], patch_artist=True,
                          labels=["Unchanged", "Changed"],
                          boxprops=dict(facecolor="#1f77b4"),
                          medianprops=dict(color='black'))
        box['boxes'][1].set_facecolor('#ff7f0e')
        plt.title("Test Node Degree Distribution")
        plt.ylabel("Degree")
        y_max = max(max(unchanged_vals, default=0), max(changed_vals, default=0)) * 1.1
        u_stat, p_val = mannwhitneyu(unchanged_vals, changed_vals, alternative="two-sided")
        label = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else f"p={p_val:.2f}"
        plt.text(1.5, y_max, label, ha="center", va="bottom", fontsize=12, color="red")
        handles = [
            plt.Line2D([0], [0], color="#1f77b4", lw=10, label="Unchanged"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=10, label="Changed")
        ]
        plt.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        box_path = os.path.join(self.save_dir, f"{self.dataset}_{self.remain_model}_testnode_deg_box.png")
        plt.savefig(box_path)
        plt.close()
        print(f"Saved boxplot to: {box_path}")

        # Bar (binned)
        bins = [0, 1, 2, 4, 8, 16, 32, 64, 128]
        c_hist, _ = np.histogram(changed_vals, bins)
        u_hist, _ = np.histogram(unchanged_vals, bins)
        x = np.arange(len(bins) - 1)
        width = 0.4

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, u_hist, width, label="Unchanged", color="#1f77b4")
        plt.bar(x + width/2, c_hist, width, label="Changed", color="#ff7f0e")
        plt.xticks(x, [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)])
        plt.xlabel("Test Node Degree Range")
        plt.ylabel("# of Nodes")
        plt.title("Binned Test Node Degree")
        plt.legend()
        plt.tight_layout()
        bar_path = os.path.join(self.save_dir, f"{self.dataset}_{self.remain_model}_testnode_deg_bar.png")
        plt.savefig(bar_path)
        plt.close()
        print(f"Saved bar chart to: {bar_path}")

        # Save statistics
        stat_result = pd.DataFrame([{
            "Metric": "TestNodeDegree",
            "Unchanged_Mean": np.mean(unchanged_vals),
            "Changed_Mean": np.mean(changed_vals),
            "Unchanged_Median": np.median(unchanged_vals),
            "Changed_Median": np.median(changed_vals),
            "MannWhitneyU_p": p_val
        }])
        stat_path = os.path.join(self.save_dir, f"{self.dataset}_{self.remain_model}_testnode_deg_stat.csv")
        stat_result.to_csv(stat_path, index=False)
        print(f"Saved statistical results to: {stat_path}")

    def run(self):
        unchanged, changed = self.get_unchanged_nodes()
        self.analyze_test_node_degree(changed, unchanged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="saved/stage2_y_edge_0.3")
    parser.add_argument("--remain_model", type=str, required=True)
    parser.add_argument("--original_model", type=str, required=True)
    parser.add_argument("--remain_run_mode", type=str, required=True)
    parser.add_argument("--original_run_mode", type=str, required=True)

    # for remain data
    parser.add_argument("--explainer_name", type=str, required=True)
    # parser.add_argument("--node_choose", type=str, default="top_pagerank")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--fraction_feat", type=float, default=0.0)
    parser.add_argument("--feature_to_node", action="store_true")
    parser.add_argument("--same_feat", action="store_true")
    parser.add_argument("--feature_type", type=str, default="categorical")
    parser.add_argument("--strategy", type=str, default="top_pagerank", choices=["top_pagerank", "random", "top_degree", "top_k_hop_subgraph"])

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


    analyzer = TestDegreeAnalyzer(
        dataset=args.dataset,
        base_dir=args.base_dir,
        remain_model=args.remain_model,
        original_model=args.original_model,
        remain_run_mode=args.remain_run_mode,
        original_run_mode=args.original_run_mode,
        orig_data=orig_data,
        remain_data=remain_data

    )
    analyzer.run()


# python analysis/test_degree_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --remain_model 1_GCN2Classifier --original_run_mode original --original_model 1_GCN2Classifier  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0 --strategy random
