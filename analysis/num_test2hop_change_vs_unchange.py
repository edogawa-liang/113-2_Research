# python analysis/num_test2hop_change_vs_unchange.py --dataset Cora --remain_run_mode remove_from_GNNExplainer --original_run_mode original --remain_model 1_GCN2Classifier --original_model 1_GCN2Classifier  --strategy random  --explainer_name GNNExplainer --fraction 0.2 --fraction_feat 0.0

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from data.dataset_loader import GraphDatasetLoader
from models.basic_GCN import GCN2Classifier
from utils.device import DEVICE
from scipy.stats import mannwhitneyu
from subgraph_selector.remaining_graph import RemainingGraphConstructor
from subgraph_selector.explainer_selector import ExplainerEdgeSelector



class TwoHopNodeAnalyzer:
    def __init__(self, dataset, base_dir, remain_model, original_model, remain_run_mode, 
                 original_run_mode, strategy, orig_data, remain_data):
        self.dataset = dataset
        self.remain_model = remain_model
        self.original_model_name = original_model
        self.remain_run_mode = remain_run_mode
        self.original_run_mode = original_run_mode
        self.strategy = strategy
        self.orig_data = orig_data
        self.remain_data = remain_data

        self.csv_dir = os.path.join(base_dir, "GNNExplainer", dataset, original_model)
        self.record_path = os.path.join(self.csv_dir, "node_record.csv")

        self.model = self.load_model(self.remain_run_mode, self.remain_model)
        self.original_model = self.load_model(self.original_run_mode, self.original_model_name)

        self.save_dir = os.path.join("saved", "analysis", "test2hop_num")
        os.makedirs(self.save_dir, exist_ok=True)



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
    

    def get_selected_nodes(self):
        df = pd.read_csv(self.record_path)
        if self.strategy not in df.columns:
            raise ValueError(f"CSV must contain column '{self.strategy}'")

        selected_nodes = df["Node"][df[self.strategy] == 1].tolist()
        return set(selected_nodes)

    def count_selected_in_2hop(self, selected_nodes):
        counts = {}
        test_nodes = self.orig_data.test_mask.nonzero(as_tuple=True)[0]
        for node in test_nodes:
            sub_nodes, _, _, _ = k_hop_subgraph(node.item(), 2, self.orig_data.edge_index, relabel_nodes=False)
            count = len(set(sub_nodes.tolist()) & selected_nodes)
            counts[node.item()] = count
        return counts


    def plot_bar_and_box(self, count_dict, changed, unchanged):
        bar_data = {}
        for node, count in count_dict.items():
            label = "Changed" if node in changed else "Unchanged"
            if count not in bar_data:
                bar_data[count] = {"Changed": 0, "Unchanged": 0}
            bar_data[count][label] += 1

        sorted_keys = sorted(bar_data.keys())
        changed_vals = [bar_data[k]["Changed"] for k in sorted_keys]
        unchanged_vals = [bar_data[k]["Unchanged"] for k in sorted_keys]

        x = np.arange(len(sorted_keys))
        width = 0.4

        # Bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, changed_vals, width, label="Changed", color="#ff7f0e")
        plt.bar(x + width/2, unchanged_vals, width, label="Unchanged", color="#1f77b4")
        plt.xticks(x, sorted_keys)
        plt.xlabel("Number of Selected Nodes in 2-hop")
        plt.ylabel("Number of Test Nodes")
        plt.title("2-hop Coverage of Selected Nodes")
        plt.legend()
        plt.tight_layout()

        bar_path = os.path.join(self.save_dir, f"{self.dataset}_{self.remain_model}_bar_by_2hop_covered.png")
        plt.savefig(bar_path)
        plt.close()
        print(f"Saved bar chart to: {bar_path}")

        # Boxplot + Mann-Whitney U Test
        counts_changed = [v for k, v in count_dict.items() if k in changed]
        counts_unchanged = [v for k, v in count_dict.items() if k in unchanged]

        # Mann-Whitney U test
        u_stat, p_val = mannwhitneyu(counts_unchanged, counts_changed, alternative="two-sided")

        plt.figure(figsize=(8, 6))
        box = plt.boxplot([counts_unchanged, counts_changed], patch_artist=True,
                        labels=["Unchanged", "Changed"],
                        boxprops=dict(facecolor="#1f77b4"),
                        medianprops=dict(color='black'))
        box['boxes'][1].set_facecolor('#ff7f0e')
        plt.ylabel("Number of Selected Nodes in 2-hop")
        plt.title("Boxplot of Selected Nodes in 2-hop")

        # 標註 p-value
        y_max = max(max(counts_unchanged), max(counts_changed)) * 1.1
        label = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else f"p={p_val:.2f}"
        plt.text(1.5, y_max, label, ha="center", va="bottom", fontsize=10, color="red")

        # Legend
        handles = [
            plt.Line2D([0], [0], color="#1f77b4", lw=10, label="Unchanged"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=10, label="Changed")
        ]
        plt.legend(handles=handles, loc='upper right')
        plt.tight_layout()

        box_path = os.path.join(self.save_dir, f"{self.dataset}_{self.remain_model}_box_by_2hop_covered.png")
        plt.savefig(box_path)
        plt.close()
        print(f"Saved boxplot to: {box_path}")

        # 儲存統計資料
        stat_result = pd.DataFrame([{
            "Metric": "NumSelectedIn2Hop",
            "Unchanged_Mean": np.mean(counts_unchanged),
            "Changed_Mean": np.mean(counts_changed),
            "Unchanged_Median": np.median(counts_unchanged),
            "Changed_Median": np.median(counts_changed),
            "MannWhitneyU_p": p_val
        }])
        stat_path = os.path.join(self.save_dir, f"{self.dataset}_{self.remain_model}_{self.strategy}_stat.csv")
        stat_result.to_csv(stat_path, index=False)
        print(f"Saved statistical results to: {stat_path}")


    def run(self):
        unchanged, changed = self.get_unchanged_nodes()
        selected_nodes = self.get_selected_nodes()
        count_dict = self.count_selected_in_2hop(selected_nodes)
        self.plot_bar_and_box(count_dict, changed, unchanged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_dir", type=str, default="saved/stage2_y_edge_0.3", help="Base path to saved results")
    parser.add_argument("--remain_model", type=str, required=True)
    parser.add_argument("--original_model", type=str, required=True)
    parser.add_argument("--remain_run_mode", type=str, required=True)
    parser.add_argument("--original_run_mode", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)

    # for remain data
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


    analyzer = TwoHopNodeAnalyzer(
        dataset=args.dataset,
        base_dir=args.base_dir,
        remain_model=args.remain_model,
        original_model=args.original_model,
        remain_run_mode=args.remain_run_mode,
        original_run_mode=args.original_run_mode,
        strategy=args.strategy,
        orig_data=orig_data,
        remain_data=remain_data
    )
    analyzer.run()
