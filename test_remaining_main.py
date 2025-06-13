# 新增一個test_remaining_main 直接匯入解釋子圖的結果，然後直接跑 test
import torch
import os
import numpy as np
import argparse
from utils.device import DEVICE

from data.dataset_loader import GraphDatasetLoader
from subgraph_selector.random_selector import RandomEdgeSelector, RandomFeatureSelector
from subgraph_selector.explainer_selector import ExplainerEdgeSelector
from subgraph_selector.random_walk_selector import RandomWalkEdgeSelector
from subgraph_selector.remaining_graph import RemainingGraphConstructor
from subgraph_selector.cf_remaining_graph import CFSubgraphRemover
from subgraph_selector.utils.pick_node import NodePicker

from models.basic_GCN import GCN2Classifier, GCN3Classifier
from trainer.gnn_trainer import GNNClassifierTrainer
from utils.save_result import ExperimentLogger

from data.structure import StructureFeatureBuilder, extract_edges
from data.feature2node import FeatureNodeConverter
from data.node2feature import FeatureNodeReverter
from data.prepare_split import load_split_csv
from data.split_unknown_to_test import load_split_test
from utils.node_coverage_summary import save_coverage_log
from utils.feature_utils import remove_all_zero_features, remove_top_common_features
from subgraph_selector.subgraph import CoreSubgraphExtractor


# 核心子圖包含整個節點
# 移除部分邊後的節點分類結果

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN after removing a selected subgraph.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the dataset.")

    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    
    parser.add_argument("--selector_type", type=str, default="random", choices=["random", "explainer", "random_walk"], help="Subgraph selector type")
    parser.add_argument("--fraction", type=float, default=None, help="Fraction of edges to remove for the subgraph (None means not used)")

    # random
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # explainer
    parser.add_argument("--base_dir", type=str, default="saved/stage2_node_ratio_0.01", help="Base directory for explainer results")
    parser.add_argument("--explainer_name", type=str, default="GNNExplainer", choices=["GNNExplainer", "PGExplainer", "DummyExplainer", "CFExplainer"], help="Name of the explainer model")
    parser.add_argument("--node_choose", type=str, default="random", choices=["random", "high_degree", "top_pagerank", "manual", "high_betweenness", "stratified_by_degree", "all"],  help="Name of the experiment folder") # both for explainer and random walk

    parser.add_argument("--run_mode", type=str, default="try", help="Run mode") 
    parser.add_argument("--filename", type=str, default="result", help="File name for saving results")
    parser.add_argument("--note", type=str, default="", help="Note for the experiment")

    # only structure
    parser.add_argument("--only_structure", action="store_true", help="Use only structural information (all features set to 1)")
    # feature to node
    parser.add_argument("--feature_to_node", action="store_true", help="Convert features into nodes and edges.")
    # Only use feature-node (no node-node edges)
    parser.add_argument("--only_feature_node", action="store_true", help="Use only feature-node edges, no node-node edges.")
    # Structure Mode
    parser.add_argument("--structure_mode", type=str, default="random+imp", choices=["one+imp", "random+imp"], help="Mode for structure features: 'one' or 'random+imp'")

    # select feature
    parser.add_argument("--fraction_feat", type=float, default=0, help="Fraction of features to select for feature-to-node conversion")
    parser.add_argument("--same_feat", type=lambda x: x.lower() == "true", default=True, help="If true, all nodes select the same features; otherwise, each node selects independently.")
    
    # repeat settings
    parser.add_argument("--repeat_start", type=int, default=0, help="Start repeat id (inclusive)")
    parser.add_argument("--repeat_end", type=int, default=9, help="End repeat id (inclusive)")

    # 使用的 split_id
    parser.add_argument("--split_id", type=int, default=0, help="Split ID to use for fixed train/valid masks (default: 0)")

    return parser.parse_args()


# 比較對象是 Original graph, (一般的含有節點特徵的圖)
if __name__ == "__main__":
    args = parse_args()
    selected_feat = None
    selected_feat_mask = None
    zero_feature_cols = None

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, num_classes, feature_type, num_ori_edges = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)
    ori_data = data.clone() # 備份原始資料


    print("\nLoading fixed train/valid masks...")
    # 固定取 repeat_id = 0 的 train/valid mask
    train_mask, val_mask, _, unknown_mask = load_split_csv(args.dataset, args.split_id, DEVICE)
    num_orig_nodes = train_mask.shape[0]
    num_total_nodes = data.x.shape[0]

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.unknown_mask = unknown_mask

    # Load core subgraph mask
    print("\nLoading core subgraph mask...")
    save_dir = os.path.join("saved", "core_subgraph_mask", f"split_{args.split_id}", args.run_mode, args.dataset)
    extractor = CoreSubgraphExtractor(
        ori_data=ori_data,
        remaining_graph=None,  # not used for loading
        run_mode=args.run_mode,
        dataset=args.dataset,
        repeat_id=args.split_id,
        is_undirected=True,
        trial_number=args.trial_number
    )
    feature_removed_mask, edge_removed_mask = extractor.load()

    # Build remaining graph


    # 移除特徵全為0的欄位 (只在移除相同特徵時使用)
    if args.same_feat:
        remaining_graph.x, zero_feature_cols = remove_all_zero_features(remaining_graph.x)
        print(f"Original features: {ori_data.x.shape[1]}, Removed features (all-zero): {len(zero_feature_cols)}")
    
    num_features = remaining_graph.x.size(1)


    # Train GNN on the remaining graph
    print("\nTraining GNN on the remaining graph after removing subgraph...")

    print(f"Training Classification Model - Trial {trial_number}")
    trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=remaining_graph, 
                                num_features=num_features, num_classes=num_classes, 
                                model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                trial_number=trial_number, device=DEVICE,
                                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                run_mode=save_dir)


    # 因為這時的 run_mode 後面會+split_0，所以要去掉
    model_dir = os.path.join("saved", save_dir.replace(f"_split{args.split_id}", ""), "model", args.dataset)
    model_name = f"{args.split_id}_{trainer.model_name}.pth"  # 你的 save_model_path 是這個 pattern
    path_to_model = os.path.join(model_dir, model_name)
    print(f"Loading fixed model from {path_to_model}")
    trainer.load_model(path_to_model)



    # 每次repeat 挑選的節點都不一樣，分別找子圖與訓練模型
    for repeat_id in range(args.repeat_start, args.repeat_end + 1):
        print(f"\n===== [Test Repeat {repeat_id}] =====")
        test_mask = load_split_test(args.dataset, args.split_id, repeat_id, DEVICE)
        data.test_mask = test_mask


        save_dir = os.path.join(args.run_mode, f"split_{repeat_id}")
        logger = ExperimentLogger(file_name=args.filename, note=args.note, copy_old=True, run_mode=save_dir)
        trial_number = logger.get_next_trial_number(args.dataset + "_remaining_graph")

        result = trainer.test()

        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source, repeat_id=repeat_id, model_name=model_name)

        print("Experiment finished and results saved.")
