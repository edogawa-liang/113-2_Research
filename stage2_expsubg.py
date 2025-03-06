import argparse
import os
import torch
import random
from data.dataset_loader import GraphDatasetLoader
from data.data_modifier import GraphModifier
from models.explainer import SubgraphExplainer
from models.basic_GCN import GCN2Regressor, GCN3Regressor  
from subgraph_selector.utils.feat_extract import FeatureExtractorXLSX  

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run GNN Explainer for node regression.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--nodes", type=str, default='random', help="Comma-separated list of node indices to explain")
    parser.add_argument("--node_ratio", type=float, default=0.01, help="Ratio of training nodes to explain")
    parser.add_argument("--explainer_type", type=str, default="GNNExplainer", choices=["GNNExplainer", "PGExplainer"], help="Type of explainer to use")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs for explainer")
    parser.add_argument("--run_mode", type=str, default="stage2_expsubg", help="Run mode")
    parser.add_argument("--stage1_path", type=str, default="saved/stage1", help="Directory for stage1 results")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    loader = GraphDatasetLoader()
    data, num_features, num_classes = loader.load_dataset(args.dataset)
    data = data.to(device)

    # Set model class
    model_class = GCN2Regressor if args.model == "GCN2" else GCN3Regressor

    # Read feature selection results 
    ## 因為 Explainer 也需要 input data，需給他移除一個 feature 後的 data
    feature_extractor = FeatureExtractorXLSX(os.path.join(args.stage1_path, "result"))
    feature_trials, feature_indices = feature_extractor.extract_feature_numbers(args.dataset)
    print(f"Feature trials: {feature_trials}")
    print(f"Feature indices: {feature_indices}")

    # Modify graph by removing the selected feature
    modifier = GraphModifier(data)
    modified_graphs = modifier.modify_graph(feature_indices) 
    
    # ======= 選擇一次 nodes，確保不同 feature_trials 下都使用相同的 nodes =======
    if args.nodes == "random":
        train_nodes = data.train_mask.nonzero().view(-1).tolist()  # 只會從 training nodes 中選
        num_selected = max(1, int(len(train_nodes) * args.node_ratio))
        node_indices = random.sample(train_nodes, num_selected)
        print(f"Randomly selected {num_selected} training nodes for explanation: {node_indices}")
    else:  # 指定 node
        node_indices = [int(n) for n in args.nodes.split(",")]
            

    # feature_trials,  feature_indices, modified_graphs 數量一樣多
    for i in range(len(feature_trials)):
        # model path
        model_path = os.path.join(args.stage1_path, "model", args.dataset, f"{feature_trials[i]}_{model_class.__name__}.pth")

        explainer = SubgraphExplainer(
            model_class=model_class,
            dataset=args.dataset,
            data=modified_graphs[i],
            model_path=model_path,
            explainer_type=args.explainer_type,
            hop= 2 if args.model == "GCN2" else 3,
            epoch=args.epoch,
            run_mode=args.run_mode,
            trial_name=feature_trials[i],
            remove_feature=feature_indices[i],
            device=device
        )

        # Explain each node
        for node_idx in node_indices:
            print(f"\nExplaining node {node_idx} with feature {feature_indices[i]} removed...")
            explainer.explain_node(node_idx, modified_graphs[i], save=True)

    print("Node explanations completed and saved.")
