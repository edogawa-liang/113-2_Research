import argparse
import torch
import os
from data.dataset_loader import GraphDatasetLoader
from data.data_modifier import GraphModifier
from subgraph_selector.utils.choose_node import ChooseNodeSelector
from models.explainer import SubgraphExplainer
from models.basic_GCN import GCN2Regressor, GCN3Regressor, GCN2Classifier, GCN3Classifier
from subgraph_selector.utils.feat_extract import FeatureExtractorXLSX  

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run GNN Explainer for node regression.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--normalize", type=lambda x: x.lower() == "true", default=False, help="Whether to normalize the dataset")

    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--choose_nodes", type=str, default='random', choices=["random", "high_degree", "top_pagerank", "manual", "high_betweenness"], help="Node selection strategy")
    parser.add_argument("--manual_nodes", type=str, default=None, help="Comma-separated list of node indices to explain")
    
    parser.add_argument("--node_ratio", type=str, default="auto", help="'auto' for automatic calculation or a numeric value to manually set node selection ratio")
    parser.add_argument("--edge_ratio", type=float, default=0.5, help="Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'")

    parser.add_argument("--explainer_type", type=str, default="GNNExplainer", choices=["GNNExplainer", "PGExplainer", "DummyExplainer", "CFExplainer"], help="Type of explainer to use")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs for explainer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for explainer")
    parser.add_argument("--run_mode", type=str, default="stage2_edge_0.5", help="Run mode")
    parser.add_argument("--stage1_path", type=str, default="saved/stage1", help="Directory for stage1 results")
    
    # 使用 data 的原始 y 生成解釋
    parser.add_argument("--use_raw_data", action="store_true", help="If set, use original data without removing any feature")

    # cf_explainer
    parser.add_argument("--cf_beta", type=float, default=0.5, help="Tradeoff for dist loss")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, _ = loader.load_dataset(args.dataset)
    data = data.to(device)

    if args.use_raw_data:
        print("Using raw data without removing any feature.")
        feature_trials = [1]
        feature_indices = [None]  # 表示不移除任何 feature
        model_names = [f"{args.model}Classifier"]  # 若為regression 這裡要改
        modified_graphs = [data]
    else:
        # Read feature selection results 
        ## 因為 Explainer 也需要 input data，需給他移除一個 feature 後的 data
        feature_extractor = FeatureExtractorXLSX(os.path.join(args.stage1_path, "result"))
        feature_trials, feature_indices, model_names = feature_extractor.extract_feature_numbers(args.dataset)
        print(f"Feature trials: {feature_trials}")
        print(f"Feature indices: {feature_indices}")

        # Modify graph by removing the selected feature
        modifier = GraphModifier(data)
        modified_graphs = modifier.modify_graph(feature_indices) 
        print(modified_graphs[0])

    # Model mapping
    model_mapping = {"GCN2Regressor": GCN2Regressor, "GCN3Regressor": GCN3Regressor, "GCN2Classifier": GCN2Classifier, "GCN3Classifier": GCN3Classifier}

    # Select nodes to explain
    node_selector = ChooseNodeSelector(data, node_ratio=args.node_ratio, edge_ratio=args.edge_ratio, strategy=args.choose_nodes, manual_nodes=args.manual_nodes)
    node_indices = node_selector.select_nodes()
    print(f"Selected {len(node_indices)} nodes using strategy: {args.choose_nodes}")

    # feature_trials,  feature_indices, modified_graphs 數量一樣多
    for i in range(len(feature_trials)):
        
        # model class
        model_class = model_mapping[model_names[i]]
        # model path
        model_path = os.path.join(args.stage1_path, "model", args.dataset, f"{feature_trials[i]}_{model_class.__name__}.pth")
        print(f"Loading model from {model_path}")
        explainer = SubgraphExplainer(
            model_class=model_class,
            dataset=args.dataset,
            data=modified_graphs[i],
            model_path=model_path,
            explainer_type=args.explainer_type,
            hop= 2 if args.model == "GCN2" else 3,
            epoch=args.epoch,
            lr=args.lr,
            run_mode=args.run_mode,
            trial_name=feature_trials[i],
            remove_feature=feature_indices[i] if feature_indices[i] is not None else -1,
            device=device,
            choose_nodes=args.choose_nodes,
            cf_beta=args.cf_beta,
        )

        # Explain each node
        for node_idx in node_indices:
            print(f"\nExplaining node {node_idx} with feature {feature_indices[i]} removed...")
            explainer.explain_node(node_idx, modified_graphs[i], save=True)

    print("Node explanations completed and saved.")
