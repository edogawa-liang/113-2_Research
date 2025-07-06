import argparse
import torch
import os
import pickle
from utils.device import DEVICE
from data.dataset_loader import GraphDatasetLoader
from data.data_modifier import GraphModifier
from subgraph_selector.utils.choose_node import ChooseNodeSelector
from models.explainer import SubgraphExplainer
from models.basic_GCN import GCN2Regressor, GCN3Regressor, GCN2Classifier, GCN3Classifier
from subgraph_selector.utils.feat_extract import FeatureExtractorXLSX  
from data.feature2node import FeatureNodeConverter


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run GNN Explainer for node regression.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--normalize", type=lambda x: x.lower() == "true", default=False, help="Whether to normalize the dataset")

    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--choose_nodes", type=str, default='random', choices=["random", "high_degree", "top_pagerank", "manual", "high_betweenness", "stratified_by_degree", "all"], help="Node selection strategy")
    parser.add_argument("--manual_nodes", type=str, default=None, help="Comma-separated list of node indices to explain")
    parser.add_argument("--mask_type", type=str, default="train", choices=["train", "test", "all"], help="Mask type for node selection")
    
    parser.add_argument("--node_ratio", type=str, default="auto", help="'auto' for automatic calculation or a numeric value to manually set node selection ratio")
    parser.add_argument("--edge_ratio", type=float, default=0.5, help="Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'")

    parser.add_argument("--explainer_type", type=str, default="GNNExplainer", choices=["GNNExplainer", "PGExplainer", "DummyExplainer", "CFExplainer"], help="Type of explainer to use")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs for explainer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for explainer")
    parser.add_argument("--run_mode", type=str, default="stage2_edge_0.5", help="Run mode") # 如果生成test的解釋子圖, run_mode前改成 "test_stage2_edge_0.5"
    parser.add_argument("--stage1_path", type=str, default="saved/stage1", help="Directory for stage1 results")
    
    # 使用 data 的原始 y 生成解釋
    parser.add_argument("--use_raw_data", action="store_true", help="If set, use original data without removing any feature")

    # cf_explainer
    parser.add_argument("--cf_beta", type=float, default=0.5, help="Tradeoff for dist loss")

    # only structure
    parser.add_argument("--only_structure", action="store_true", help="Use only structural information (all features set to 1)")

    # feature to node
    parser.add_argument("--feature_to_node", action="store_true", help="Convert features into nodes and edges.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Using DEVICE: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, _, feature_type, _ = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)
    ori_data = data.clone()  # 儲存原始資料，給feature2node時choose_nodes用

    # 若使用 only_structure 模式，將所有特徵設為 1 維常數
    if args.only_structure:
        print("Using only structure: all node features set to 1.")
        data.x = torch.ones((data.num_nodes, 1), device=DEVICE)
        num_features = 1  # 更新特徵數量，避免 downstream 模型出錯

    # 如果要把特徵轉換成節點，則使用 FeatureNodeConverter
    if args.feature_to_node:
        print("Converting node features into feature-nodes...")
        converter = FeatureNodeConverter(feature_type=feature_type, device=DEVICE)
        data = converter.convert(data)
        num_features = data.x.size(1)  # 更新特徵維度（此時為 1）


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
    # 如果有使用到特徵節點，注意在 Choose Node 時要用原始的 data (ori_data)
    node_selector = ChooseNodeSelector(ori_data, node_ratio=args.node_ratio, edge_ratio=args.edge_ratio, strategy=args.choose_nodes, manual_nodes=args.manual_nodes, mask_type=args.mask_type)
    node_indices = node_selector.select_nodes() # 因為特徵節點接在普通節點後，可以直接把ori data 的 node_indices 當成新 data 要解釋的indices
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
            choose_nodes=args.choose_nodes,
            cf_beta=args.cf_beta,
        )


    # Explain each node
    cf_success_nodes = []
    cf_fail_nodes = []

    for node_idx in node_indices:
        print(f"\nExplaining node {node_idx} with feature {feature_indices[i]} removed...")
        result = explainer.explain_node(node_idx, modified_graphs[i], save=True)

        # 只有CF要計算
        if args.explainer_type == "CFExplainer":
            if result is not None:
                cf_success_nodes.append(node_idx)
            else:
                cf_fail_nodes.append(node_idx)

    if args.explainer_type == "CFExplainer":
        # Print summary
        print("\n========== CF Explanation Summary ==========")
        print(f"Total nodes selected: {len(node_indices)}")
        print(f"Nodes with counterfactual explanation: {len(cf_success_nodes)}")
        print(f"Nodes without counterfactual explanation: {len(cf_fail_nodes)}")
        if cf_fail_nodes:
            print(f"Nodes without explanation: {cf_fail_nodes}")
        print("============================================\n")


        # Save to pickle
        cf_summary = {
            "success_nodes": cf_success_nodes,
            "fail_nodes": cf_fail_nodes
        }
        save_dir = os.path.join("saved", args.run_mode, args.explainer_type, args.dataset, args.choose_nodes, f"{feature_trials[i]}_{explainer.model_class.__name__}")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, "cf_summary.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(cf_summary, f)

        print(f"Saved CF explanation summary to {save_path}")

        print("Node explanations completed and saved.")
