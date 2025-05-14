import torch
import os
import argparse
from data.dataset_loader import GraphDatasetLoader

from subgraph_selector.random_selector import RandomEdgeSelector
from subgraph_selector.explainer_selector import ExplainerEdgeSelector
from subgraph_selector.random_walk_selector import RandomWalkEdgeSelector
from subgraph_selector.remaining_graph import RemainingGraphConstructor
from subgraph_selector.cf_remaining_graph import CFSubgraphRemover

from models.basic_GCN import GCN2Classifier, GCN3Classifier
from trainer.gnn_trainer import GNNClassifierTrainer
from utils.save_result import ExperimentLogger

# 核心子圖包含整個節點
# 移除部分邊後的節點分類結果

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN after removing a selected subgraph.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--normalize", type=lambda x: x.lower() == "true", default=False, help="Whether to normalize the dataset")

    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    
    parser.add_argument("--selector_type", type=str, default="random", choices=["random", "explainer", "random_walk"], help="Subgraph selector type")
    # parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of edges to remove for the subgraph") # random, explainer, random_walk
    parser.add_argument("--fraction", type=float, default=None, help="Fraction of edges to remove for the subgraph (None means not used)")

    # random
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # explainer
    parser.add_argument("--base_dir", type=str, default="saved/stage2_node_ratio_0.01", help="Base directory for explainer results")
    parser.add_argument("--explainer_name", type=str, default="GNNExplainer", choices=["GNNExplainer", "PGExplainer", "DummyExplainer", "CFExplainer"], help="Name of the explainer model")
    parser.add_argument("--node_choose", type=str, default="random", choices=["random", "high_degree", "top_pagerank", "manual", "high_betweenness", "stratified_by_degree", "all"],  help="Name of the experiment folder") # both for explainer and random walk
    # random walk
    parser.add_argument("--walk_length", type=int, default=10, help="Number of steps per random walk")
    parser.add_argument("--num_walks", type=int, default=5, help="Number of walks per selected node")
    parser.add_argument("--node_ratio", type=str, default="auto", help="'auto' for automatic calculation or a numeric value to manually set node selection ratio")
    parser.add_argument("--edge_ratio", type=float, default=0.5, help="Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'")
    parser.add_argument("--mask_type", type=str, default="train", choices=["train", "test", "all"], help="Mask type for node selection")

    parser.add_argument("--run_mode", type=str, default="try", help="Run mode") # 如果跑的是移除test的子圖，run_mode 前加入"test"
    parser.add_argument("--filename", type=str, default="result", help="File name for saving results")
    parser.add_argument("--note", type=str, default="", help="Note for the experiment")

    parser.add_argument("--only_structure", action="store_true", help="Use only structural information (all features set to 1)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, num_classes = loader.load_dataset(args.dataset)
    data = data.to(device)
    if args.only_structure:
        print("Using only structure: all node features set to 1.")
        data.x = torch.ones((data.num_nodes, 1), device=device)
        num_features = 1  # 更新特徵維度，否則模型初始化會錯


    # Select subgraph
    if args.selector_type == "random":
        print("Using Random Selector")
        selector = RandomEdgeSelector(data, fraction=args.fraction, seed=args.seed, device=device)
        selected_edges = selector.select_edges()
    
    elif args.selector_type == "explainer": # 處理PyG支援的可解釋方法
        if args.explainer_name != "CFExplainer": # CF另外處理
            print("Using Explainer Selector")
            selector = ExplainerEdgeSelector(args.base_dir, args.explainer_name, args.dataset, args.node_choose, args.fraction, device=device)
            selector.load_data()
            selector.plot_edge_distribution()
            num_node = selector.get_node_count()
            num_edge = selector.get_edge_count()
            selected_edges = selector.select_edges()

    elif args.selector_type == "random_walk":
        print("Using Random Walk Selector")
        selector = RandomWalkEdgeSelector(data, node_ratio=args.node_ratio, edge_ratio =args.edge_ratio , fraction=args.fraction, 
                                          walk_length=args.walk_length, num_walks=args.num_walks, node_choose=args.node_choose, device=device, mask_type=args.mask_type)
        node_start_ratio = selector.get_final_node_ratio()
        edge_neighbor_ratio = selector.get_neighbor_edge_ratio()
        selected_edges = selector.select_edges()


    # Remove subgraph from the original graph
    if args.explainer_name == "CFExplainer":
        remaining_graph_constructor = CFSubgraphRemover(data, args.base_dir, args.explainer_name, args.dataset, args.node_choose, device=device)
        remaining_graph_constructor.load_data()
        remaining_graph = remaining_graph_constructor.get_remaining_graph()
        num_node = remaining_graph_constructor.get_node_count()
        num_edge = remaining_graph_constructor.get_edge_count()
    else:
        remaining_graph_constructor = RemainingGraphConstructor(data, selected_edges, device=device)
        remaining_graph = remaining_graph_constructor.get_remaining_graph()

    # Train GNN on the remaining graph
    print("\nTraining GNN on the remaining graph after removing subgraph...")

    logger = ExperimentLogger(file_name=args.filename, note=args.note, copy_old=True, run_mode=args.run_mode)
    trial_number = logger.get_next_trial_number(args.dataset + "_remaining_graph")
    print(f"Training Classification Model - Trial {trial_number}")
    trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=remaining_graph, 
                                   num_features=num_features, num_classes=num_classes, 
                                   model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                   trial_number=trial_number, device=device,
                                   epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                   run_mode=args.run_mode)

    result = trainer.run()

    # Save experiment results
    # 移除的邊數量都是 fraction
    if args.selector_type == "random":
        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, fraction=args.fraction)
    
    elif args.selector_type == "explainer":
        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, explaner=args.explainer_name, node_choose=args.node_choose, fraction=args.fraction, node_explain_ratio=num_node/data.x.shape[0], edge_explain_ratio=num_edge/data.edge_index.shape[1]), 
    
    elif args.selector_type == "random_walk":
        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, walk_length=args.walk_length, num_walks=args.num_walks, node_choose=args.node_choose, fraction=args.fraction, node_start_ratio=node_start_ratio, edge_neighbor_ratio= edge_neighbor_ratio)

    print("Experiment finished and results saved.")
