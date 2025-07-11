import torch
import os
import argparse
from utils.device import DEVICE

from data.dataset_loader import GraphDatasetLoader
from subgraph_selector.random_selector import RandomEdgeSelector, RandomFeatureSelector
from subgraph_selector.explainer_selector import ExplainerEdgeSelector
from subgraph_selector.random_walk_selector import RandomWalkEdgeSelector
from subgraph_selector.remaining_graph import RemainingGraphConstructor
from subgraph_selector.cf_remaining_graph import CFSubgraphRemover

from models.basic_GCN import GCN2Classifier, GCN3Classifier
from trainer.gnn_trainer import GNNClassifierTrainer
from utils.save_result import ExperimentLogger

from data.feature2node import FeatureNodeConverter
from data.node2feature import FeatureNodeReverter

from utils.feature_utils import remove_all_zero_features, remove_top_common_features



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

    # only structure
    parser.add_argument("--only_structure", action="store_true", help="Use only structural information (all features set to 1)")

    # feature to node
    parser.add_argument("--feature_to_node", action="store_true", help="Convert features into nodes and edges.")

    # select feature
    parser.add_argument("--fraction_feat", type=float, default=0, help="Fraction of features to select for feature-to-node conversion")
    parser.add_argument("--same_feat", type=lambda x: x.lower() == "true", default=True, help="If true, all nodes select the same features; otherwise, each node selects independently.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    selected_feat = None

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, num_classes, feature_type, num_ori_edges = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)
    ori_data = data.clone() # 備份原始資料
    selected_feat_mask = None
    zero_feature_cols = None

    # 如果只用結構，則把所有節點特徵設為 1
    if args.only_structure:
        print("Using only structure: all node features set to 1.")
        data.x = torch.ones((data.num_nodes, 1), device=DEVICE)
        num_features = 1  # 更新特徵維度，否則模型初始化會錯

    # 如果要把特徵轉換成節點，則使用 FeatureNodeConverter 
    # random 需要, explainer 不需要, random_walk 需要
    if args.feature_to_node:
        print("Converting node features into feature-nodes...")
        converter = FeatureNodeConverter(feature_type=feature_type, device=DEVICE)
        data = converter.convert(data)
        num_features = data.x.size(1)  # 更新特徵維度（此時為 1）


    # Select subgraph
    if args.selector_type == "random": # 使用時不要加feature_to_node
        print("Using Random Selector")
        selector = RandomEdgeSelector(
            data=data,
            fraction=args.fraction,
            seed=args.seed,
            device=DEVICE,
            top_k_percent_feat=args.fraction_feat,
            feature_to_node=args.feature_to_node
        )
        selected_edges = selector.select_edges(num_ori_edges=num_ori_edges)#  # 必須傳入以正確區分 node-node 和 node-feature 邊)

        # 特徵的選擇器（如果有指定 fraction_feat）
        # 可指定要不要選擇相同的特徵
        if args.fraction_feat > 0 and not args.feature_to_node:
            print("Using Random Feature Selector")
            feat_selector = RandomFeatureSelector(
                num_nodes=data.num_nodes,
                num_features=num_features,
                top_k_percent_feat=args.fraction_feat,
                same_feat=args.same_feat,
                seed=args.seed,
                device=DEVICE
            )
            selected_feat = feat_selector.select_node_features()  # shape: [num_nodes, num_features]

    
    elif args.selector_type == "explainer": # 處理PyG支援的可解釋方法
        if args.explainer_name != "CFExplainer": # CF另外處理
            print("Using Explainer Selector")
            
            selector = ExplainerEdgeSelector(
                base_dir=args.base_dir,
                explainer_name=args.explainer_name,
                dataset_name=args.dataset,
                node_choose=args.node_choose,
                top_k_percent=args.fraction,
                feature_type=feature_type,
                device=DEVICE,
                top_k_percent_feat=args.fraction_feat,
                use_feature_to_node=args.feature_to_node # 若要使用特徵，分成直接使用(node_mask)還是將特徵轉為結構使用(edge_mask)
            )            
            selector.load_data()
            selector.plot_edge_distribution()
            num_node = selector.get_node_count()
            num_edge = selector.get_edge_count()
            # 只有需要選特徵邊時，selected_feat_ids 才會有值，不然就是空的
            selected_edges, selected_feat_ids = selector.select_edges(num_ori_edges=num_ori_edges, num_ori_nodes=ori_data.num_nodes,
                                                                      ori_num_features=ori_data.x.size(1), return_feat_ids=True)
            # 如果需要使用解釋子圖自身的特徵重要度
            if not args.feature_to_node and args.fraction_feat > 0:
                selected_feat = selector.select_node_features(num_ori_nodes=ori_data.num_nodes, same_feat=args.same_feat)


    elif args.selector_type == "random_walk":
        print("Using Random Walk Selector")
        selector = RandomWalkEdgeSelector(data, node_ratio=args.node_ratio, edge_ratio =args.edge_ratio , fraction=args.fraction, 
                                          walk_length=args.walk_length, num_walks=args.num_walks, node_choose=args.node_choose, feature_type=feature_type, 
                                          device=DEVICE, mask_type=args.mask_type, top_k_percent_feat=args.fraction_feat, ori_data=ori_data)
        node_start_ratio = selector.get_final_node_ratio()
        edge_neighbor_ratio = selector.get_neighbor_edge_ratio()
        selected_edges, selected_feat_ids, ori_edge_visit_ratio, feat_edge_visit_ratio = selector.select_edges(return_feat_ids=True)

    # Remove subgraph from the original graph
    if args.explainer_name == "CFExplainer":
        remaining_graph_constructor = CFSubgraphRemover(data, args.base_dir, args.explainer_name, args.dataset, args.node_choose, device=DEVICE)
        remaining_graph_constructor.load_data()
        remaining_graph = remaining_graph_constructor.get_remaining_graph()
        num_node = remaining_graph_constructor.get_node_count()
        num_edge = remaining_graph_constructor.get_edge_count()
    else:
        remaining_graph_constructor = RemainingGraphConstructor(data, selected_edges, selected_feat_mask=selected_feat, device=DEVICE)
        remaining_graph = remaining_graph_constructor.get_remaining_graph()


    # if use feature to node, revert the feature node to original node (add feature value into original graph)
    if args.feature_to_node:
        print("Reverting feature nodes to original nodes...")
        revertor = FeatureNodeReverter(feature_type=feature_type, device=DEVICE)
        remaining_graph = revertor.revert(remaining_graph, ori_data)
        num_features = remaining_graph.x.size(1)

        # 如果設定了要遮蔽相同的特徵，且有 selected_feat_ids，就從 ori_data 上遮蔽
        if args.same_feat and selected_feat_ids and args.fraction_feat > 0: # 如果沒有使用feat2node, selected_feat_ids會是空的
            print("改成 remove same features")
            remaining_graph.x, removed_feat_ids = remove_top_common_features(
                x=ori_data.x, # 放入的是原始的data
                selected_feat_ids=selected_feat_ids,
                fraction_feat=args.fraction_feat
            )

    # 移除特徵全為0的欄位 (只在移除相同特徵時使用)
    if args.same_feat:
        remaining_graph.x, zero_feature_cols = remove_all_zero_features(remaining_graph.x)
        print(f"Original features: {ori_data.x.shape[1]}, Removed features (all-zero): {len(zero_feature_cols)}")
    
    num_features = remaining_graph.x.size(1)


    # Train GNN on the remaining graph
    print("\nTraining GNN on the remaining graph after removing subgraph...")
    print("Final data imput to model:", remaining_graph)

    logger = ExperimentLogger(file_name=args.filename, note=args.note, copy_old=True, run_mode=args.run_mode)
    trial_number = logger.get_next_trial_number(args.dataset + "_remaining_graph")
    print(f"Training Classification Model - Trial {trial_number}")
    trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=remaining_graph, 
                                   num_features=num_features, num_classes=num_classes, 
                                   model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                   trial_number=trial_number, device=DEVICE,
                                   epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                   run_mode=args.run_mode)

    result = trainer.run()

    # Save experiment results
    # 移除的邊數量都是 fraction
    if args.selector_type == "random":
        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, fraction=args.fraction, fraction_feat=args.fraction_feat, remove_feat=zero_feature_cols)
    
    elif args.selector_type == "explainer":
        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, explaner=args.explainer_name, node_choose=args.node_choose, fraction=args.fraction, 
                              node_explain_ratio=num_node/data.x.shape[0], edge_explain_ratio=num_edge/data.edge_index.shape[1], fraction_feat=args.fraction_feat, remove_feat=zero_feature_cols)
    
    elif args.selector_type == "random_walk":
        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, walk_length=args.walk_length, num_walks=args.num_walks, node_choose=args.node_choose, 
                              fraction=args.fraction, node_start_ratio=node_start_ratio, edge_neighbor_ratio= edge_neighbor_ratio, fraction_feat=args.fraction_feat, ori_edge_visit_ratio=ori_edge_visit_ratio, feat_edge_visit_ratio=feat_edge_visit_ratio, remove_feat=zero_feature_cols)

    print("Experiment finished and results saved.")
