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
    parser.add_argument("--fraction", type=float, default=0, help="Fraction of edges to remove for the subgraph (None means not used)")

    # random
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # explainer
    parser.add_argument("--base_dir", type=str, default="saved/stage2", help="Base directory for explainer results")
    parser.add_argument("--explainer_name", type=str, default="GNNExplainer", choices=["GNNExplainer", "PGExplainer", "DummyExplainer", "CFExplainer"], help="Name of the explainer model")
    parser.add_argument("--node_choose", type=str, default="stratified_by_degree", choices=["all_train", "random", "stratified_by_degree", "degree", "pagerank", "betweenness", "closeness"],  help="Name of the experiment folder") # both for explainer and random walk
    # random walk
    parser.add_argument("--walk_length", type=int, default=10, help="Number of steps per random walk")
    parser.add_argument("--num_walks", type=int, default=3, help="Number of walks per selected node")
    parser.add_argument("--node_ratio", type=str, default="auto", help="'auto' for automatic calculation or a numeric value to manually set node selection ratio")
    parser.add_argument("--edge_ratio", type=float, default=0.5, help="Ensures sufficient edges in the subgraph, required only if node_ratio is 'auto'")

    parser.add_argument("--run_mode", type=str, default="try", help="Run mode") # 如果跑的是移除test的子圖，run_mode 前加入"test"
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
    parser.add_argument("--same_feat", action="store_true", help="all nodes select the same features; otherwise, each node selects independently.")

    # split settings
    parser.add_argument("--split_start", type=int, default=0, help="Start repeat id (inclusive)")
    parser.add_argument("--split_end", type=int, default=0, help="End repeat id (inclusive)")

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

    # define pad_mask function
    def pad_mask(mask, pad_len):
        return torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool, device=mask.device)], dim=0)

    # 1. Feature to node conversion
    if args.feature_to_node:
        print("Converting node features into feature-nodes...")
        converter = FeatureNodeConverter(feature_type=feature_type, device=DEVICE)
        # FeatureNodeConverter 提供feature連到node後的所有邊，並多了 node_node_mask, node_feat_mask (後續要記得處理!)
        data = converter.convert(data)

    # 2. Structure feature building (node features)
    if args.feature_to_node or args.only_structure:
        print("Using StructureFeatureBuilder...")
        # 將 Feature 改成新的算法 (random (32 dim)+ [PageRank, Betweenness, Degree, Closeness])
        builder = StructureFeatureBuilder(
            data=data, device=DEVICE, dataset_name=args.dataset,
            feature_to_node=args.feature_to_node,
            only_feature_node=args.only_feature_node,
            only_structure=args.only_structure,
            mode=args.structure_mode,
            emb_dim=32,
            normalize_type="row_l1",
            learn_embedding=True,  # 不學習 init random embedding
        )
        structure_x = builder.build()
        num_features = structure_x.shape[1]
        data.x = structure_x

    # 3. Use original node features
    else:
        print("Using original graph and node features.")
        num_features = data.x.shape[1]

    # 統一更新 edge_index, edge_weight (不論原 graph 或 feature to node 都可以用 extract_edges)
    edge_index, edge_weight = extract_edges(data, args.feature_to_node, args.only_feature_node)
    data.edge_index = edge_index
    data.edge_weight = edge_weight


    # 每次 repeat 挑選的節點都不一樣，分別找子圖與訓練模型
    for split_id in range(args.split_start, args.split_end + 1):
        print(f"\n===== [Repeat {split_id}] =====")

        train_mask, val_mask, test_mask, unknown_mask = load_split_csv(args.dataset, split_id, DEVICE) # 這裏的mask是原dataset的長度
        num_orig_nodes = train_mask.shape[0]
        num_total_nodes = data.x.shape[0]

        # add padding to feature node (mask會補滿，但y不會)
        if args.feature_to_node and num_total_nodes > num_orig_nodes:
            pad_len = num_total_nodes - num_orig_nodes
            print(f"Padding masks with {pad_len} additional nodes for feature nodes...")

            train_mask = pad_mask(train_mask, pad_len)
            val_mask = pad_mask(val_mask, pad_len)
            test_mask = pad_mask(test_mask, pad_len)
            unknown_mask = pad_mask(unknown_mask, pad_len) 

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.unknown_mask = unknown_mask


        # pick node: 挑選所有的訓練節點作為起點 (all_train) or 部分的訓練節點 (by random, Degree, PageRank, Betweenness, Closeness)
        if args.selector_type == "explainer" or args.selector_type == "random_walk":
            picker = NodePicker(
                data=data, dataset_name=args.dataset, node_choose=args.node_choose,
                feature_to_node=args.feature_to_node, only_feature_node=args.only_feature_node, # 要從哪個csv import imp
                node_ratio=args.node_ratio, edge_ratio=args.edge_ratio # 控制挑選的節點數量
            )
            selected_nodes = picker.pick_nodes()
            coverage_stats = picker.compute_coverage(selected_nodes)  # 獲取 coverage 統計


            save_coverage_log(args, split_id, coverage_stats, selected_nodes=selected_nodes, save_dir="saved/node_coverage")


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
            selected_edges = selector.select_edges(num_ori_edges)#  # 必須傳入以正確區分 node-node 和 node-feature 邊)

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
                    data=data,
                    base_dir=args.base_dir,
                    explainer_name=args.explainer_name,
                    dataset_name=args.dataset,
                    selected_nodes=selected_nodes,
                    top_k_percent=args.fraction,
                    feature_type=feature_type,
                    device=DEVICE,
                    top_k_percent_feat=args.fraction_feat,
                    use_feature_to_node=args.feature_to_node # 若要使用特徵，分成直接使用(node_mask)還是將特徵轉為結構使用(edge_mask)
                )            
                selector.load_data(split_id)
                selector.plot_edge_distribution()

                # 只有需要選特徵邊時，selected_feat_ids 才會有值，不然就是空的
                selected_edges, selected_feat_ids = selector.select_edges() # 選到的所有邊 (包含特徵邊), 即這些特徵是哪幾個
                
                # 如果需要使用解釋子圖自身的特徵重要度
                if not args.feature_to_node and args.fraction_feat > 0:
                    # 改 select_node_features
                    selected_feat = selector.select_node_features(same_feat=args.same_feat)


        elif args.selector_type == "random_walk":
            print("Using Random Walk Selector")
            # 改成直接傳入node
            selector = RandomWalkEdgeSelector(data, fraction=args.fraction, selected_nodes=selected_nodes,
                                            walk_length=args.walk_length, num_walks=args.num_walks, feature_type=feature_type, 
                                            device=DEVICE, top_k_percent_feat=args.fraction_feat)
            selected_edges, selected_feat_ids, ori_edge_visit_ratio, feat_edge_visit_ratio = selector.select_edges()

        # check
        print("Number of Selected edges:", len(selected_edges))
        print("selected feature mask shape:", selected_feat.shape if selected_feat is not None else "No feature mask")


        # Remove subgraph from the original graph
        if args.explainer_name == "CFExplainer":
            remaining_graph_constructor = CFSubgraphRemover(data, args.base_dir, args.explainer_name, args.dataset, args.node_choose, device=DEVICE)
            remaining_graph_constructor.load_data()
            remaining_graph = remaining_graph_constructor.get_remaining_graph()
            num_node = remaining_graph_constructor.get_node_count()
            num_edge = remaining_graph_constructor.get_edge_count()
        else:
            remaining_graph_constructor = RemainingGraphConstructor(data, selected_edges, selected_feat_mask=selected_feat, device=DEVICE) # selected_feat 沒有經過 feature_to_node 才會有
            remaining_graph = remaining_graph_constructor.get_remaining_graph()


# ===============Revert to Original Graph================ #
        # if use feature to node, revert the feature node to original node (add feature value into original graph)
        if args.feature_to_node:
            print("Reverting feature nodes to original nodes...")
            revertor = FeatureNodeReverter(feature_type=feature_type, device=DEVICE)
            remaining_graph = revertor.revert(remaining_graph, ori_data)
            num_features = remaining_graph.x.size(1)

            if args.only_feature_node: # 只會用在[只移除特徵]
                # 要加回 node-node 邊，不然還原回原圖會沒有邊
                remaining_graph.edge_index = ori_data.edge_index.clone()
                remaining_graph.edge_weight = ori_data.edge_weight.clone() if ori_data.edge_weight is not None else None

            # 如果設定了要遮蔽相同的特徵，且有 selected_feat_ids，就從 ori_data 上遮蔽
            if args.same_feat and selected_feat_ids is not None and len(selected_feat_ids) > 0 and args.fraction_feat > 0: # 如果沒有使用feat2node, selected_feat_ids會是空的
                print("改成 remove same features")
                remaining_graph.x, removed_feat_ids = remove_top_common_features(
                    x=ori_data.x, # 放入的是原始的data
                    selected_feat_ids=selected_feat_ids,
                    fraction_feat=args.fraction_feat
                ) # 把most common features 都變成0

        if args.only_structure:
            # 如果只透過結構資訊選邊，還原圖做預測時依然要加入原始特徵 (才會是移除只有結構的核心子圖)
            remaining_graph.x = ori_data.x.clone()


        # 希望模型跟結果都存在 split_id 資料夾下。但檔名是trial_number開頭
        save_dir = os.path.join(args.run_mode, f"split_{split_id}")

        logger = ExperimentLogger(file_name=args.filename, note=args.note, copy_old=True, run_mode=save_dir)
        trial_number = logger.get_next_trial_number(args.dataset + "_remaining_graph")

        # 匯出核心子圖 mask  (存在 split_id folder 下)
        extractor = CoreSubgraphExtractor(
            ori_data=ori_data,
            remaining_graph=remaining_graph,
            save_dir=save_dir,
            dataset=args.dataset,
            is_undirected=True, # 原本的 GNN 通常是 undirected
            trial_number=trial_number
        )
        extractor.compute_removed_masks()
        extractor.summary()  # 印出移除的特徵和邊的統計
        extractor.save()

        # 移除特徵全為0的欄位 (只在移除相同特徵時使用)
        if args.same_feat:
            remaining_graph.x, zero_feature_cols = remove_all_zero_features(remaining_graph.x)
            print(f"Original features: {ori_data.x.shape[1]}, Removed features (all-zero): {len(zero_feature_cols)}")
        
        num_features = remaining_graph.x.size(1)

        # 如果 data 沒有邊了，報錯
        if remaining_graph.edge_index.size(1) == 0:
            raise ValueError("Remaining graph has no edges after removing the subgraph. Please check the selected edges and feature masks.")

        # Train GNN on the remaining graph
        print("\nTraining GNN on the remaining graph after removing subgraph...")
        print("Final data imput to model:", remaining_graph)

        print(f"Training Classification Model - Trial {trial_number}")
        trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=remaining_graph, 
                                    num_features=num_features, num_classes=num_classes, 
                                    model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                    trial_number=trial_number, device=DEVICE,
                                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                    run_mode=save_dir)


        result = trainer.run()


        # Save experiment results
        if args.selector_type == "random":
            logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, fraction=args.fraction, fraction_feat=args.fraction_feat, remove_feat=zero_feature_cols, 
                                  only_structure=args.only_structure, feature_to_node=args.feature_to_node, only_feature_node=args.only_feature_node, structure_mode=args.structure_mode, same_feat=args.same_feat)
        
        elif args.selector_type == "explainer":
            logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, explaner=args.explainer_name, node_choose=args.node_choose, 
                                  fraction=args.fraction, fraction_feat=args.fraction_feat, remove_feat=zero_feature_cols,
                                  only_structure=args.only_structure, feature_to_node=args.feature_to_node, only_feature_node=args.only_feature_node, structure_mode=args.structure_mode, same_feat=args.same_feat)
        
        
        elif args.selector_type == "random_walk":
            logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, 
                                  walk_length=args.walk_length, num_walks=args.num_walks, node_choose=args.node_choose, fraction=args.fraction, fraction_feat=args.fraction_feat, 
                                  ori_edge_visit_ratio=ori_edge_visit_ratio, feat_edge_visit_ratio=feat_edge_visit_ratio, remove_feat=zero_feature_cols, 
                                  only_structure=args.only_structure, feature_to_node=args.feature_to_node, only_feature_node=args.only_feature_node, structure_mode=args.structure_mode, same_feat=args.same_feat)
        

        print("Experiment finished and results saved.")


