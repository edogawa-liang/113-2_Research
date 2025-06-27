import torch
import os
import numpy as np
import pandas as pd
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

from data.node2feature import FeatureNodeReverter
from utils.node_coverage_summary import save_coverage_log
from utils.feature_utils import remove_all_zero_features, remove_top_common_features
from subgraph_selector.subgraph import CoreSubgraphExtractor
from data.prepare_split import load_split_csv


# 還沒試跑

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

    parser.add_argument("--stage1_path", type=str, default="saved/stage1", help="Directory for stage1 results")
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

    # trial settings for stage 1   (not the same trial as stage 3)
    parser.add_argument("--trial_name", type=int, default=0, help="Trial name for saving results")

    # split settings
    parser.add_argument("--split_start", type=int, default=0, help="Start repeat id (inclusive)")
    parser.add_argument("--split_end", type=int, default=0, help="End repeat id (inclusive)")

    return parser.parse_args()


# 比較對象是 Original graph, (一般的含有節點特徵的圖)
if __name__ == "__main__":
    args = parse_args()
    selected_feat = None
    selected_feat_ids = None
    selected_feat_mask = None
    zero_feature_cols = None

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    ori_data, num_features, num_classes, feature_type, num_ori_edges = loader.load_dataset(args.dataset)
    ori_data = ori_data.to(DEVICE)

    # define pad_mask function
    def pad_mask(mask, pad_len):
        return torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool, device=mask.device)], dim=0)


    # 每次 repeat 挑選的節點都不一樣，分別找子圖與訓練模型
    for split_id in range(args.split_start, args.split_end + 1):
        print(f"\n===== [Split {split_id}] =====")
        # 希望模型跟結果都存在 split_id 資料夾下。但檔名是trial_number開頭
        save_dir = os.path.join(args.run_mode, f"split_{split_id}")

        logger = ExperimentLogger(file_name=args.filename, note=args.note, copy_old=True, run_mode=save_dir)
        trial_number = logger.get_next_trial_number(args.dataset)


        graph_path = os.path.join(args.stage1_path, f"split_{split_id}", "feat2node_graph", args.dataset, "converted_data.pt")
        if not os.path.exists(graph_path):
            if "saved/stage1" == args.stage1_path:
                print(f"[Warning] Converted graph not found: {graph_path}")
                print("[Info] Reload original data from dataset.")
                data = ori_data.clone()  # 這裏的 data 是原始的圖
                train_mask, val_mask, test_mask, unknown_mask = load_split_csv(args.dataset, split_id, DEVICE) # 這裏的mask是原dataset的長度
                data.train_mask, data.val_mask, data.test_mask, data.unknown_mask = train_mask, val_mask, test_mask, unknown_mask   
                data = data.to(DEVICE)
                print(data)
            else:
                raise FileNotFoundError(f"Converted graph not found: {graph_path}")
            
        else:
            print(f"[Split {split_id}] Loading converted graph from {graph_path}")
            data = torch.load(graph_path, map_location=DEVICE)
            data = data.to(DEVICE)
            print(data)
        
        # ori_data 也要更新 mask
        ori_data.train_mask, ori_data.val_mask, ori_data.test_mask, ori_data.unknown_mask = \
        data.train_mask, data.val_mask, data.test_mask, data.unknown_mask


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

        print(f"======= Using {args.selector_type} Selector =======")

        # Select subgraph
        if args.selector_type == "random": # 使用時不要加feature_to_node
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
            base_dir = os.path.join(args.base_dir, f"split_{split_id}")
            if args.explainer_name != "CFExplainer": # CF另外處理
                
                selector = ExplainerEdgeSelector(
                    data=data,
                    base_dir=base_dir,
                    explainer_name=args.explainer_name,
                    dataset_name=args.dataset,
                    selected_nodes=selected_nodes,
                    top_k_percent=args.fraction,
                    feature_type=feature_type,
                    device=DEVICE,
                    top_k_percent_feat=args.fraction_feat,
                    use_feature_to_node=args.feature_to_node, # 若要使用特徵，分成直接使用(node_mask)還是將特徵轉為結構使用(edge_mask)
                    only_feature_node = args.only_feature_node
                )            
                selector.load_data(args.trial_name)
                selector.plot_edge_distribution()

                # 只有需要選特徵邊時，selected_feat_ids 才會有值，不然就是空的
                selected_edges, selected_feat_ids = selector.select_edges() # 選到的所有邊 (包含特徵邊), 即這些特徵是哪幾個
                
                # 如果需要使用解釋子圖自身的特徵重要度
                if not args.feature_to_node and args.fraction_feat > 0:
                    # 改 select_node_features
                    selected_feat = selector.select_node_features(same_feat=args.same_feat)


        elif args.selector_type == "random_walk":
            # 改成直接傳入node
            # 如果會用 random walk 選特徵，一定有經過 feature_to_node
            selector = RandomWalkEdgeSelector(data, fraction=args.fraction, selected_nodes=selected_nodes,
                                            walk_length=args.walk_length, num_walks=args.num_walks, feature_type=feature_type, 
                                            device=DEVICE, top_k_percent_feat=args.fraction_feat, only_feature_node=args.only_feature_node,)
            selected_edges, selected_feat_ids, ori_edge_visit_ratio, feat_edge_visit_ratio = selector.select_edges()


        # Get the remaining graph. Remove subgraph from the original graph
        # 如果用 CFExplainer 選特徵，一定有經過 feature_to_node
        if args.explainer_name == "CFExplainer":
            remaining_graph_constructor = CFSubgraphRemover(data, base_dir, args.dataset, device=DEVICE, selected_nodes=selected_nodes, 
                                                            fraction=args.fraction, top_k_percent_feat=args.fraction_feat, 
                                                            only_feature_node = args.only_feature_node)
            remaining_graph_constructor.load_data(args.trial_name)
            remaining_graph, stats, selected_feat_ids = remaining_graph_constructor.get_remaining_graph()
            
            # save stats
            cf_summary_dir = os.path.join(base_dir, args.dataset, "cf_summary")
            os.makedirs(cf_summary_dir, exist_ok=True)

            summary_save_path = os.path.join(cf_summary_dir, f"{trial_number}_summary.csv")
            df = pd.DataFrame([stats])  # stats 已經是字典
            df.to_csv(summary_save_path, index=False)
            print(f"CF summary saved to {summary_save_path}")

        else:
            remaining_graph_constructor = RemainingGraphConstructor(data, selected_edges, selected_feat_mask=selected_feat, device=DEVICE) # selected_feat 沒有經過 feature_to_node 才會有
            remaining_graph = remaining_graph_constructor.get_remaining_graph()


        print("\nRemaining graph before reverting feature nodes:\n", remaining_graph)


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


        # 匯出核心子圖 mask  (存在 split_id folder 下)
        extractor = CoreSubgraphExtractor(
            ori_data=ori_data,
            remaining_graph=remaining_graph,
            save_dir=save_dir,
            dataset=args.dataset,
            trial_number=trial_number
        )
        extractor.compute_removed_masks()
        extractor.summary()  # 印出移除的特徵和邊的統計
        extractor.save()

        keep_feature_cols = list(range(ori_data.x.shape[1]))
        # 移除特徵全為0的欄位 (只在移除相同特徵時使用)
        if args.same_feat:
            remaining_graph.x, zero_feature_cols = remove_all_zero_features(remaining_graph.x)
            print(f"Original features: {ori_data.x.shape[1]}, Removed features (all-zero): {len(zero_feature_cols)}")
            keep_feature_cols = [i for i in range(ori_data.x.shape[1]) if i not in zero_feature_cols]

        num_features = remaining_graph.x.size(1)

        # 如果 data 沒有邊了，報錯
        if remaining_graph.edge_index.size(1) == 0:
            raise ValueError("Remaining graph has no edges after removing the subgraph. Please check the selected edges and feature masks.")

        # 如果有移除特徵，應該在測試節點保留完整特徵，
        if args.fraction_feat > 0:
            print("Restoring full features for test nodes...")
            test_node_idx = torch.where(remaining_graph.test_mask)[0]
            ori_x_reduced = ori_data.x[:, keep_feature_cols]
            remaining_graph.x[test_node_idx] = ori_x_reduced[test_node_idx]

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
            logger.log_experiment(args.dataset, result, label_source="Original", selector_type=args.selector_type, fraction=args.fraction, fraction_feat=args.fraction_feat, remove_feat=zero_feature_cols, 
                                  only_structure=args.only_structure, feature_to_node=args.feature_to_node, only_feature_node=args.only_feature_node, structure_mode=args.structure_mode, same_feat=args.same_feat)
        
        elif args.selector_type == "explainer":
            logger.log_experiment(args.dataset, result, label_source="Original", selector_type=args.selector_type, explainer=args.explainer_name, node_choose=args.node_choose, 
                                  fraction=args.fraction, fraction_feat=args.fraction_feat, remove_feat=zero_feature_cols,
                                  only_structure=args.only_structure, feature_to_node=args.feature_to_node, only_feature_node=args.only_feature_node, structure_mode=args.structure_mode, same_feat=args.same_feat)
        
        
        elif args.selector_type == "random_walk":
            logger.log_experiment(args.dataset, result, label_source="Original", selector_type=args.selector_type, 
                                  walk_length=args.walk_length, num_walks=args.num_walks, node_choose=args.node_choose, fraction=args.fraction, fraction_feat=args.fraction_feat, 
                                  ori_edge_visit_ratio=ori_edge_visit_ratio, feat_edge_visit_ratio=feat_edge_visit_ratio, remove_feat=zero_feature_cols, 
                                  only_structure=args.only_structure, feature_to_node=args.feature_to_node, only_feature_node=args.only_feature_node, structure_mode=args.structure_mode, same_feat=args.same_feat)
        

        print("Experiment finished and results saved.")


