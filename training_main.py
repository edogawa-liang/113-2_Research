import argparse
import torch
import numpy as np
import os
from data.dataset_loader import GraphDatasetLoader
from models.basic_GCN import GCN2Classifier, GCN3Classifier
from trainer.gnn_trainer import GNNClassifierTrainer
from utils.save_result import ExperimentLogger
from models.basic_mlp import MLPClassifier
from trainer.mlp_trainer import MLPClassifierTrainer
from data.feature2node import FeatureNodeConverter
from data.structure import StructureFeatureBuilder, extract_edges
from data.prepare_split import load_split_csv
from utils.device import DEVICE

# train/valid/test 每一次都重抽 (確定方法有效)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN model with specified parameters.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the dataset.")
    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3", "MLP"], help="Model type")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for binary classification")
    
    # result settings
    parser.add_argument("--run_mode", type=str, default="try", help="Experiment run mode: 'try' (testing) or formal runs")
    parser.add_argument("--result_filename", type=str, default="result", help="File name for results document")
    parser.add_argument("--note", type=str, default="", help="Note for the experiment")
    parser.add_argument("--copy_old", type=lambda x: x.lower() == "true", default=True, help="Whether to backup old experiment data (true/false).")
    
    # only structure
    parser.add_argument("--only_structure", action="store_true", help="Use only structural information (all features set to 1)")

    # feature to node
    parser.add_argument("--feature_to_node", action="store_true", help="Convert features into nodes and edges.")

    # Only use feature-node (no node-node edges)
    parser.add_argument("--only_feature_node", action="store_true", help="Use only feature-node edges, no node-node edges.")
    
    # Structure Mode
    parser.add_argument("--structure_mode", type=str, default=None, choices=["one+imp", "random+imp"], help="Mode for structure features: 'one' or 'random+imp'")

    # structure mode 是 "random+imp" 時，要不要使用 learnable embedding
    parser.add_argument("--learn_embedding", action="store_true", help="Use learnable random embedding")
    parser.add_argument("--emb_dim", type=int, default=16, help="Dimension of the embedding if using random+imp mode")

    # split settings
    parser.add_argument("--split_start", type=int, default=0, help="Start split id (inclusive)")
    parser.add_argument("--split_end", type=int, default=0, help="End split id (inclusive)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, _, feature_type, _ = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)

    # define pad_mask function
    def pad_mask(mask, pad_len):
        return torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool, device=mask.device)], dim=0)

    # 1. Feature to node conversion
    if args.feature_to_node:
        print("Converting node features into feature-nodes...")
        converter = FeatureNodeConverter(feature_type=feature_type, num_nodes=data.num_nodes, device=DEVICE)
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
            emb_dim=args.emb_dim if args.structure_mode == "random+imp" else 0, # 如果是 one+imp 就不需要 embedding
            normalize_type="row_l1",
            learn_embedding=args.learn_embedding, # 在訓練階段時是 True, Inference 階段是 False
            external_embedding=None # 只有inference時才會有
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


    # Split 10 time selecting different splits
    # 在 train/valid/test 情況下是 load 不同種組合
    for split_id in range(args.split_start, args.split_end + 1):

        print(f"\n===== [Split {split_id}] =====")
        
        # 希望模型跟結果都存在 split_id 資料夾下。但檔名是trial_number開頭
        save_dir = os.path.join(args.run_mode, f"split_{split_id}")

        # Load the split mask
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

        # 統一存一份 graph，不管是不是 feature_to_node，避免後續錯誤
        output_dir = os.path.join("saved", save_dir, "feat2node_graph", args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(data, os.path.join(output_dir, "converted_data.pt"))
        print(f"Saved graph to {output_dir}/converted_data.pt")

        # 只有 feature_to_node 才存 importance
        if args.feature_to_node:
            torch.save(converter.node_feature_vs_structure.data.cpu(), os.path.join(output_dir, "node_feature_vs_structure_imp.pt"))
            print("Saved node-feature importance.")


        # Initialize logger
        logger = ExperimentLogger(file_name=args.result_filename, note=args.note, copy_old=args.copy_old, run_mode=save_dir)

        # Loop over all modified graphs
        print(f"\nTraining on Graph - Split {split_id}")
        label_source = "Original Label"
        try:
            trial_number = logger.get_next_trial_number(args.dataset + "_classification")
            num_classes = len(torch.unique(data.y))
            print(f"Training Classification Model - Trial {trial_number}")
            
            if args.model == "MLP":
                trainer = MLPClassifierTrainer(dataset_name=args.dataset, data=data,
                                    num_features=num_features, num_classes=num_classes,
                                    model_class= MLPClassifier, 
                                    trial_number=trial_number, device=DEVICE,
                                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                    run_mode=save_dir, threshold=args.threshold)
            else: # GNN
                trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=data, 
                                            num_features=num_features, num_classes=num_classes,  
                                            model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                            trial_number=trial_number, device=DEVICE,
                                            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, 
                                            run_mode=save_dir, threshold=args.threshold,
                                            extra_params=[builder.embedding.weight] if (args.feature_to_node or args.only_structure) and args.structure_mode == "random+imp" and args.learn_embedding else None)
            
            result = trainer.run()

            logger.log_experiment(args.dataset + "_cls", result, label_source, split_id=split_id)

        except ValueError as e:
            raise