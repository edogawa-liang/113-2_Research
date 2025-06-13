# 新增一個test_remaining_main 直接匯入解釋子圖的結果，然後直接跑 test
import torch
import os
import numpy as np
import argparse
from utils.device import DEVICE

from data.dataset_loader import GraphDatasetLoader

from models.basic_GCN import GCN2Classifier, GCN3Classifier
from trainer.gnn_trainer import GNNClassifierTrainer
from utils.save_result import ExperimentLogger

from data.prepare_split import load_split_csv
from data.split_unknown_to_test import load_split_test
from utils.feature_utils import remove_all_zero_features
from subgraph_selector.subgraph import CoreSubgraphExtractor


# 核心子圖包含整個節點
# 移除部分邊後的節點分類結果

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN after removing a selected subgraph.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the dataset.")
    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")

    # explainer
    parser.add_argument("--run_mode", type=str, default="try", help="Run mode") 
    parser.add_argument("--filename", type=str, default="result", help="File name for saving results")
    parser.add_argument("--note", type=str, default="", help="Note for the experiment")

    
    # repeat settings (for testing different group)
    parser.add_argument("--repeat_start", type=int, default=0, help="Start repeat id (inclusive)")
    parser.add_argument("--repeat_end", type=int, default=9, help="End repeat id (inclusive)")

    # 使用的 split_id (第幾次的triain/valid/test)
    parser.add_argument("--split_id", type=int, default=0, help="Split ID to use for fixed train/valid masks (default: 0)")

    # trial number (要對應好)
    parser.add_argument("--trial_number", type=int, default=0, help="Trial number for logging Model")

    return parser.parse_args()


# 比較對象是 Original graph, (一般的含有節點特徵的圖)
if __name__ == "__main__":
    args = parse_args()

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, num_classes, feature_type, num_ori_edges = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)
    ori_data = data.clone() # 備份原始資料

    save_dir = os.path.join("saved", "core_subgraph_mask", f"split_{args.split_id}", args.run_mode, args.dataset)


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
    extractor = CoreSubgraphExtractor(
    ori_data=ori_data,
    remaining_graph=None, # not used for loading
    save_dir=save_dir,
    dataset=args.dataset,
    is_undirected=True, # 原本的 GNN 通常是 undirected
    trial_number=args.trial_number
    )
    feature_removed_mask, edge_removed_mask = extractor.load()

    # Build remaining graph
    remaining_graph = data.clone() 
    
    remaining_feature_mask = (feature_removed_mask == 0).astype(np.int32)
    remaining_edge_mask = (edge_removed_mask == 0).astype(np.int32)
    # feature
    ori_x_np = ori_data.x.cpu().numpy()
    remaining_x_np = ori_x_np * remaining_feature_mask  # element-wise mask
    remaining_graph.x = torch.tensor(remaining_x_np, device=DEVICE, dtype=ori_data.x.dtype)

    # edge
    ori_edge_index = ori_data.edge_index.clone()
    edge_mask_tensor = torch.tensor(remaining_edge_mask[0], device=DEVICE, dtype=torch.bool)
    remaining_graph.edge_index = ori_edge_index[:, edge_mask_tensor]

    # Remove all-zero features
    if args.run_mode.endswith("_samefeat"):
        remaining_graph.x, zero_feature_cols = remove_all_zero_features(remaining_graph.x)
        print(f"Original features: {ori_data.x.shape[1]}, Removed features (all-zero): {len(zero_feature_cols)}")

    num_features = remaining_graph.x.size(1)


    # Inference GNN on the remaining graph
    print(f"Training Classification Model - Trial {args.trial_number}")
    trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=remaining_graph, 
                                num_features=num_features, num_classes=num_classes, 
                                model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                trial_number=args.trial_number, device=DEVICE,
                                run_mode=save_dir)


    # 因為這時的 run_mode 後面會+split_0，所以要去掉
    model_dir = os.path.join("saved", save_dir.replace(f"_split{args.split_id}", ""), "model", args.dataset)
    model_name = f"{args.split_id}_{trainer.model_name}.pth"  # 你的 save_model_path 是這個 pattern
    path_to_model = os.path.join(model_dir, model_name)
    print(f"Loading fixed model from {path_to_model}")
    trainer.load_model(path_to_model)

    save_dir = os.path.join(args.run_mode, f"split_{args.split_id}")
    print(f"Saving results to {save_dir}")


    # 每次repeat 挑選的節點都不一樣，分別找子圖與訓練模型
    for repeat_id in range(args.repeat_start, args.repeat_end + 1):
        print(f"\n===== [Test Repeat {repeat_id}] =====")
        test_mask = load_split_test(args.dataset, args.split_id, repeat_id, DEVICE)
        remaining_graph.test_mask = test_mask

        logger = ExperimentLogger(file_name=args.filename, note=args.note, copy_old=True, run_mode=save_dir)
        trial_number = logger.get_next_trial_number(args.dataset + "_remaining_graph")

        result = trainer.test()

        logger.log_experiment(args.dataset + "_remaining_graph", result, label_source = "Original", repeat_id=repeat_id, model_name=model_name)

        print("Experiment finished and results saved.")
