# 新增一個test_remaining_main 直接匯入解釋子圖的結果，然後直接跑 test
import torch
import os
import numpy as np
import pandas as pd
import argparse
from utils.device import DEVICE

from data.dataset_loader import GraphDatasetLoader

from models.basic_GCN import GCN2Classifier, GCN3Classifier
from trainer.gnn_trainer import GNNClassifierTrainer

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

    # 挑選想要 test 10 次的模型 (trial)
    parser.add_argument("--trial_start", type=int, default=0, help="Start trial ID (inclusive)")
    parser.add_argument("--trial_end", type=int, default=0, help="End trial ID (exclusive)")

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


    print("\nLoading fixed train/valid masks...")
    # 固定取 repeat_id = 0 的 train/valid mask
    train_mask, val_mask, _, unknown_mask = load_split_csv(args.dataset, args.split_id, DEVICE)
    num_orig_nodes = train_mask.shape[0]
    num_total_nodes = data.x.shape[0]

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.unknown_mask = unknown_mask


    for trial_id in range(args.trial_start, args.trial_end + 1):

        print(f"\n=== [Trial {trial_id}] ===")

        # Load core subgraph mask
        save_mask_dir = os.path.join("saved", "core_subgraph_mask", f"split_{args.split_id}", args.run_mode, args.dataset)
        extractor = CoreSubgraphExtractor(ori_data, None, save_mask_dir, args.dataset, trial_id)
        feature_removed_mask, edge_removed_mask = extractor.load()

        # Build remaining graph
        ## feature
        remaining_graph = data.clone()
        remaining_x_np = ori_data.x.cpu().numpy() * (feature_removed_mask == 0).astype(np.int32)
        remaining_graph.x = torch.tensor(remaining_x_np, device=DEVICE, dtype=ori_data.x.dtype)
        ## edge
        edge_mask_tensor = torch.tensor(edge_removed_mask[0], device=DEVICE, dtype=torch.bool)
        remaining_graph.edge_index = ori_data.edge_index[:, edge_mask_tensor]

        # Remove all-zero features
        if args.run_mode.endswith("_samefeat"):
            remaining_graph.x, zero_cols = remove_all_zero_features(remaining_graph.x)
            print(f"Removed {len(zero_cols)} all-zero features.")

        num_features = remaining_graph.x.size(1)

    
        # 每次repeat 挑選的節點都不一樣，分別找子圖與訓練模型
        for repeat_id in range(args.repeat_start, args.repeat_end + 1):
            print(f"\n===== [Test Repeat {repeat_id}] =====")
            test_mask = load_split_test(args.dataset, args.split_id, repeat_id, DEVICE)
            remaining_graph.test_mask = test_mask

            # # 如果有移除特徵，應該在測試節點保留完整特徵，
            print("Restoring full features for test nodes...")
            test_node_idx = torch.where(test_mask)[0]
            remaining_graph.x[test_node_idx] = ori_data.x[test_node_idx]

            # Inference GNN on the remaining graph
            print(f"Training Classification Model - Trial {trial_id}")
            trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=remaining_graph, 
                                        num_features=num_features, num_classes=num_classes, 
                                        model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                        trial_number=trial_id, device=DEVICE,)

            model_dir = os.path.join("saved", args.run_mode, "model", args.dataset)
            model_name = f"{args.split_id}_{trainer.model_name}.pth"
            model_path = os.path.join(model_dir, model_name)
            print(f"Loading fixed model from {model_path}")
            trainer.load_model(model_path)
            
            result = trainer.test()
            result["repeat_id"] = repeat_id 
            result["trial_id"] = trial_id  

            # Save results
            save_dir = os.path.join("saved", args.run_mode, f"split_{args.split_id}", "repeat_result", args.dataset)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{trial_id}_{trainer.model_name}.csv")

            df = pd.DataFrame([result])
            if os.path.exists(save_path):
                df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                df.to_csv(save_path, index=False)
            
            print("Experiment finished and results saved.")
