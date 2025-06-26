import argparse
import torch
import os
import numpy as np
import pickle
from utils.device import DEVICE
from models.explainer import SubgraphExplainer
from models.basic_GCN import GCN2Classifier, GCN3Classifier
from data.prepare_split import load_split_csv


# 看split_id多少，抓那次的模型, 並讀取那10次的split檔，聚集10次training node，一起生成解釋子圖，
def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run GNN Explainer for node regression.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the dataset.")
    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")

    parser.add_argument("--explainer_type", type=str, default="GNNExplainer", choices=["GNNExplainer", "PGExplainer", "DummyExplainer", "CFExplainer"], help="Type of explainer to use")
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs for explainer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for explainer")
    parser.add_argument("--run_mode", type=str, default="stage2_edge_0.5", help="Run mode") # 如果生成test的解釋子圖, run_mode前改成 "test_stage2_edge_0.5"
    parser.add_argument("--stage1_path", type=str, default="saved/stage1", help="Directory for stage1 results")
    parser.add_argument("--trial_name", type=int, default=0, help="Trial name for saving results")
    # cf_explainer
    parser.add_argument("--cf_beta", type=float, default=0.5, help="Tradeoff for dist loss")
    
    # split settings
    parser.add_argument("--split_start", type=int, default=0, help="Start split id (inclusive)")
    parser.add_argument("--split_end", type=int, default=0, help="End split id (inclusive)")

    # 補解釋
    parser.add_argument("--check_unexplained", action="store_true", help="Check which train nodes have not been explained yet.")

    return parser.parse_args()


def filter_unexplained_nodes(train_nodes, save_dir):
    """
    從目前已生成的結果中，找出最大 node_id，從該節點之後開始。
    如果都沒生成，從頭開始。
    """
    if not os.path.exists(save_dir):
        print(f"[Check] {save_dir} does not exist. Start from the beginning.")
        return train_nodes

    existing_nodes = []
    for filename in os.listdir(save_dir):
        if filename.startswith("node_") and filename.endswith(".npz"):
            node_id = int(filename.replace("node_", "").replace(".npz", ""))
            existing_nodes.append(node_id)

    if not existing_nodes:
        print("[Check] No existing explanations found. Start from the beginning.")
        return train_nodes

    max_existing = max(existing_nodes)
    unexplained_nodes = [n for n in train_nodes if n > max_existing]

    print(f"[Check] Found {len(existing_nodes)} explained nodes, max id: {max_existing}")
    print(f"[Check] {len(unexplained_nodes)} unexplained nodes remain.")

    return unexplained_nodes



if __name__ == "__main__":
    args = parse_args()
    print(f"Using DEVICE: {DEVICE}")

    # model class
    model_mapping = {"GCN2": GCN2Classifier, "GCN3": GCN3Classifier}
    model_class = model_mapping[args.model]

    # Select nodes to explain 
    for split_id in range(args.split_start, args.split_end + 1):

        print(f"\n===== [Split {split_id}] =====")
        
        # 不管有沒有經過 feature to node，都會讀入 feat2node_graph 內資料夾的 graph
        graph_path = os.path.join(args.stage1_path, f"split_{split_id}", "feat2node_graph", args.dataset, "converted_data.pt")
        
        if not os.path.exists(graph_path):
            if "saved/stage1" == args.stage1_path:
                print(f"[Warning] Converted graph not found: {graph_path}")
                print("[Info] Reload original data from dataset.")
                
                from data.dataset_loader import GraphDatasetLoader  # 確保只有需要時才 import
                loader = GraphDatasetLoader(normalize=args.normalize)  
                data, _, _, _, _ = loader.load_dataset(args.dataset)
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

        train_nodes = data.train_mask.nonzero(as_tuple=True)[0].cpu().tolist() # 原始節點的編號        
        # # try only one node
        # print("====Note: For testing, only one node will be selected.====")
        # train_nodes=train_nodes[0:2] 

        # model path
        model_path = os.path.join(args.stage1_path, f"split_{split_id}", "model", args.dataset, f"{args.trial_name}_{model_class.__name__}.pth")
        print(f"Loading model from {model_path}")
    
        # Initialize explainer
        explainer = SubgraphExplainer(
            model_class=model_class,
            dataset=args.dataset,
            data=data,
            model_path=model_path,
            explainer_type=args.explainer_type,
            hop= 2 if args.model == "GCN2" else 3,
            epoch=args.epoch,
            lr=args.lr,
            run_mode=args.run_mode,
            trial_name=args.trial_name,
            split_id=split_id,
            cf_beta=args.cf_beta,
        )

        if args.check_unexplained:
            save_dir = os.path.join("saved", args.run_mode, f"split_{split_id}", args.explainer_type, args.dataset, f"{args.trial_name}_{explainer.model_class.__name__}")
            train_nodes = filter_unexplained_nodes(train_nodes, save_dir)

        for node_idx in train_nodes:
            print(f"\nExplaining node {node_idx}.")
            result = explainer.explain_node(node_idx, data, save=True)


        print(f"\n[Split {split_id}] Explanation completed.")