import argparse
import torch
import os
import numpy as np
import pickle
from utils.device import DEVICE
from data.dataset_loader import GraphDatasetLoader
from models.explainer import SubgraphExplainer
from models.basic_GCN import GCN2Classifier, GCN3Classifier
from data.feature2node import FeatureNodeConverter
from data.prepare_split import load_split_csv
from data.structure import StructureFeatureBuilder, extract_edges

# 看repeat_id多少，抓那次的模型, 並讀取那10次的split檔，聚集10次training node，一起生成解釋子圖，
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
    
    # cf_explainer
    parser.add_argument("--cf_beta", type=float, default=0.5, help="Tradeoff for dist loss")
    
    # only structure
    parser.add_argument("--only_structure", action="store_true", help="Use only structural information (all features set to 1)")
    # feature to node
    parser.add_argument("--feature_to_node", action="store_true", help="Convert features into nodes and edges.")
    # Only use feature-node (no node-node edges)
    parser.add_argument("--only_feature_node", action="store_true", help="Use only feature-node edges, no node-node edges.")
    # Structure Mode
    parser.add_argument("--structure_mode", type=str, default="random+imp", choices=["one", "random+imp"], help="Mode for structure features: 'one' or 'random+imp'")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Using DEVICE: {DEVICE}")

    # Load dataset
    loader = GraphDatasetLoader(args.normalize)
    data, num_features, _, feature_type, _ = loader.load_dataset(args.dataset)
    data = data.to(DEVICE)

    # 1. Feature to node conversion
    if args.feature_to_node:
        print("Converting node features into feature-nodes...")
        converter = FeatureNodeConverter(feature_type=feature_type, device=DEVICE)
        # FeatureNodeConverter 提供feature連到node後的所有邊，並多了 node_node_mask, node_feat_mask (後續要記得處理!)
        data = converter.convert(data)


    # 統一更新 edge_index, edge_weight (不論原 graph 或 feature to node 都可以用 extract_edges)
    edge_index, edge_weight = extract_edges(data, args.feature_to_node, args.only_feature_node)
    data.edge_index = edge_index
    data.edge_weight = edge_weight


    # Model mapping
    model_mapping = {"GCN2Classifier": GCN2Classifier, "GCN3Classifier": GCN3Classifier}
    
    # model class
    model_class = model_mapping[f"{args.model}Classifier"]

    # Select nodes to explain 
    for repeat_id in range(10): 
        # Load the split mask
        train_mask, _, _, _ = load_split_csv(args.dataset, repeat_id, DEVICE) # 這裏的mask是原dataset的長度
        train_nodes = train_mask.nonzero(as_tuple=True)[0].cpu().tolist() # 原始節點的編號
        
        # # try only one node
        # print("====Note: For testing, only one node will be selected.====")
        # train_nodes=train_nodes[0:2] 

        # Build feature X
        if (args.feature_to_node or args.only_structure):
            if args.structure_mode == "random+imp":
                embedding_save_dir = os.path.join(args.stage1_path, "embedding", args.dataset)
                embedding_save_path = os.path.join(embedding_save_dir, f"{repeat_id}_embedding.npy")
                print(f"[Repeat {repeat_id}] Loading embedding from {embedding_save_path}")

                embedding_np = np.load(embedding_save_path)
                print(f"loaded embedding from {embedding_save_path}, shape: {embedding_np.shape}")
                embedding_tensor = torch.tensor(embedding_np, device=DEVICE, dtype=torch.float)

                builder = StructureFeatureBuilder(
                    data=data, device=DEVICE, dataset_name=args.dataset,
                    feature_to_node=args.feature_to_node,
                    only_feature_node=args.only_feature_node,
                    only_structure=args.only_structure,
                    mode=args.structure_mode,
                    emb_dim=32,
                    normalize_type="row_l1",
                    learn_embedding=False,
                    external_embedding=embedding_tensor
                )
            else: # "one" mode
                print(f"[Repeat {repeat_id}] Using StructureFeatureBuilder with mode={args.structure_mode} (no external embedding)")
                builder = StructureFeatureBuilder(
                    data=data, device=DEVICE, dataset_name=args.dataset,
                    feature_to_node=args.feature_to_node,
                    only_feature_node=args.only_feature_node,
                    only_structure=args.only_structure,
                    mode=args.structure_mode,
                    emb_dim=32,
                    normalize_type="row_l1",
                    learn_embedding=False
                )

            structure_x = builder.build()
            data.x = structure_x

        else:
            print(f"[Repeat {repeat_id}] Using original data.x (no StructureFeatureBuilder rebuild).")


        # model path
        model_path = os.path.join(args.stage1_path, "model", args.dataset, f"{repeat_id}_{model_class.__name__}.pth")
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
            trial_name=repeat_id,
            cf_beta=args.cf_beta,
        )


        # Explain each node
        cf_success_nodes = []
        cf_fail_nodes = []

        for node_idx in train_nodes:
            print(f"\nExplaining node {node_idx}.")
            result = explainer.explain_node(node_idx, data, save=True)

            # 只有CF要計算
            if args.explainer_type == "CFExplainer":
                if result is not None:
                    cf_success_nodes.append(node_idx)
                else:
                    cf_fail_nodes.append(node_idx)

        if args.explainer_type == "CFExplainer":
            # Print summary
            print("\n========== CF Explanation Summary ==========")
            print(f"Total nodes selected: {len(train_nodes)}")
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
            save_dir = os.path.join("saved", args.run_mode, args.explainer_type, args.dataset, f"{repeat_id}_{explainer.model_class.__name__}")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, "cf_summary.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(cf_summary, f)

            print(f"Saved CF explanation summary to {save_path}")

            print("Node explanations completed and saved.")
