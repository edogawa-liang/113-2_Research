import pandas as pd
import os

def save_coverage_log(args, coverage_stats, selected_nodes, save_dir="saved/node_coverage"):
    os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在

    # 準備一筆要寫入的資料
    record = {
        "split_id": args.split_id,
        "selector_type": args.selector_type,
        "dataset_name": args.dataset,
        "node_choose": args.node_choose,
        "feature_to_node": args.feature_to_node,
        "only_feature_node": args.only_feature_node,
        "node_ratio": args.node_ratio,
        "edge_ratio": args.edge_ratio,
        "num_selected_nodes": coverage_stats[0],  # - selected node 數量 
        "node_ratio_ori": coverage_stats[1],  # - selected node 在原圖的比例 
        "node_ratio_current": coverage_stats[2],  # - selected node 在現在圖的比例 
        "node_2hop_ratio_ori": coverage_stats[3], # - selected node 的 2hop 在原圖的比例 (original node)
        "node_2hop_ratio_current": coverage_stats[4], # - selected node 的 2hop 在現在圖的比例 (original node + feature node)
        "edge_2hop_ratio_ori": coverage_stats[5],  # - selected node 在 2hop edge 的原圖的比例 (node-node)
        "edge_2hop_ratio_current": coverage_stats[6], # - selected node 在 2hop edge 的現在的圖的比例 (node-node + feature-node)
        "selected_nodes": selected_nodes,  # - 選擇的節點
    }

    save_path = os.path.join(save_dir, f"split{args.split_id}_{args.dataset}.csv")

    # 如果檔案存在，append，否則新建
    file_exists = os.path.exists(save_path)

    df = pd.DataFrame([record])

    df.to_csv(save_path, mode='a', header=not file_exists, index=False)

    print(f"[Coverage log] Saved coverage stats to {save_path}")
