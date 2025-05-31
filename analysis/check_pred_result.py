# 1. 舊模型與新模型預測相同 → 都是錯的
# 2. 舊模型與新模型預測相同 → 都是對的
# 3. 舊模型與新模型預測不同 → 錯的改成錯的
# 4. 舊模型與新模型預測不同 → 錯的改成對的
# 5. 舊模型與新模型預測不同 → 對的改成錯的  

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import pandas as pd
from data.dataset_loader import GraphDatasetLoader
from models.basic_GCN import GCN2Classifier
from utils.device import DEVICE
from subgraph_selector.remaining_graph import RemainingGraphConstructor
from subgraph_selector.explainer_selector import ExplainerEdgeSelector


def load_data(dataset):
    loader = GraphDatasetLoader()
    data, num_features, num_classes, _, _ = loader.load_dataset(dataset)
    return data.to(DEVICE), num_features, num_classes

def load_model(run_mode, dataset, model_name, num_features, num_classes):
    config_path = os.path.join("saved", run_mode, "model", dataset, f"{model_name}_config.pth")
    model_path = os.path.join("saved", run_mode, "model", dataset, f"{model_name}.pth")
    config = torch.load(config_path)

    import inspect
    allowed_params = inspect.signature(GCN2Classifier.__init__).parameters
    allowed_keys = set(allowed_params.keys()) - {"self"}

    model_config = {
        key: config.get(key, num_features if key == "in_channels" else num_classes if key == "out_channels" else 64)
        for key in allowed_keys
    }

    model = GCN2Classifier(**model_config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def analyze_prediction_cases(orig_data, remain_data, original_model, new_model, dataset, new_model_name):
    with torch.no_grad():
        orig_out = original_model(orig_data.x, orig_data.edge_index)
        new_out = new_model(remain_data.x, remain_data.edge_index)

    orig_preds = orig_out.argmax(dim=1)
    new_preds = new_out.argmax(dim=1)
    true_labels = orig_data.y

    test_idx = orig_data.test_mask.nonzero(as_tuple=True)[0]

    case1 = case2 = case3 = case4 = case5 = 0

    for idx in test_idx:
        o_pred = orig_preds[idx].item()
        n_pred = new_preds[idx].item()
        label = true_labels[idx].item()

        if o_pred == n_pred:
            if o_pred == label:
                case2 += 1  # same, both correct
            else:
                case1 += 1  # same, both wrong
        else:
            if o_pred != label and n_pred != label:
                case3 += 1  # different, wrong to wrong
            elif o_pred != label and n_pred == label:
                case4 += 1  # wrong to correct
            elif o_pred == label and n_pred != label:
                case5 += 1  # correct to wrong

    print("Case 1 (same, both wrong):", case1)
    print("Case 2 (same, both correct):", case2)
    print("Case 3 (different, wrong → wrong):", case3)
    print("Case 4 (different, wrong → correct):", case4)
    print("Case 5 (different, correct → wrong):", case5)

    os.makedirs("saved/analysis/check_pred_result", exist_ok=True)
    df = pd.DataFrame([{
        "Case1_both_wrong": case1,
        "Case2_both_correct": case2,
        "Case3_wrong_to_wrong": case3,
        "Case4_wrong_to_correct": case4,
        "Case5_correct_to_wrong": case5
    }])
    save_path = os.path.join("saved/analysis/check_pred_result", f"{dataset}_{new_model_name}_case_count.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved case count to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--original_run_mode", type=str, required=True)
    parser.add_argument("--new_run_mode", type=str, required=True)
    parser.add_argument("--original_model", type=str, required=True)
    parser.add_argument("--new_model", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    
    # 得到解釋子圖資料集需要
    parser.add_argument("--explainer_name", type=str, required=True)
    parser.add_argument("--node_choose", type=str, default="top_pagerank")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--fraction_feat", type=float, default=0.0)
    parser.add_argument("--feature_to_node", action="store_true")
    parser.add_argument("--same_feat", action="store_true")
    parser.add_argument("--feature_type", type=str, default="categorical")
    args = parser.parse_args()

    orig_data, num_features, num_classes = load_data(args.dataset)
    orig_model = load_model(args.original_run_mode, args.dataset, args.original_model, num_features, num_classes)
    new_model = load_model(args.new_run_mode, args.dataset, args.new_model, num_features, num_classes)

    selector = ExplainerEdgeSelector(
        base_dir=args.base_dir,
        explainer_name=args.explainer_name,
        dataset_name=args.dataset,
        node_choose=args.node_choose,
        top_k_percent=args.fraction,
        feature_type=args.feature_type,
        device=DEVICE,
        top_k_percent_feat=args.fraction_feat,
        use_feature_to_node=args.feature_to_node
    )
    selector.load_data()
    selected_edges, selected_feat_ids = selector.select_edges(
        num_ori_edges=orig_data.num_edges,
        num_ori_nodes=orig_data.num_nodes,
        ori_num_features=orig_data.x.size(1),
        return_feat_ids=True
    )
    selected_feat = None
    if not args.feature_to_node and args.fraction_feat > 0:
        selected_feat = selector.select_node_features(
            num_ori_nodes=orig_data.num_nodes,
            same_feat=args.same_feat
        )

    remaining_graph_constructor = RemainingGraphConstructor(orig_data, selected_edges, selected_feat_mask=selected_feat, device=DEVICE)
    remain_data = remaining_graph_constructor.get_remaining_graph()

    analyze_prediction_cases(orig_data, remain_data, orig_model, new_model, args.dataset, args.new_model)

# 執行方式：
# python analysis/check_pred_result.py --dataset Cora --original_run_mode original --original_model 1_GCN2Classifier --new_run_mode remove_from_GNNExplainer --new_model 1_GCN2Classifier --base_dir saved/stage2_y_edge_0.3 --explainer_name GNNExplainer --node_choose random --fraction 0.1 --fraction_feat 0.0 
