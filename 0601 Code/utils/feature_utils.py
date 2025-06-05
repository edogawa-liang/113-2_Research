import torch
from typing import List, Tuple
from collections import Counter


def remove_top_common_features(x: torch.Tensor, selected_feat_ids: List[int], fraction_feat: float) -> Tuple[torch.Tensor, List[int]]:
    """
    統計 selected_feat_ids 中出現最多的 feature，移除前 k 個欄位。
    """
    if not selected_feat_ids:
        print("No selected feature IDs provided; returning original features.")
        return x, []

    num_features = x.shape[1]
    k = int(num_features * fraction_feat)

    counter = Counter(selected_feat_ids)
    most_common_feats = [feat_id for feat_id, _ in counter.most_common(k)]

    print(f"Removing top {k} most commonly selected features")#: {most_common_feats}")
    x_new = x.clone()
    x_new[:, most_common_feats] = 0.0
    return x_new, most_common_feats


def remove_all_zero_features(x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """
    移除所有特徵全為 0 的欄位，並記錄被移除的欄位 index。
    """
    feature_sums = x.sum(dim=0)
    zero_feature_cols = (feature_sums == 0).nonzero(as_tuple=True)[0].tolist()
    print(f"Removed {len(zero_feature_cols)} all-zero feature columns")#: {zero_feature_cols}")
    x_new = x[:, feature_sums != 0]
    return x_new, zero_feature_cols

