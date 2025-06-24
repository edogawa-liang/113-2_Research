#!/bin/bash
set -e
set -x 


# =================== 重新生成經過 feature to node 的解釋子圖 =========================
# 注意不要先normalize
# ------------------------------------------------------------
# 2. 生成解釋
## Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset FacebookPagePage--explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_feature2node_fn --stage1_path saved/stage1_feature2node_fn 

# 
wait