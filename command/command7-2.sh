#!/bin/bash
set -e
set -x 

# Stage 2: 生成解釋子圖 (GNNExplainer)
# 1. 原圖
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 0 --repeat_end 0
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 0 --repeat_end 0
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 0 --repeat_end 0
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 0 --repeat_end 0
python stage2_expsubg.py --dataset GitHub --normalize --explainer_type GNNExplainer --run_mode stage2 --stage1_path saved/original --repeat_start 0 --repeat_end 0

# 2. 原圖 (只有結構)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 0 --repeat_end 0
python stage2_expsubg.py --dataset CiteSeer --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 0 --repeat_end 0
python stage2_expsubg.py --dataset Amazon --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 0 --repeat_end 0 
python stage2_expsubg.py --dataset GitHub --explainer_type GNNExplainer --run_mode stage2_structure --stage1_path saved/original_structure --only_structure --structure_mode random+imp --repeat_start 0 --repeat_end 0 

wait