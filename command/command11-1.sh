#!/bin/bash
set -e
set -x 

# ==================== Test 10 times ======================== 

# CFExplainer
# 1. 移除結構 (節點邊)
# command11-1.sh
# python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_CFExplainer_edge --split_id 0  --trial_start 0 --trial_end 1
# python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_CFExplainer_edge --split_id 0  --trial_start 0 --trial_end 1
# python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_CFExplainer_edge --split_id 0  --trial_start 0 --trial_end 1
# python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_CFExplainer_edge --split_id 0  --trial_start 0 --trial_end 1
# python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_CFExplainer_edge --split_id 0  --trial_start 0 --trial_end 1
# # summary
# python tools/result_summary.py --run_mode stage3_CFExplainer_edge --split_id 0 --dataset Cora --trial_start 0 --trial_end 1
# python tools/result_summary.py --run_mode stage3_CFExplainer_edge --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 1
# python tools/result_summary.py --run_mode stage3_CFExplainer_edge --split_id 0 --dataset Amazon --trial_start 0 --trial_end 1
# python tools/result_summary.py --run_mode stage3_CFExplainer_edge --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 1
# python tools/result_summary.py --run_mode stage3_CFExplainer_edge --split_id 0 --dataset GitHub --trial_start 0 --trial_end 1


# ============================================

# GNNExplainer

# 3. 移除特徵 (特徵不同)
# random
python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_random_difffeat --split_id 0  --trial_start 0 --trial_end 0
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_random_difffeat --split_id 0  --trial_start 0 --trial_end 0
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_random_difffeat --split_id 0  --trial_start 0 --trial_end 0
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_random_difffeat --split_id 0  --trial_start 0 --trial_end 0
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_random_difffeat --split_id 0  --trial_start 0 --trial_end 0

# GNNExplainer
python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_GNNExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 8
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_GNNExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 8
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_GNNExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 8
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_GNNExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 8
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_GNNExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 8

# CFExplainer
python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_CFExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_CFExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_CFExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_CFExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 1
# python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_CFExplainer_difffeat --split_id 0  --trial_start 0 --trial_end 1


# random
python tools/result_summary.py --run_mode stage3_random_difffeat --split_id 0 --dataset Cora --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage3_random_difffeat --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage3_random_difffeat --split_id 0 --dataset Amazon --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage3_random_difffeat --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage3_random_difffeat --split_id 0 --dataset GitHub --trial_start 0 --trial_end 0

# GNNExplainer
python tools/result_summary.py --run_mode stage3_GNNExplainer_difffeat --split_id 0 --dataset Cora --trial_start 0 --trial_end 8
python tools/result_summary.py --run_mode stage3_GNNExplainer_difffeat --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 8
python tools/result_summary.py --run_mode stage3_GNNExplainer_difffeat --split_id 0 --dataset Amazon --trial_start 0 --trial_end 8
python tools/result_summary.py --run_mode stage3_GNNExplainer_difffeat --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 8
# python tools/result_summary.py --run_mode stage3_GNNExplainer_difffeat --split_id 0 --dataset GitHub --trial_start 0 --trial_end 8

# CFExplainer
python tools/result_summary.py --run_mode stage3_CFExplainer_difffeat --split_id 0 --dataset Cora --trial_start 0 --trial_end 1
python tools/result_summary.py --run_mode stage3_CFExplainer_difffeat --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 1
python tools/result_summary.py --run_mode stage3_CFExplainer_difffeat --split_id 0 --dataset Amazon --trial_start 0 --trial_end 1
python tools/result_summary.py --run_mode stage3_CFExplainer_difffeat --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 1
# python tools/result_summary.py --run_mode stage3_CFExplainer_difffeat --split_id 0 --dataset GitHub --trial_start 0 --trial_end 1

wait