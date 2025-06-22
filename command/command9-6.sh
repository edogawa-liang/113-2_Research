#!/bin/bash
set -e
set -x 



# 3. Summary Test 10 times
# ------------------------------------------------------------
## Original GNN
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset Cora --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset Amazon --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1 --split_id 0 --dataset GitHub --trial_start 0 --trial_end 0  

## GNN Only Look at Structure
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset Cora --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset Amazon --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_structure --split_id 0 --dataset GitHub --trial_start 0 --trial_end 0

## MLP
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset Cora --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset CiteSeer --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset Amazon --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset FacebookPagePage --trial_start 0 --trial_end 0
python tools/result_summary.py --run_mode stage1_mlp --split_id 0 --dataset GitHub --trial_start 0 --trial_end 0 


wait