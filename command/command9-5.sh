#!/bin/bash
set -e
set -x 

# 2. Test 10 times
# ------------------------------------------------------------
## Random remove edge
python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_baseline_edge --split_id 0  --trial_start 0 --trial_end 1    

## GNNExplainer remove edge
python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_GNNExplainer_edge --split_id 0  --trial_start 0 --trial_end 5    

python test_remaining_main.py --dataset Cora --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset CiteSeer --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset Amazon --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset FacebookPagePage --normalize --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5
python test_remaining_main.py --dataset GitHub --normalize --model GCN2  --run_mode stage3_GNNExplainer_structure --split_id 0  --trial_start 0 --trial_end 5    

# 
wait