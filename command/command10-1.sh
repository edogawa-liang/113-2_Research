#!/bin/bash
set -e
set -x 

# CFExplainer

# 1. 移除結構 (節點邊)
### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize  --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.1 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1

### [2] explainer select
python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset CiteSeer --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset FacebookPagePage --normalize  --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1
python train_remaining_main.py --dataset GitHub --normalize --model GCN2 --epochs 300 --lr 0.01 --selector_type explainer --fraction 0.3 --base_dir saved/stage2 --explainer_name CFExplainer --node_choose all_train --run_mode stage3_CFExplainer_edge --note split0_cf_edge --stage1_path saved/stage1


wait