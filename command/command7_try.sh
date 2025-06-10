#!/bin/bash
set -e
set -x 

# =============== try =============== 
# 切分資料集
python data/prepare_split.py --dataset Cora --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1
python data/prepare_split.py --dataset FacebookPagePage --num_repeat 10 --train_ratio 0.2 --val_ratio 0.05 --test_ratio 0.1

# Stage 1: 訓練原模型
# 1. 原圖 (連續型記得+normalize)
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original --note original
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original --note original


# 2. 原圖 (只有結構)
## (1) random32+imp
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure --note random+imp --only_structure --structure_mode random+imp --learn_embedding


## (2) one+imp
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure_one --note one+imp --only_structure --structure_mode one+imp
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_structure_one --note one+imp --only_structure --structure_mode one+imp


# 3. Feature to Node (含節點邊 & 特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_nn_fn --note feature2node_nn_fn  --feature_to_node --structure_mode random+imp --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_nn_fn --note feature2node_nn_fn --feature_to_node --structure_mode random+imp --learn_embedding

# 4. Feature to Node (只有特徵邊)
python training_main.py --dataset Cora --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_fn --note feature2node_fn  --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding
python training_main.py --dataset FacebookPagePage --normalize --model GCN2 --epochs 1 --lr 0.01 --run_mode try_original_feature2node_fn --note feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --learn_embedding


# Stage 2: 生成解釋子圖 (GNNExplainer)

# 1. 原圖
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original --epoch 1
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original  --epoch 1

# 2. 原圖 (只有結構)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original_structure --only_structure --structure_mode random+imp --epoch 1
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_structure --stage1_path saved/try_original_structure --only_structure --structure_mode random+imp --epoch 1

# 3. Feature to Node (含節點邊 & 特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_nn_fn --stage1_path saved/try_original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --epoch 1 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_nn_fn --stage1_path saved/try_original_feature2node_nn_fn --feature_to_node --structure_mode random+imp --epoch 1

# 4. Feature to Node (只有特徵邊)
python stage2_expsubg.py --dataset Cora --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_fn --stage1_path saved/try_original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --epoch 1 
python stage2_expsubg.py --dataset FacebookPagePage --normalize --explainer_type GNNExplainer --run_mode try_stage2_original_feature2node_fn --stage1_path saved/try_original_feature2node_fn --feature_to_node --structure_mode random+imp --only_feature_node --epoch 1 


# Stage 3: 訓練移除解釋子圖後模型




# 表格式資料
wait