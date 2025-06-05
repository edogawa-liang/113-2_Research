#!/bin/bash
set -e
set -x 

# 計算重要度
# 1. 原圖
python tools/cal_node_imp.py --dataset Cora 
python tools/cal_node_imp.py --dataset CiteSeer
python tools/cal_node_imp.py --dataset Amazon

# 2. feature to node (包含 node-node, node-feature)
python tools/cal_node_imp.py --dataset Cora --feature_to_node
python tools/cal_node_imp.py --dataset CiteSeer --feature_to_node
python tools/cal_node_imp.py --dataset Amazon --feature_to_node

# 3. feature to node (只有 node-feature)
python tools/cal_node_imp.py --dataset Cora --feature_to_node --only_feature_node
python tools/cal_node_imp.py --dataset CiteSeer --feature_to_node --only_feature_node
python tools/cal_node_imp.py --dataset Amazon --feature_to_node --only_feature_node


# 每個節點都生成解釋子圖
# 1. 原圖 (包含特徵)

# 2. 只有結構

# 3. Feature to Node (含節點邊 & 特徵邊)

# 4. Feature to Node (只有特徵邊)


wait