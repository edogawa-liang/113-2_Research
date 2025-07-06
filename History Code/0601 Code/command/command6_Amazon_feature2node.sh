#!/bin/bash
set -e
set -x 


# Start Stage 3 (先固定移除20%邊)
# remove same feature
# fraction_feat=0.1
# train remove randomwalk
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note random_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat True --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note top_pagerank_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat True --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note high_betweenness_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat True --feature_to_node 
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note stratified_by_degree_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat True --feature_to_node  

# remove different feature
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note random_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat False --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note top_pagerank_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat False --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note high_betweenness_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat False --feature_to_node 
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note stratified_by_degree_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.1 --same_feat False --feature_to_node  


#---
# Start Stage 3 (先固定移除20%邊)
# fraction_feat=0.2
# train remove randomwalk
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note random_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat True --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note top_pagerank_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat True --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note high_betweenness_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat True --feature_to_node 
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note stratified_by_degree_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat True --feature_to_node 

# remove different feature
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note random_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat False --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note top_pagerank_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat False --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note high_betweenness_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat False --feature_to_node 
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note stratified_by_degree_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.2  --same_feat False --feature_to_node 


#---
# Start Stage 3 (先固定移除20%邊)
# fraction_feat=0.3
# train remove randomwalk (不跑 Amazon)
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note random_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat True --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note top_pagerank_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat True --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note high_betweenness_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat True --feature_to_node  
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note stratified_by_degree_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat True --feature_to_node  

# remove different feature
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note random_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose random --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat False --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note top_pagerank_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose top_pagerank --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat False --feature_to_node
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note high_betweenness_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose high_betweenness --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat False --feature_to_node  
python train_remaining_main.py --dataset Amazon --model GCN2 --epochs 300 --lr 0.01 --run_mode remove_from_RandomWalk_samefeat --note stratified_by_degree_RandomWalk_samefeat --selector_type random_walk --walk_length 50  --fraction 0.2  --node_choose stratified_by_degree --node_ratio auto --edge_ratio 0.3 --fraction_feat 0.3  --same_feat False --feature_to_node  

