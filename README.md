# 尋找核心子圖
找到最重要的子圖，移除核心子圖後，每個 user 的 label 都預測不好 
- 情境
	1. 核心子圖包含整個節點
	2. 核心子圖的節點只保留重要特徵 ，每位 user 有相同的重要特徵
	3. 核心子圖的節點只保留重要的特徵 ，每位 user 的重要特徵不同

- 攻擊方角度 (y不知) vs 防禦方角度 (y已知)
- 針對整體做保護 (訓練集挑選節點) vs 保護特定族群 (測試集節點)

## Method
### 1. 隨機挑選子圖 (Baseline)

### 2. Random Subgraph
### 3. RandomWalk

### 4. Explainable Subgraph
- GNNExplainer, CFExplainer…

---

### 5. Feature-to-node
將每個特徵都作為一個新節點，變為一個新圖有 (節點數+特徵數) 個節點，節點特徵為一維的1
- 類別型特徵：特徵存在則將邊連過去，形成特徵邊 (node-feature)
- 連續型特徵：將 feature value 經過 L1 normalization，作為 edge weight，形成節點邊 (node-node)

### 6. 節點選擇策略
- 方法：random、PageRank、Betweenness、stratified by degree
- 數量：$\frac{E \cdot r}{d^2}$ (邊數量 $E$  × 目標覆蓋的比例 $r$ ÷ 節點平均degree^2) (因設定 2-hop)


## Dataset
1. Cora
2. Citeseer
3. Amazon

## How to run
### Main Code
1. Stage 1 : Train basic GNN model 
	```bash 
	python training_main.py
	```
2. Stage 2 : Generate explanation subgraph
	```bash 
	python stage2_expsubg.py
	```	
3. Stage 2 : save the space of explanation subgraph (補充)
	```bash 
	python tools/stage2_node_share.py
	```
4. Stage 3 :Remove the core subgraph (random, explainer) and train GNN model
	```bash 
	python train_remaining_main.py
	```

---
### Tools

5. 經過模型有改變預測與沒改變預測的節點，並且計算這些節點與測試節點的距離
	```bash 
	python  tools/check_result.py
	```
6. 檢查 CFGNNExplainer 成功生成解釋的節點數量
	```bash 
	python  tools/check_cf_summary2.py
	```
7. 畫圖: 被選中節點到測試節點的距離。
	```bash 
	python  tools/distance_from_selected_to_test.py
	```
8. 畫圖: 測試節點到被選中的距離。
	```bash 
	python  tools/distance_from_test_to_selected.py
	```
9. 解釋子圖的節點與邊重要度分布
	```bash 
	python  tools/plot_stage2.py
	```


