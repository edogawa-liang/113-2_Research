# 尋找核心子圖
- 找到最重要的子圖(不超過10%)，移除核心子圖後，每個 user 的 label 都預測不好 
- 應用
	1. 為保護隱私，應將關鍵子圖移除 
	2. 廣告公司想為 user 貼標籤，需注意要取得這個子圖，鼓勵 user 要提供這個欄位
	3. 社群網路爬蟲，如果可以先知道哪些 user 的關係是重要的，就可以往這個方向做 sample，避免 sample 到不重要的邊 
- 種類
	- 核心子圖包含整個節點
	- 核心子圖只保留重要的節點特徵


## Method
### 1. 隨機挑選子圖 (Baseline)
- 挑選 10% 數量的邊(及其節點對) 作為核心子圖。 (邊僅從 "節點對都屬於訓練集" 的邊中挑選)

### 2. 解釋子圖作為核心子圖
- 解釋方法: GNNExplainer, PGExplainer…
- 將選定的 m 個特徵分別訓練GNN模型 ，挑選 n 個節點，生成 m*n 解釋子圖(GNNExplainer, PGExplainer…)，再找出重疊率高的交集子圖
- m 個特徵
	- 使用 PCA 找重要欄位: 取前 k 個 PCA 主成分，並在每個主成分內找出 loading 最大的 r 個特徵 (default 不超過6 (前k=3個主成份, 前r=2個特徵))
- n 個解釋節點
	- 隨機挑選原圖 (訓練集) 1% 的節點. (整個資料集抽?)
	- high_centrality
    
### 3. 聰明的挑選節點的和邊

## Dataset
1. Github
2. Facebook
3. Cora
4. Citeseer
5. Pubmed

## How to run
See `command.txt`
1. Train basic GNN model (including Original and select feature for multiple GNN Explaination)
	```bash 
	python training_main.py
	```
2. Remove the core subgraph (random, explainer) and train GNN model
	```bash 
	python train_remaining_main.py
	```
3. Generate explanation subgraph
	```bash 
	python stage2_expsubg.py
	```	

## Saved folder
- original\: Node classification for original graph
- stage1\: Node regression for explainer, used for subgraph selection 
- baselineResult\: Node classification for baseline approach (random, Explainer)
- stage2_expsubg\: explanation subgraph for selected node (base on stage1 models)
- remove_from_GNNExplainer\: Node classification for remove subgraph from GNNExplainer