# 尋找核心子圖
- 找到最重要的子圖(不超過10%)，移除核心子圖後，每個 user 的 label 都預測不好 
- 應用
	1. 為保護隱私，應將關鍵子圖移除 
	2. 廣告公司想為 user 貼標籤，需注意要取得這個子圖，鼓勵 user 要提供這個欄位
	3. 社群網路爬蟲，如果可以先知道哪些 user 的關係是重要的，就可以往這個方向做 sample，避免 sample 到不重要的邊 

## Method
### 1. 隨機挑選子圖 (Baseline)

### 2. 隨機挑選解釋子圖
- 設定 Budget，挑選 Budget 內的節點和特徵生成解釋子圖，將重疊率高的部份作為核心子圖
- 大部分的特徵是連續型，先轉為類別型
	- 使用 GMM → 可考慮不同群的密度，找到最適合的切點
	- 使用 PCA 找重要欄位: 取前 k 個 PCA 主成分，並在每個主成分內找出 loading 最大的 m 個特徵

    
- GNNExplainer, PGExplainer…
### 3. 聰明的挑選節點的和邊

## Dataset
1. Cora
2. Citeseer
3. Pubmed
4. Facebook
5. Github

## How to run
1. Train basic GNN model (including Original and select feature for multiple GNN Explaination)
	```bash 
	python training_main.py
	```
2. Remove the core subgraph and train GNN model
	```bash 
	python train_remaining_main.py
	```
3. Generate explanation subgraph
	```bash 
	python stage2_expsubg.py
	```	

## Saved
original\: Node classification for original graph
stage1\: Node regression for explainer, used for subgraph selection 
baselineResult\: Node classification for baseline approach (random, Explainer)