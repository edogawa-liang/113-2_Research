# 流程
# 1. 將表格式資料用 Logistics, XBG 訓練，得到移除前的基準值
# 2. 將表格式資料轉換成圖結構，並用 GNN 訓練，得到生成解釋的 base model
# 3. 用 GNNExplainer 得到解釋子圖 (可以直接用 stage2_expsubg.py)
# 4. 移除核心子圖，並轉換成表格結構，重新用 Logistics, XBG 訓練

# 比較對象
# 1. 原始表格資料
# 2. 隨機移除 feature 的表格資料
# 3. 用 ML feature importance 移除 feature 的表格資料
