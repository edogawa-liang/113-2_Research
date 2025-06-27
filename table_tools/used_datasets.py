import pandas as pd
import os
import random

# 讀入 summary.csv
df = pd.read_csv("table_datasets/summary.csv")

# 新增 used 欄位，預設為 1
df["used"] = 1

# 不使用條件：class > 10, num_feature < 10, num_sample > 25000, num_sample < 200
df.loc[(df["num_classes"] > 10) | (df["num_features"] < 10) | (df["num_samples"] > 25000) | (df["num_samples"] < 200), "used"] = 0

# 新增 used_sample 欄位，預設為 0
df["used_sample"] = 0

# 分組條件
group_folders = ["large", "small"]
feature_types = ["numerical", "categorical", "balance"]

# 隨機抽樣、設置 used_sample
for group in group_folders:
    for feat in feature_types:
        # 篩選符合條件的資料集
        cond = (df["dataset_size"] == group) & (df["feature_type"] == feat) & (df["used"] == 1)
        candidates = df[cond]

        # 抽兩個隨機標記
        if len(candidates) >= 2:
            sampled = candidates.sample(2, random_state=42)
            df.loc[sampled.index, "used_sample"] = 1
        elif len(candidates) > 0:
            # 若不足兩筆，盡量標記現有的
            df.loc[candidates.index, "used_sample"] = 1

# 存回
df.to_csv("table_datasets/summary_with_used.csv", index=False)
print("完成，結果存於 table_datasets/summary_with_used.csv")
