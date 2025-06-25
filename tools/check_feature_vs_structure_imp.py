import torch
import os

print(os.getcwd())

# 檔案路徑
# path = "113-2_Research/saved/stage1_feature2node_nn_fn/split_0/feat2node_graph/Cora/node_feature_vs_structure_imp.pt"

dataset = ["Cora", "CiteSeer","Amazon", "FacebookPagePage", "GitHub"]

for i in dataset:
    file = f"saved/stage1_feature2node_nn_fn/split_0/feat2node_graph/{i}/node_feature_vs_structure_imp.pt"
    if not os.path.exists(file):
        print(f"File {file} does not exist.")
        continue
    data = torch.load(file)

    # 印出內容
    print("Dataset:", i)
    # 統計數量
    num_eq_1 = (data == 1).sum().item()
    num_gt_1 = (data > 1).sum().item()
    num_lt_1 = (data < 1).sum().item()

    print(f"= 1 的數量: {num_eq_1}")
    print(f"> 1 的數量: {num_gt_1}")
    print(f"< 1 的數量: {num_lt_1}")


# python tools/check_feature_vs_structure_imp.py