import os
import argparse

def count_all_npz(run_mode):
    base_path = f"saved/{run_mode}/split_0/CFExplainer"

    if not os.path.exists(base_path):
        print(f"[Error] 資料夾不存在：{base_path}")
        return

    total_count = 0
    print(f"=== 統計 {base_path} 下所有資料集 ===")

    for dataset_name in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        dataset_count = 0
        for trial_folder in os.listdir(dataset_path):
            trial_path = os.path.join(dataset_path, trial_folder)
            if not os.path.isdir(trial_path):
                continue

            npz_count = len([f for f in os.listdir(trial_path) if f.endswith('.npz')])
            dataset_count += npz_count

        print(f"{dataset_name}：{dataset_count} 個 .npz 檔案")
        total_count += dataset_count

    print(f"=== 總計：{total_count} 個 .npz 檔案 ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, required=True)
    args = parser.parse_args()

    count_all_npz(args.run_mode)


# python tools/check_cf_node.py --run_mode stage2_feature2node_nn_fn
# python tools/check_cf_node.py --run_mode stage2_feature2node_fn




