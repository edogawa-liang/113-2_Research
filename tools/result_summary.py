import argparse
import pandas as pd
import os

# python tools/result_summary.py --run_mode stage3_GNNExplainer_edge --split_id 0 --dataset Cora --trial_start 0 --trial_end 9

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GNN repeat results.")
    parser.add_argument("--run_mode", type=str, required=True, help="Run mode (folder name)")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--trial_start", type=int, default=0, help="Start trial ID (inclusive)")
    parser.add_argument("--trial_end", type=int, default=0, help="End trial ID (inclusive)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    base_dir = f"saved/{args.run_mode}/split_{args.split_id}/repeat_result/{args.dataset}"

    for trial_id in range(args.trial_start, args.trial_end + 1):
        file_name = f"{trial_id}_GCN2Classifier.csv"
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            file_name = f"{trial_id}_MLPClassifier.csv"
            file_path = os.path.join(base_dir, file_name)

        if not os.path.exists(file_path):
            print(f"[Trial {trial_id}]  File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        if "Acc" not in df.columns or "repeat_id" not in df.columns:
            print(f"[Trial {trial_id}] Missing 'Acc' or 'repeat_id' columns.")
            continue

        df = df[df["repeat_id"].isin(range(10))]

        acc_mean = df["Acc"].mean()
        acc_std = df["Acc"].std()
        print(f"\n[Trial {trial_id}] Acc mean = {acc_mean:.4f}, std = {acc_std:.4f}")

        # === 額外欄位統計 ===
        exclude_columns = ["Model", "LR", "Epochs", "Best Epoch", "Loss", "Acc", "Auc", "Precision", "Recall", "F1", "CM", "Threshold"]
        remaining_columns = [col for col in df.columns if col not in exclude_columns]

        if len(remaining_columns) > 0:
            print("Other columns:")
            nunique_series = df[remaining_columns].nunique()
            varying_columns = nunique_series[nunique_series > 1].index.tolist()
            constant_columns = nunique_series[nunique_series == 1].index.tolist()

            for col in varying_columns:
                unique_values = df[col].unique()
                unique_values_str = ", ".join(map(str, unique_values))
                print(f"  {col} = {unique_values_str}")

            for col in constant_columns:
                value = df[col].iloc[0]
                print(f"  {col} = {value}")
