# 計算 10 次 repeat 的結果 & 標準差
# analyze_result.py
import argparse
import pandas as pd
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze result ACC per group")
    parser.add_argument("--run_mode", type=str, required=True, help="Run mode (folder name)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 搜尋檔案
    result_files = glob.glob(f"113-2_Research/saved/{args.run_mode}/result/result_*.xlsx")
    print(f"Found {len(result_files)} result files.")

    for file_path in result_files:
        print(f"\n=== Processing file: {file_path} ===")

        # 讀取所有 sheets
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        print(f"Found sheets: {sheet_names}")

        for sheet_name in sheet_names:
            print(f"\n-- Processing sheet: {sheet_name} --")
            df = xls.parse(sheet_name)

            # 確認 repeat_id 存在
            if 'repeat_id' not in df.columns:
                print(f"Sheet {sheet_name} missing repeat_id column, skipping.")
                continue

            # Group by repeat_id → 你希望是「0-9 一組」，但可能有很多組 → 要分組
            # 方法：假設一組 repeat_id=0~9 出現一次，下一組再出現一遍0~9
            # 我用 group_counter 記錄 → 這樣可以拆多組
            repeat_id_counts = df['repeat_id'].value_counts().to_dict()

            # 先找到 0 出現幾次 → 就是幾組
            num_groups = repeat_id_counts.get(0, 0)

            print(f"Detected {num_groups} group(s) of repeat_id 0-9.")

            for group_idx in range(num_groups):
                # 篩一組 → repeat_id = 0~9，每個取第 group_idx 筆
                group_rows = []
                for rid in range(10):
                    rows_rid = df[df['repeat_id'] == rid]
                    if len(rows_rid) <= group_idx:
                        print(f"Warning: repeat_id={rid} does not have enough rows for group {group_idx}. Skipping this group.")
                        break
                    row = rows_rid.iloc[group_idx]
                    group_rows.append(row)
                else:
                    # 如果完整收集到一組 0~9
                    group_df = pd.DataFrame(group_rows)

                    acc_mean = group_df['ACC'].mean()
                    acc_std = group_df['ACC'].std()

                    print(f"\n[Sheet: {sheet_name}] [Group {group_idx}] ACC mean: {acc_mean:.4f}, std: {acc_std:.4f}")

                    # 排除要排除的欄位
                    exclude_columns = ["Model", "LR", "Epochs", "Best Epoch", "Loss", "ACC", "Auc", "Precision", "Recall", "F1", "CM", "Threshold"]

                    remaining_columns = [col for col in df.columns if col not in exclude_columns]

                    if len(remaining_columns) > 0:
                        print("Other columns:")
                        # 印 group 內第一筆 row 的其他欄位
                        print(group_df[remaining_columns].iloc[0].to_string())
                    else:
                        print("(No other columns)")

