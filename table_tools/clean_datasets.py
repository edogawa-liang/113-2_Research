import os
import shutil
import pandas as pd

src_dir = "datasets"
dst_dir = "table_datasets"
summary = []

os.makedirs(dst_dir, exist_ok=True)

for root, dirs, files in os.walk(src_dir):
    
    if "regression" in root:
        continue

    lower_root = root.lower().replace("\\", "/")
    if lower_root.endswith("large_datasets/datasets") or lower_root.endswith("small_datasets/datasets"):
        continue

    for file in files:
        if not file.endswith(".csv"):
            continue
        
        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, src_dir)
        lower_path = relative_path.lower().replace("\\", "/")

        dst_filename = file
        dst_path = os.path.join(dst_dir, dst_filename)

        shutil.copy(file_path, dst_path)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}, error: {e}")
            continue

        if "target" not in df.columns:
            print(f"Warning: 'target' column not found in {file_path}, skipping.")
            continue

        num_samples = df.shape[0]
        feature_cols = [col for col in df.columns if col != "target"]
        num_features = len(feature_cols)
        num_continuous = sum(1 for col in feature_cols if col.startswith("N_"))
        num_categorical = sum(1 for col in feature_cols if col.startswith("C_"))
        num_classes = df["target"].nunique()

        if "large_datasets" in lower_path:
            dataset_size = "large"
        elif "small_datasets" in lower_path:
            dataset_size = "small"
        else:
            dataset_size = "unknown"

        if "binclass" in lower_path:
            task_type = "binclass"
        elif "multiclass" in lower_path:
            task_type = "multiclass"
        elif "regression" in lower_path:
            task_type = "regression"
        else:
            task_type = "unknown"

        # feature_type 多條件組合，固定順序
        feature_type_list = []
        if "numerical" in lower_path:
            feature_type_list.append("numerical")
        if "categorical" in lower_path:
            feature_type_list.append("categorical")
        if "balance" in lower_path:
            feature_type_list.append("balance")

        feature_type = ", ".join(feature_type_list) if feature_type_list else "unknown"

        summary.append({
            "filename": dst_filename,
            "num_samples": num_samples,
            "dataset_size": dataset_size,
            "num_classes": num_classes,
            "num_features": num_features,
            "num_continuous": num_continuous,
            "num_categorical": num_categorical,
            "task_type": task_type,
            "feature_type": feature_type,
            "source_path": relative_path,
        })

# 存 summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(dst_dir, "summary.csv"), index=False)

print(f"Finished! {len(summary)} files moved to {dst_dir} with summary.")
