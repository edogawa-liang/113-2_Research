import argparse
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# python stage2_node_share.py --stage_dir stage2_tree_edge_0.3 --explainer GNNExplainer --dataset Actor
# python stage2_node_share.py --stage_dir stage2_y_edge_0.3 --explainer GNNExplainer --dataset Actor

def parse_args():
    parser = argparse.ArgumentParser(description="Organize and update node selection records.")
    parser.add_argument("--saved_dir", type=str, default="saved", help="Top-level saved directory")
    parser.add_argument("--stage_dir", type=str, required=True, help="Stage folder, e.g. stage2_y_edge_0.3")
    parser.add_argument("--explainer", type=str, required=True, help="Explainer name, e.g. GNNExplainer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Actor")
    parser.add_argument("--log_file", type=str, default="update_log.txt", help="Log file name")
    return parser.parse_args()

class NodeSelectionManager:
    def __init__(self, saved_dir, stage_dir, explainer, dataset, log_file):
        self.root_path = Path(saved_dir) / stage_dir / explainer / dataset
        self.explainer = explainer
        self.dataset = dataset
        self.log_file = self.root_path / log_file

    def log(self, message):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} {message}\n")
        print(f"{timestamp} {message}")

    def _is_method_folder(self, folder: Path) -> bool:
        # 如果底下有 share_nodes 資料夾 → 這是 model，不是 method
        return not (folder / "share_nodes").exists()

    def run(self):
        self.method_folders = [f for f in self.root_path.iterdir() if f.is_dir() and self._is_method_folder(f)]
        model_node_map = {}

        for method_folder in self.method_folders:
            method = method_folder.name
            self.log(f"Processing method: {method}")

            for model_folder in method_folder.iterdir():
                if not model_folder.is_dir():
                    continue
                model = model_folder.name
                for node_file in model_folder.glob("node_*.npz"):
                    try:
                        node_id = int(node_file.stem.split("_")[1])
                        model_node_map.setdefault(model, []).append({
                            "Node": node_id,
                            "Method": method,
                            "Path": node_file
                        })
                    except:
                        continue

        for model, node_records in model_node_map.items():
            model_dir = self.root_path / model
            share_dir = model_dir / "share_nodes"
            share_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(node_records)
            all_nodes = sorted(df["Node"].unique())
            all_methods = sorted(df["Method"].unique())

            node_df = pd.DataFrame(index=all_nodes)
            for method in all_methods:
                selected_nodes = set(df[df["Method"] == method]["Node"])
                node_df[method] = node_df.index.map(lambda x: 1 if x in selected_nodes else 0)

            node_df.insert(0, "Model", model)
            node_df.insert(0, "Dataset", self.dataset)
            node_df.insert(0, "Explainer", self.explainer)

            self._merge_or_create_node_csv(node_df, model_dir)

            for _, row in df.iterrows():
                target_file = share_dir / f"node_{row['Node']}.npz" # 新的 node 才會將解釋放進shared，因為相同node的解釋是一樣的
                if not target_file.exists():
                    shutil.copy(row["Path"], target_file)
                    self.log(f"Copied node {row['Node']} to {target_file}")

        # 移除處理過的 method 資料夾
        self.remove_method_folders()

    def _merge_or_create_node_csv(self, new_df, model_dir):
        csv_path = model_dir / "node_record.csv"
        if csv_path.exists():
            old_df = pd.read_csv(csv_path, index_col="Node")
            merged_index = sorted(set(old_df.index).union(new_df.index))
            info_cols = ["Explainer", "Dataset", "Model"]
            all_methods = sorted(set(old_df.columns).union(new_df.columns) - set(info_cols))
            columns = info_cols + all_methods

            merged_df = pd.DataFrame(index=merged_index, columns=columns).fillna(0)

            for col in info_cols:
                merged_df[col] = new_df[col].iloc[0] if col in new_df.columns else old_df[col].iloc[0]

            for col in all_methods:
                if col in old_df.columns:
                    merged_df.loc[old_df.index, col] = old_df[col]
                if col in new_df.columns:
                    merged_df.loc[new_df.index, col] = new_df[col]

            merged_df.to_csv(csv_path, index_label="Node")
        else:
            new_df.to_csv(csv_path, index_label="Node")
            self.log(f"Created new node_record.csv in {model_dir}")

    def remove_method_folders(self):
        for method_folder in self.method_folders:
            try:
                shutil.rmtree(method_folder)
                self.log(f"Removed method folder: {method_folder}")
            except Exception as e:
                self.log(f"Failed to remove method folder {method_folder}: {e}")

def main():
    args = parse_args()
    manager = NodeSelectionManager(
        saved_dir=args.saved_dir,
        stage_dir=args.stage_dir,
        explainer=args.explainer,
        dataset=args.dataset,
        log_file=args.log_file
    )
    manager.run()

if __name__ == "__main__":
    main()
