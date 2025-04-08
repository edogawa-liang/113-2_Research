import argparse
import pandas as pd
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Organize and update node selection records.")
    parser.add_argument("--saved_dir", type=str, default="saved", help="Top-level saved directory")
    parser.add_argument("--stage_dir", type=str, required=True, help="Stage folder, e.g. stage2_y_edge_0.3")
    parser.add_argument("--explainer", type=str, required=True, help="Explainer name, e.g. GNNExplainer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Actor")
    return parser.parse_args()

class NodeSelectionManager:
    def __init__(self, saved_dir, stage_dir, explainer, dataset):
        self.root_path = Path(saved_dir) / stage_dir / explainer / dataset
        self.explainer = explainer
        self.dataset = dataset

    def run(self):
        method_folders = [f for f in self.root_path.iterdir() if f.is_dir()]
        model_node_map = {}

        for method_folder in method_folders:
            method = method_folder.name
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

            # 加入 explainer/dataset/model info
            node_df.insert(0, "Model", model)
            node_df.insert(0, "Dataset", self.dataset)
            node_df.insert(0, "Explainer", self.explainer)

            self._merge_or_create_node_csv(node_df, model_dir)

            for _, row in df.iterrows():
                target_file = share_dir / f"node_{row['Node']}.npz"
                if not target_file.exists():
                    shutil.copy(row["Path"], target_file)

    def _merge_or_create_node_csv(self, new_df, model_dir):
        csv_path = model_dir / "node_record.csv"
        if csv_path.exists():
            old_df = pd.read_csv(csv_path, index_col="Node")

            # 合併所有節點
            merged_index = sorted(set(old_df.index).union(new_df.index))

            # 合併所有欄位
            info_cols = ["Explainer", "Dataset", "Model"]
            all_methods = sorted(set(old_df.columns).union(new_df.columns) - set(info_cols))
            columns = info_cols + all_methods

            merged_df = pd.DataFrame(index=merged_index, columns=columns).fillna(0)

            # 填寫 Explainer / Dataset / Model（用新資料的值）
            for col in info_cols:
                merged_df[col] = new_df[col].iloc[0] if col in new_df.columns else old_df[col].iloc[0]

            # 合併資料（方法 one-hot）
            for col in all_methods:
                if col in old_df.columns:
                    merged_df.loc[old_df.index, col] = old_df[col]
                if col in new_df.columns:
                    merged_df.loc[new_df.index, col] = new_df[col]

            merged_df.to_csv(csv_path, index_label="Node")

        else:
            new_df.to_csv(csv_path, index_label="Node")

def main():
    args = parse_args()
    manager = NodeSelectionManager(
        saved_dir=args.saved_dir,
        stage_dir=args.stage_dir,
        explainer=args.explainer,
        dataset=args.dataset
    )
    manager.run()

if __name__ == "__main__":
    main()
