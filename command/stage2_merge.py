import argparse
import pandas as pd
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Create or update node selection record from subfolders.")
    parser.add_argument("--saved_dir", type=str, default="saved" help="Stage folder name, e.g. stage2_y_edge_0.3")
    parser.add_argument("--stage_dir", type=str, required=True, help="Stage folder name, e.g. stage2_y_edge_0.3")
    parser.add_argument("--explainer", type=str, required=True, help="Explainer name, e.g. GNNExplainer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Actor")
    parser.add_argument("--multi_model", action="store_true", help="Set this flag to handle multiple model folders")
    return parser.parse_args()

class NodeSelectionManager:
    def __init__(self, stage_dir, explainer, dataset, multi_model):
        self.base_path = os.path.join("saved", stage_dir, explainer, dataset, ".txt")
        self.explainer = explainer
        self.dataset = dataset
        self.multi_model = multi_model

    def run(self):
        if self.multi_model:
            self._handle_multi_model()
        else:
            self._handle_single_model()

    def _get_method_folders(self):
        return [f for f in self.base_path.iterdir() if f.is_dir() and f.name != "all_nodes_combined"]

    def _collect_nodes_by_model(self, method_folders):
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
        return model_node_map

    def _collect_selected_nodes(self, method_folders):
        all_nodes = set()
        method_node_dict = {}
        for method_folder in method_folders:
            method = method_folder.name
            method_node_dict[method] = set()
            for file in method_folder.glob("node_*.npz"):
                try:
                    node_id = int(file.stem.split("_")[1])
                    all_nodes.add(node_id)
                    method_node_dict[method].add(node_id)
                except:
                    continue
        return sorted(all_nodes), method_node_dict

    def _build_dataframe(self, model_name, node_records):
        df = pd.DataFrame(node_records)
        df["Explainer"] = self.explainer
        df["Dataset"] = self.dataset
        df["Model"] = model_name

        pivot = pd.pivot_table(df, index="Node", columns="Method", aggfunc=lambda x: 1, fill_value=0)
        pivot.columns = pivot.columns.droplevel(0)
        result = pivot.reset_index()
        result.insert(0, "Model", model_name)
        result.insert(0, "Dataset", self.dataset)
        result.insert(0, "Explainer", self.explainer)
        return result, df

    def _move_shared_nodes(self, df_file_rows, target_dir):
        share_dir = target_dir / "share_nodes"
        share_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df_file_rows.iterrows():
            target_path = share_dir / f"node_{row['Node']}.npz"
            if not target_path.exists():
                shutil.copy(row["Path"], target_path)

    def _save_node_csv(self, df, target_dir):
        df.to_csv(target_dir / "node_record.csv", index=False)

    def _handle_multi_model(self):
        method_folders = self._get_method_folders()
        model_node_map = self._collect_nodes_by_model(method_folders)

        for model, node_records in model_node_map.items():
            model_dir = self.base_path / model
            model_dir.mkdir(parents=True, exist_ok=True)

            df_summary, df_files = self._build_dataframe(model, node_records)
            self._move_shared_nodes(df_files, model_dir)
            self._save_node_csv(df_summary, model_dir)

    def _handle_single_model(self):
        method_folders = self._get_method_folders()
        methods = [f.name for f in method_folders]
        all_nodes, method_node_dict = self._collect_selected_nodes(method_folders)

        df = pd.DataFrame(index=all_nodes)
        df["Explainer"] = self.explainer
        df["Dataset"] = self.dataset
        for method in methods:
            df[method] = df.index.map(lambda x: 1 if x in method_node_dict[method] else 0)

        output_dir = self.base_path / "all_nodes_combined"
        output_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            for file in (self.base_path / method).glob("node_*.npz"):
                target = output_dir / file.name
                if not target.exists():
                    file.rename(target)

        df.to_csv(self.base_path / "node_record.csv", index_label="Node")



if __name__ == "__main__":
    args = parse_args()
    manager = NodeSelectionManager(
        stage_dir=args.stage_dir,
        explainer=args.explainer,
        dataset=args.dataset,
        multi_model=args.multi_model
    )
    manager.run()
