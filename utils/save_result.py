import os
import pandas as pd
from datetime import datetime

class ExperimentLogger:
    """
    A logger for storing GNN training results in a Excel file.
    """

    def __init__(self, save_dir="saved/results", file_name="gnn_results", move_old=True):
        """
        Initializes the logger.

        :param save_dir: Directory to store the results.
        :param file_name: Custom name for the Excel file (without extension).
        :param move_old: Whether to move old results to history before saving new ones.
        """
        self.save_dir = save_dir
        self.history_dir = os.path.join(self.save_dir, "history")
        self.move_old = move_old
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        self.date_str = datetime.now().strftime("%m%d")  
        self.file_name = f"{file_name}_{self.date_str}.xlsx"
        self.file_path = os.path.join(self.save_dir, self.file_name)

    def _move_old_results(self):
        """Moves old Excel results to the history folder before saving new ones."""
        if self.move_old and os.path.exists(self.file_path):
            new_path = os.path.join(self.history_dir, self.file_name)
            os.rename(self.file_path, new_path)
            print(f"Moved old results to {new_path}")

    def _load_existing_results(self):
        """Loads existing results from the Excel file."""
        if os.path.exists(self.file_path):
            return pd.read_excel(self.file_path, sheet_name=None, engine="openpyxl")  # Load all sheets as a dictionary
        return {}
    

    def get_next_trial_number(self, df):
        """Finds the next available 'trial' number for a given dataset."""
        if "trial" in df.columns and not df.empty:
            return df["trial"].max() + 1  # get the max 'trial' number and add 1
        return 1  
    

    def log_experiment(self, dataset_name, experiment_data):
        """
        Logs an experiment result in the Excel file using a dictionary.
        If the dataset does not exist, a new sheet is created.

        :param dataset_name: Name of the dataset.
        :param experiment_data: Dictionary containing experiment details.
        """

        # Load existing results
        all_sheets = self._load_existing_results()

        # Move old results to history folder
        self._move_old_results()

        # Get existing records for the dataset or create a new DataFrame
        df = all_sheets.get(dataset_name, pd.DataFrame())

        # Add 'trial' number
        trial_number = self.get_next_trial_number(df)
        experiment_data = {"trial": trial_number, **experiment_data}  # insert 'trial' number

        # Ensure all columns exist in the DataFrame
        for col in experiment_data.keys():
            if col not in df.columns:
                df[col] = pd.NA  # Add missing column

        # Convert CM (confusion matrix) to string
        if "CM" in experiment_data and isinstance(experiment_data["CM"], list):
            experiment_data["CM"] = str(experiment_data["CM"])

        # Append new record
        new_record = pd.DataFrame([experiment_data])
        df = pd.concat([df, new_record], ignore_index=True)

        # Save back to Excel
        with pd.ExcelWriter(self.file_path, mode="w", engine="openpyxl") as writer:
            for sheet_name, sheet_data in all_sheets.items():
                if sheet_name != dataset_name:
                    sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            df.to_excel(writer, sheet_name=dataset_name, index=False)

        print(f"Saved experiment result for dataset: {dataset_name} in {self.file_name}")

    def show_results(self, dataset_name):
        """Prints existing results for a given dataset."""
        all_sheets = self._load_existing_results()
        if dataset_name in all_sheets:
            print(all_sheets[dataset_name])
        else:
            print(f"No records found for dataset: {dataset_name}")


# Example Usage
if __name__ == "__main__":
    logger = ExperimentLogger(file_name="GCN_experiments", move_old=True)  # 檔名 = GCN_experiments_YYYYMMDD.xlsx

    experiment_data_1 = {
        "Model": "GCN2",
        "lr": 0.01,
        "n_epoch": 1000,
        "Loss": 0.345,
        "AUC": 0.87,
        "Acc": 0.92,
        "Pr": 0.88,
        "Re": 0.85,
        "F1": 0.86,
        "CM": [[50, 10], [5, 80]],  # 轉成字串儲存
        "Threshold": 0.5,
        "note": "Standard settings"
    }

    experiment_data_2 = {
        "Model": "GCN3",
        "lr": 0.005,
        "n_epoch": 500,
        "Loss": 0.312,
        "AUC": 0.90,
        "Acc": 0.94,
        "Pr": 0.89,
        "Re": 0.86,
        "F1": 0.87,
        "CM": [[45, 12], [4, 82]],
        "Threshold": 0.6,
        "note": "Lower LR, better AUC",
        "123": "456"
    }

    logger.log_experiment("Facebook", experiment_data_1)
    logger.show_results("Facebook")

    logger.log_experiment("Cora", experiment_data_2)
    logger.show_results("Cora")
