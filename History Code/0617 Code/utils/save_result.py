import os
import pandas as pd
import glob
from datetime import datetime

class ExperimentLogger:
    """
    A logger for storing GNN training results in an Excel file.
    """

    def __init__(self, file_name, note, copy_old, run_mode):
        """
        Initializes the logger.

        :param file_name: Custom name for the Excel file (without extension).
        :param note: Experiment note.
        :param copy_old: Whether to move old results to history before saving new ones.
        :param run_mode: The experiment mode (e.g., 'try', 'stage1', 'stage2').
        """

        # Set base directory based on the run mode (e.g., try, stage1, stage2)
        base_dir = os.path.join("saved", run_mode)
        result_dir = os.path.join(base_dir, "result")
        os.makedirs(result_dir, exist_ok=True)

        # Set paths for saving results
        self.save_dir = result_dir
        self.history_dir = os.path.join(self.save_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)

        # Generate a unique file name with a timestamp (minutes included)
        self.date_str = datetime.now().strftime("%m%d_%H%M")  
        self.file_prefix = file_name  # File prefix, e.g., 'result'
        self.file_name = f"{self.file_prefix}_{self.date_str}.xlsx"
        self.file_path = os.path.join(self.save_dir, self.file_name)

        self.note = note
        self.copy_old = copy_old

        # Load existing results from the latest matching file
        self.trial_cache = self._load_existing_results()

    def _get_latest_matching_file(self):
        """
        Finds the latest Excel file that matches the file prefix in the result directory.
        This ensures continuity in logging without creating unnecessary new files.
        """
        matching_files = glob.glob(os.path.join(self.save_dir, f"{self.file_prefix}_*.xlsx"))
        if matching_files:
            return sorted(matching_files)[-1]  # Return the most recent matching file
        return None

    def _copy_old_results(self):
        """
        Moves old Excel files with the same prefix to the 'history' folder
        to prevent accidental overwrites.
        """
        latest_file = self._get_latest_matching_file()
        if self.copy_old and latest_file and latest_file != self.file_path:
            new_path = os.path.join(self.history_dir, os.path.basename(latest_file))
            os.rename(latest_file, new_path)
            print(f"Moved old results to {new_path}")

    def _load_existing_results(self):
        """
        Loads the latest existing results from an Excel file with the same prefix.
        If no previous file exists, an empty dictionary is returned.
        """
        latest_file = self._get_latest_matching_file()
        if latest_file:
            print(f"Loading existing results from {latest_file}")
            return pd.read_excel(latest_file, sheet_name=None, engine="openpyxl")
        return {}

    def get_next_trial_number(self, dataset_name):
        """
        Finds the next available trial number for a given dataset.
        Ensures that new experiments are sequentially numbered.
        """
        df = self.trial_cache.get(dataset_name, pd.DataFrame())
        if "Trial" in df.columns and not df.empty:
            return df["Trial"].max() + 1
        return 0  

    def log_experiment(self, dataset_name, experiment_data, label_source, **extra_columns):
        """
        Logs an experiment result into an Excel file.
        If the dataset does not exist in the current file, a new sheet is created.

        :param dataset_name: Name of the dataset.
        :param experiment_data: Dictionary containing experiment details.
        """
        # Load existing data
        df = self.trial_cache.get(dataset_name, pd.DataFrame())

        # Move old files to history before writing new data
        self._copy_old_results()

        # Get the next trial number
        trial_number = self.get_next_trial_number(dataset_name)
        experiment_data = {"Trial": trial_number, "Label": label_source, **extra_columns, **experiment_data, "Note": self.note} 

        # Ensure all columns exist in the DataFrame
        for col in experiment_data.keys():
            if col not in df.columns:
                df[col] = pd.NA  # Add missing column

        # Convert Confusion Matrix (CM) to string for better storage in Excel
        if "CM" in experiment_data and isinstance(experiment_data["CM"], list):
            experiment_data["CM"] = str(experiment_data["CM"])

        # Append new record
        new_record = pd.DataFrame([experiment_data])
        df = pd.concat([df, new_record], ignore_index=True)

        # Update cache with new results
        self.trial_cache[dataset_name] = df

        # Save updated results to Excel
        with pd.ExcelWriter(self.file_path, mode="w", engine="openpyxl") as writer:
            for sheet_name, sheet_data in self.trial_cache.items():
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved experiment result for dataset: {dataset_name} in {self.file_path}\n")

    def show_results(self, dataset_name):
        """
        Prints existing results for a given dataset.
        """
        if dataset_name in self.trial_cache:
            print(self.trial_cache[dataset_name])
        else:
            print(f"No records found for dataset: {dataset_name}")
