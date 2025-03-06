import os
import pandas as pd
import glob

class FeatureExtractorXLSX:
    """
    A utility class for extracting selected feature indices from results files.
    """

    def __init__(self, result_dir):
        """
        Initializes the FeatureExtractor.

        :param result_dir: Directory containing results files.
        """
        self.result_dir = result_dir

    def find_latest_result_file(self):
        """
        Finds the latest results file that starts with 'results_'.

        :return: Path to the latest results file.
        """
        result_files = sorted(glob.glob(os.path.join(self.result_dir, "result_*.xlsx")), reverse=True)
        if not result_files:
            raise FileNotFoundError(f"No results file found in {self.result_dir} with prefix 'results_'")
        return result_files[0]

    def extract_feature_numbers(self, dataset_name):
        """
        Extracts feature indices from the sheet '{dataset_name}_regression' in the latest results file.

        :param dataset_name: Dataset name to match the sheet name (e.g., 'Facebook_regression').
        :return: List of feature indices.
        """
        result_file = self.find_latest_result_file()
        print(f"Using result file: {result_file}")

        sheet_name = f"{dataset_name}_regression"
        try:
            df = pd.read_excel(result_file, sheet_name=sheet_name, engine="openpyxl")
        except ValueError:
            raise ValueError(f"Sheet '{sheet_name}' not found in {result_file}")

        # Extract feature columns
        feature_indices = df["Label"].str.extract(r"(\d+)").astype(int).squeeze().tolist()

        # get the correspinding trial names
        feature_trials = df["Trial"].tolist()

        return feature_trials, feature_indices
