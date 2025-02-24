import pandas as pd
import numpy as np
from scipy.stats import entropy

class SensitiveAttributeEvaluator:
    def __init__(self, df, a=1, b=1, c=1):
        """
        Initialize the evaluator.
        :param df: pandas DataFrame, the dataset to analyze
        :param a: Weight for Ratio
        :param b: Weight for Entropy
        :param c: Weight for Imbalance
        """
        self.df = df
        self.a = a
        self.b = b
        self.c = c
    
    @staticmethod
    def calculate_ratio(column):
        """Calculate the ratio of unique values (identifiability)."""
        if column.dtype == 'O':  # Categorical data
            return column.nunique() / len(column)
        else:  # Continuous data (binning required to avoid every value being unique)
            value_counts, _ = np.histogram(column, bins='auto')
            return len(value_counts) / len(column)
    
    @staticmethod
    def calculate_entropy(column):
        """Calculate normalized Shannon Entropy (information content) between 0 and 1."""
        if column.dtype == 'O':  # Categorical data
            value_counts = column.value_counts(normalize=True)
        else:  # Continuous data (binning required)
            value_counts, _ = np.histogram(column, bins='auto', density=True)
            value_counts = value_counts / value_counts.sum()
        entropy_value = entropy(value_counts, base=2) if len(value_counts) > 1 else 0
        return entropy_value / np.log2(len(value_counts)) if len(value_counts) > 1 else 0  # Normalize entropy to [0,1]
    
    @staticmethod
    def calculate_imbalance(column):
        """Calculate category imbalance (Gini Impurity)."""
        if column.dtype == 'O':  # Categorical data
            value_counts = column.value_counts(normalize=True)
        else:  # Continuous data (binning required)
            value_counts, _ = np.histogram(column, bins='auto', density=True)
            value_counts = value_counts / value_counts.sum()
        return 1 - np.sum(value_counts ** 2)  # Gini Index
    
    def compute_sensitivity(self):
        """Compute the sensitivity score S for each column."""
        scores = {}
        for column in self.df.columns:
            ratio = self.calculate_ratio(self.df[column])
            ent = self.calculate_entropy(self.df[column])
            imb = self.calculate_imbalance(self.df[column])
            S = self.a * ratio + self.b * ent + self.c * imb
            scores[column] = {"Ratio": ratio, "Entropy": ent, "Imbalance": imb, "Sensitivity Score": S}
            # print the column which ratio is greater than 0.8
            if ratio > 0.8:
                print(f"Column {column} has a high ratio: {ratio}")
        
        # Convert to DataFrame for better visualization
        result_df = pd.DataFrame.from_dict(scores, orient='index')
        result_df = result_df.sort_values(by="Sensitivity Score", ascending=False)
        return result_df, scores


if __name__ == "__main__":
    from torch_geometric.datasets import FacebookPagePage
    import torch_geometric.transforms as T
    
    # Try on Facebook dataset
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomNodeSplit(num_val=0.1, num_test=0.1),
    ])
    dataset = FacebookPagePage(root='/tmp/FacebookPagePage', transform=transform)
    data = dataset[0] 
    df = pd.DataFrame(data.x.numpy()) if data.x is not None else None
    
    
    # Initialize and compute sensitivity
    evaluator = SensitiveAttributeEvaluator(df, a=0.1, b=0.4, c=0.5)
    sensitivity_df, sensitivity_score = evaluator.compute_sensitivity()
    print(sensitivity_df)
