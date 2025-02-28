import matplotlib.pyplot as plt
import os

class MetricPlotter:
    """
    Supports loss, accuracy, AUC, precision, recall, and F1-score.
    """

    def __init__(self, save_dir, dpi=300):
        """
        Initializes the plotter.
        """
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = save_dir


    def plot_metrics(self, metrics, n_epoch):
        """
        Plots training and validation metrics over epochs.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Training Performance Metrics", fontsize=16)

        # Define metric titles and keys
        metric_info = [
            ("Training Loss", "train_loss"),
            ("Validation Accuracy vs Test Accuracy", ["val_acc", "test_acc"]),
            ("Validation AUC vs Test AUC", ["val_auc", "test_auc"]),
            ("Validation Precision vs Test Precision", ["val_precision", "test_precision"]),
            ("Validation Recall vs Test Recall", ["val_recall", "test_recall"]),
            ("Validation F1-Score vs Test F1-Score", ["val_f1", "test_f1"])
        ]

        for ax, (title, keys) in zip(axes.flat, metric_info):
            ax.set_title(title)
            ax.set_xlabel("Epochs")

            if isinstance(keys, list):  # Dual plots (Validation vs Test)
                for key in keys:
                    if key in metrics:
                        ax.plot(range(1, n_epoch+1), metrics[key], label=key.replace("_", " ").title())
                ax.legend()
            else:  # Single plot (Loss)
                if keys in metrics:
                    ax.plot(range(1, n_epoch+1), metrics[keys])

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=self.dpi)
        plt.show()




# Example Usage
if __name__ == "__main__":
    # Example metric records
    n_epoch = 100
    example_metrics = {
        "train_loss": [0.9 - 0.008*i for i in range(n_epoch)],
        "val_acc": [0.48 + 0.004*i for i in range(n_epoch)],
        "test_acc": [0.45 + 0.003*i for i in range(n_epoch)],
        "val_auc": [0.58 + 0.003*i for i in range(n_epoch)],
        "test_auc": [0.55 + 0.002*i for i in range(n_epoch)],
        "val_precision": [0.38 + 0.004*i for i in range(n_epoch)],
        "test_precision": [0.36 + 0.003*i for i in range(n_epoch)],
        "val_recall": [0.53 + 0.004*i for i in range(n_epoch)],
        "test_recall": [0.50 + 0.003*i for i in range(n_epoch)],
        "val_f1": [0.42 + 0.004*i for i in range(n_epoch)],
        "test_f1": [0.40 + 0.003*i for i in range(n_epoch)],
    }

    # Initialize and plot
    plotter = MetricPlotter()
    plotter.plot_metrics(example_metrics, n_epoch)
