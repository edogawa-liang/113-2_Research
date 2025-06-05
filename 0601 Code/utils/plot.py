import matplotlib.pyplot as plt

class ClassificationPlotter:
    """
    Supports loss, accuracy, AUC, precision, recall, and F1-score.
    """

    def __init__(self, save_dir, dpi=300):
        """
        Initializes the plotter.
        """
        self.save_dir = save_dir
        self.dpi = dpi
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
        print(f"Metrics plot saved at {self.save_path}")


class RegressionPlotter:
    """
    Plots loss, MSE, MAE, and R² over training epochs.
    """

    def __init__(self, save_dir, dpi=300):
        """
        Initializes the plotter.
        """
        self.save_dir = save_dir
        self.dpi = dpi
        self.save_path = save_dir

    def plot_metrics(self, metrics, n_epoch):
        """
        Plots training and validation metrics over epochs.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Training Performance Metrics (Regression)", fontsize=16)

        # Define metric titles and keys
        metric_info = [
            ("Training Loss", "train_loss"),
            ("Validation MSE vs Test MSE", ["val_mse", "test_mse"]),
            ("Validation MAE vs Test MAE", ["val_mae", "test_mae"]),
            ("Validation R² vs Test R²", ["val_r2", "test_r2"])
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
        print(f"Metrics plot saved at {self.save_path}")
