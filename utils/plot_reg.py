import matplotlib.pyplot as plt

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


# Example Usage
if __name__ == "__main__":
    # Example metric records
    n_epoch = 100
    example_metrics = {
        "train_loss": [0.9 - 0.008*i for i in range(n_epoch)],
        "val_mse": [0.1 + 0.002*i for i in range(n_epoch)],
        "test_mse": [0.12 + 0.002*i for i in range(n_epoch)],
        "val_mae": [0.2 + 0.0015*i for i in range(n_epoch)],
        "test_mae": [0.22 + 0.0015*i for i in range(n_epoch)],
        "val_r2": [0.5 + 0.002*i for i in range(n_epoch)],
        "test_r2": [0.48 + 0.002*i for i in range(n_epoch)],
    }

    # Initialize and plot
    plotter = MetricPlotter(save_dir="regression_metrics.png")
    plotter.plot_metrics(example_metrics, n_epoch)
