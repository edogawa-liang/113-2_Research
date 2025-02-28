import os
import torch
import torch.nn.functional as F
from utils.plot import MetricPlotter
from utils.evaluation import Evaluator  

class GNNTrainer:
    """
    A trainer class for training, calling for evaluating, and saving a GNN model.
    """

    def __init__(self, dataset_name, data, model_class, trial_number, device=None, 
                 lr=0.01, weight_decay=1e-4, epochs=1000, save_model_dir="saved/model", save_plot_dir="saved/plot"):
        # model setting
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.model = model_class(in_channels=data.num_features, out_channels=data.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs

        # save path
        self.model_name = model_class.__name__ # Automatically use class name as model name
        self.save_model_path = os.path.join(save_model_dir, dataset_name, f"{trial_number}_{self.model_name}.pth")
        self.save_plot_path = os.path.join(save_plot_dir,  dataset_name, f"{trial_number}_{self.model_name}.png")
        os.makedirs(self.save_model_path, exist_ok=True)
        os.makedirs(self.save_plot_path, exist_ok=True)

        # results
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.best_metrics = {}

        # Initialize MetricPlotter
        self.metric_plotter = MetricPlotter(save_dir=self.save_plot_path)

        # Initialize metric storage
        self.metrics = {
            "train_loss": [], "val_acc": [], "test_acc": [],
            "val_auc": [], "test_auc": [],
            "val_precision": [], "test_precision": [],
            "val_recall": [], "test_recall": [],
            "val_f1": [], "test_f1": [],
            "val_cm": [], "test_cm": []
        }

    def train(self):
        """Trains the model for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def save_model(self):
        """Saves the trained model."""
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved at {self.save_path}")

    def run(self):
        """Runs the training process, evaluates metrics, and saves the best model."""
        print(f"Training GNN on {self.device} for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            loss = self.train()
            metrics = Evaluator.evaluate(self.model, self.data)

            # Store metrics
            self.metrics["train_loss"].append(loss)
            self.metrics["val_acc"].append(metrics["Val_Acc"])
            self.metrics["test_acc"].append(metrics["Test_Acc"])
            self.metrics["val_auc"].append(metrics["Val_AUC"])
            self.metrics["test_auc"].append(metrics["Test_AUC"])
            self.metrics["val_precision"].append(metrics["Val_Pr"])
            self.metrics["test_precision"].append(metrics["Test_Pr"])
            self.metrics["val_recall"].append(metrics["Val_Re"])
            self.metrics["test_recall"].append(metrics["Test_Re"])
            self.metrics["val_f1"].append(metrics["Val_F1"])
            self.metrics["test_f1"].append(metrics["Test_F1"])

            # Save the best model and metrics based on validation accuracy
            if metrics["Val_Acc"] > self.best_val_acc:
                self.best_val_acc = metrics["Val_Acc"]
                self.best_test_acc = metrics["Test_Acc"]
                self.best_loss = loss
                self.best_epoch = epoch
                self.best_metrics = metrics.copy()
                self.save_model()

            if epoch % (self.epochs/20) == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:04d}, Loss: {loss:.4f}, Val Acc: {metrics['Val_Acc']:.4f}, Test Acc: {metrics['Test_Acc']:.4f}")

        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print(f"Best Test Accuracy: {self.best_test_acc:.4f}")

        # Plot metrics
        self.metric_plotter.plot_metrics(self.metrics, self.epochs)
        
        # Return results for external use
        return {
            "Model": self.model_name,
            'LR': self.lr,
            "Epochs": self.epochs,
            "Best Epoch": self.best_epoch,
            "Loss": self.best_loss,
            "Acc": self.best_test_acc,
            "Auc": self.best_metrics.get("Test_AUC", 0),
            "Precision": self.best_metrics.get("Test_Pr", 0),
            "Recall": self.best_metrics.get("Test_Re", 0),
            "F1": self.best_metrics.get("Test_F1", 0),
            "CM": self.best_metrics.get("CM", []),
        }



