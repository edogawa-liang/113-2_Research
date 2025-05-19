import os
import torch
import torch.nn.functional as F
from utils.plot import ClassificationPlotter, RegressionPlotter
from utils.evaluation import ClassificationEvaluator, RegressionEvaluator
from utils.save_model import save_model_and_config


class GNNClassifierTrainer:
    """
    A trainer class for training, calling for evaluating, and saving a GNN model.
    """

    def __init__(self, dataset_name, data, num_features, num_classes, model_class, trial_number, device=None, 
                 lr=0.01, weight_decay=1e-4, epochs=1000, run_mode="try", threshold=0.5):
        # model setting
        self.device = device 
        self.data = data.to(self.device)
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = model_class(in_channels=self.num_features, out_channels=self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.threshold = threshold


        # use run_mode to determine the base directory
        base_dir = os.path.join("saved", run_mode)
        # create directories
        model_dir = os.path.join(base_dir, "model", dataset_name)
        plot_dir = os.path.join(base_dir, "plot", dataset_name)

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.model_name = model_class.__name__  
        self.save_model_dir = model_dir  
        self.save_plot_dir = plot_dir  

        # create save paths
        self.save_model_path = os.path.join(self.save_model_dir, f"{trial_number}_{self.model_name}.pth")
        self.save_config_path = os.path.join(self.save_model_dir, f"{trial_number}_{self.model_name}_config.pth")
        self.save_plot_path = os.path.join(self.save_plot_dir, f"{trial_number}_{self.model_name}.png")

        # results
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.best_metrics = {}

        # Initialize Plotter
        self.metric_plotter = ClassificationPlotter(save_dir=self.save_plot_path)

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

        # 根據 y 的長度判斷：是否有加 feature-nodes
        if self.data.train_mask.shape[0] != self.data.y.shape[0]:
            mask = self.data.train_mask[:self.data.y.shape[0]]
        else:
            mask = self.data.train_mask
        # self.data.y = self.data.y.to(torch.long) # 這一步讓y都變成0了
        loss = F.cross_entropy(out[mask], self.data.y[mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def save_model(self):
        training_params = {
            "model_name": self.model_name,
            "in_channels": self.num_features,
            "out_channels": self.num_classes,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }
        save_model_and_config(self.model, self.save_model_path, self.save_config_path, training_params)


    def run(self):
        """Runs the training process, evaluates metrics, and saves the best model."""
        print(f"Training GNN on {self.device} for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            loss = self.train()
            metrics = ClassificationEvaluator.evaluate(self.model, self.data, self.threshold)

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

            if epoch % max(1, self.epochs // 20) == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:04d}, Loss: {loss:.4f}, Val Acc: {metrics['Val_Acc']:.4f}, Test Acc: {metrics['Test_Acc']:.4f}, Val F1: {metrics['Val_F1']:.4f}, Test F1: {metrics['Test_F1']:.4f}")

            # Save the best model and metrics based on validation accuracy
            if metrics["Val_Acc"] > self.best_val_acc:
                self.best_val_acc = metrics["Val_Acc"]
                self.best_test_acc = metrics["Test_Acc"]
                self.best_loss = loss
                self.best_epoch = epoch
                self.best_metrics = metrics.copy()
                self.save_model()

        print("==============================================================\n")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print(f"Best Test Accuracy: {self.best_test_acc:.4f}")
        print("==============================================================\n")

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
            "Threshold": self.best_metrics.get("Threshold", None),
        }


class GNNRegressorTrainer:
    """
    A trainer class for training, evaluating, and saving a GNN model for node regression.
    """

    def __init__(self, dataset_name, data, num_features, model_class, trial_number, device=None, 
                 lr=0.01, weight_decay=1e-4, epochs=1000, run_mode="try"):
        # Model settings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.num_features = num_features
        self.model = model_class(in_channels=self.num_features).to(self.device) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        # Set up directories based on run_mode
        base_dir = os.path.join("saved", run_mode)
        model_dir = os.path.join(base_dir, "model", dataset_name)
        plot_dir = os.path.join(base_dir, "plot", dataset_name)

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.model_name = model_class.__name__  
        self.save_model_dir = model_dir  
        self.save_plot_dir = plot_dir  

        # Create save paths
        self.save_model_path = os.path.join(self.save_model_dir, f"{trial_number}_{self.model_name}.pth")
        self.save_config_path = os.path.join(self.save_model_dir, f"{trial_number}_{self.model_name}_config.pth")
        self.save_plot_path = os.path.join(self.save_plot_dir, f"{trial_number}_{self.model_name}.png")

        # Track best model
        self.best_val_mse = float("inf")
        self.best_test_mse = float("inf")
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.best_metrics = {}

        # Initialize Plotter
        self.metric_plotter = RegressionPlotter(save_dir=self.save_plot_path)

        # Initialize metric storage
        self.metrics = {
            "train_loss": [], "val_mse": [], "test_mse": [],
            "val_mae": [], "test_mae": [],
            "val_r2": [], "test_r2": [],
        }

    def train(self):
        """Trains the model for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index).squeeze()  # 確保輸出為 1D
        loss = F.mse_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])  # MSE Loss
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def save_model(self):
        """Saves the trained model along with its hyperparameters."""
        torch.save(self.model.state_dict(), self.save_model_path)
        config = {
            "model_name": self.model_name,
            "in_channels": self.num_features,
            "out_channels": 1,  # 回歸輸出只有 1 維
            "epochs": self.epochs,  # 儲存的是完整 epochs，而不是最佳模型的 epochs
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }
        torch.save(config, self.save_config_path)
        print(f"\nModel saved at {self.save_model_path}")
        print(f"Config saved at {self.save_config_path}")

    def run(self):
        """Runs the training process, evaluates metrics, and saves the best model."""
        print(f"Training GNN on {self.device} for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            loss = self.train()
            metrics = RegressionEvaluator.evaluate(self.model, self.data)  # 回歸不需要 threshold

            # Store metrics
            self.metrics["train_loss"].append(loss)
            self.metrics["val_mse"].append(metrics["Val_MSE"])
            self.metrics["test_mse"].append(metrics["Test_MSE"])
            self.metrics["val_mae"].append(metrics["Val_MAE"])
            self.metrics["test_mae"].append(metrics["Test_MAE"])
            self.metrics["val_r2"].append(metrics["Val_R2"])
            self.metrics["test_r2"].append(metrics["Test_R2"])

            if epoch % max(1, self.epochs // 20) == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:04d}, Loss: {loss:.4f}, Val MSE: {metrics['Val_MSE']:.4f}, Test MSE: {metrics['Test_MSE']:.4f}, Val R²: {metrics['Val_R2']:.4f}, Test R²: {metrics['Test_R2']:.4f}")

            # Save the best model based on validation MSE
            if metrics["Val_MSE"] < self.best_val_mse:
                self.best_val_mse = metrics["Val_MSE"]
                self.best_test_mse = metrics["Test_MSE"]
                self.best_loss = loss
                self.best_epoch = epoch
                self.best_metrics = metrics.copy()
                self.save_model()

        print("==============================================================\n")
        print(f"Best Validation MSE: {self.best_val_mse:.4f} at epoch {self.best_epoch}")
        print(f"Best Test MSE: {self.best_test_mse:.4f}")
        print("==============================================================\n")

        # Plot metrics
        self.metric_plotter.plot_metrics(self.metrics, self.epochs)
        
        # Return results for external use
        return {
            "Model": self.model_name,
            'LR': self.lr,
            "Epochs": self.epochs,
            "Best Epoch": self.best_epoch,
            "Loss": self.best_loss,
            "MSE": self.best_test_mse,
            "MAE": self.best_metrics.get("Test_MAE", 0),
            "R2": self.best_metrics.get("Test_R2", 0),
        }
