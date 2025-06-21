import os
import torch
import torch.nn.functional as F
from utils.plot import ClassificationPlotter
from utils.evaluation import ClassificationEvaluator
from utils.save_model import save_model_and_config

class MLPClassifierTrainer:
    def __init__(self, dataset_name, data, num_features, num_classes, model_class, trial_number=None, device=None,
                 lr=0.01, weight_decay=1e-4, epochs=1000, run_mode="try", threshold=0.5):
        self.device = device
        self.data = data.to(self.device)
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = model_class(
            input_dim=num_features, hidden_dim1=512, hidden_dim2=128,
            output_dim=num_classes, dropout1=0.5, dropout2=0.5
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.threshold = threshold
        self.model_name = model_class.__name__
        self.run_mode = run_mode
        self.trial_number = trial_number
        self.dataset_name = dataset_name

        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.best_metrics = {}

        self.metrics = {
            "train_loss": [], "val_acc": [], "test_acc": [],
            "val_auc": [], "test_auc": [],
            "val_precision": [], "test_precision": [],
            "val_recall": [], "test_recall": [],
            "val_f1": [], "test_f1": [],
            "val_cm": [], "test_cm": []
        }

    def _prepare_save_paths(self, dataset_name, model_class, run_mode, trial_number):
        base_dir = os.path.join("saved", run_mode)
        model_dir = os.path.join(base_dir, "model", dataset_name)
        plot_dir = os.path.join(base_dir, "plot", dataset_name)

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.model_name = model_class.__name__
        self.save_model_dir = model_dir
        self.save_plot_dir = plot_dir

        self.save_model_path = os.path.join(model_dir, f"{trial_number}_{self.model_name}.pth")
        self.save_config_path = os.path.join(model_dir, f"{trial_number}_{self.model_name}_config.pth")
        self.save_plot_path = os.path.join(plot_dir, f"{trial_number}_{self.model_name}.png")

        self.metric_plotter = ClassificationPlotter(save_dir=self.save_plot_path)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x)

        num_orig_nodes = self.data.y.shape[0]
        if self.data.train_mask.shape[0] != num_orig_nodes:
            train_loss_mask = self.data.train_mask[:num_orig_nodes]
            out = out[:num_orig_nodes]
        else:
            train_loss_mask = self.data.train_mask

        loss = F.cross_entropy(out[train_loss_mask], self.data.y[train_loss_mask])
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
        print(f"Training MLP on {self.device} for {self.epochs} epochs...")
        if not hasattr(self, "save_model_path"):
            self._prepare_save_paths(
                dataset_name=self.dataset_name,
                model_class=self.model.__class__,
                run_mode=self.run_mode,
                trial_number=self.trial_number if self.trial_number is not None else "default"
            )

        for epoch in range(1, self.epochs + 1):
            loss = self.train()
            metrics = ClassificationEvaluator.evaluate(self.model, self.data, self.threshold)

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

        self.metric_plotter.plot_metrics(self.metrics, self.epochs)

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

    def load_model(self, model_path):
        print(f"[MLPClassifierTrainer] Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def test(self):
        print("[MLPClassifierTrainer] Running test on current test_mask...")
        metrics = ClassificationEvaluator.evaluate(self.model, self.data, self.threshold)
        print(f"Test Acc: {metrics['Test_Acc']:.4f}, Test F1: {metrics['Test_F1']:.4f}, Test AUC: {metrics['Test_AUC']:.4f}")
        return {
            "Model": self.model_name,
            "Acc": metrics.get("Test_Acc", 0),
            "Auc": metrics.get("Test_AUC", 0),
            "Precision": metrics.get("Test_Pr", 0),
            "Recall": metrics.get("Test_Re", 0),
            "F1": metrics.get("Test_F1", 0),
            "CM": metrics.get("CM", []),
        }
