import os
import torch
import torch.nn.functional as F
from utils.plot import MetricPlotter
from utils.evaluation import Evaluator  

class GNNTrainer:
    """
    A trainer class for training, evaluating, and saving a GNN model.
    """

    def __init__(self, dataset, model_class, model_name="GCN1", device=None, lr=0.01, weight_decay=1e-4, epochs=1000, save_dir="saved/models"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.data = dataset[0].to(self.device)
        self.model = model_class(in_channels=dataset.num_features, out_channels=dataset.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.save_path = os.path.join(save_dir, f"{model_name}.pth")
        self.best_val_acc = 0
        self.best_test_acc = 0

        os.makedirs(save_dir, exist_ok=True)

        # Initialize MetricPlotter
        self.metric_plotter = MetricPlotter(plot_name=model_name, save_dir=)

        # Initialize metric storage
        self.metrics = {
            "train_loss": [], "train_acc": [], "val_acc": [],
            "train_auc": [], "val_auc": [],
            "train_precision": [], "val_precision": [],
            "train_recall": [], "val_recall": [],
            "train_f1": [], "val_f1": []
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

            # Evaluate using the Evaluator class
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

            # Save the best model
            if metrics["Val_Acc"] > self.best_val_acc:
                self.best_val_acc = metrics["Val_Acc"]
                self.best_test_acc = metrics["Test_Acc"]
                self.save_model()

            if epoch % 50 == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:04d}, Loss: {loss:.4f}, Val Acc: {metrics['Val_Acc']:.4f}, Test Acc: {metrics['Test_Acc']:.4f}")

        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"Best Test Accuracy: {self.best_test_acc:.4f}")

        # Plot metrics
        self.metric_plotter.plot_metrics(self.metrics, self.epochs)




# Train GCN model
if __name__ == "__main__":
    from data.dataset_loader import GraphDatasetLoader
    from data.data_modifier import GraphModifier
    from subgraph_selector.utils.feat_sel import PCAFeatureSelector
    from models.basic_GCN import GCN2, GCN3  
    from utils.experiment import ExperimentLogger

    # load data
    loader = GraphDatasetLoader()
    dataset_name = input(f"Enter dataset name {list(loader.datasets.keys())}: ")
    data = loader.load_dataset(dataset_name)

    # calculate feature importance
    pca_selector = PCAFeatureSelector(top_n_pcs=3, top_n_features_per_pc=2)  
    pca_selector.fit(data.x.cpu().numpy())
    imp_features = pca_selector.get_top_features() # Get the union of the most important features

    # modify data
    modifier = GraphModifier(data)
    modified_graphs = modifier.modify_graph(imp_features)

    trainer = GNNTrainer(data, GCN2)
    trainer.run()


    # save data
    logger = ExperimentLogger(file_name="GCN_experiments", move_old=True)
    trial = logger.get_next_trial_number()
