import argparse
from data.dataset_loader import GraphDatasetLoader
from data.data_modifier import GraphModifier
from subgraph_selector.utils.feat_sel import PCAFeatureSelector
from models.basic_GCN import GCN2, GCN3
from trainer.gnn_trainer import GNNTrainer
from utils.save_result import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN model with specified parameters.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    
    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    
    parser.add_argument("--model_dir", type=str, default="saved/model", help="Directory to save trained models")
    parser.add_argument("--plot_dir", type=str, default="saved/plot", help="Directory to save plots")
    parser.add_argument("--result_dir", type=str, default="saved/result", help="Directory to save results")
    parser.add_argument("--result_filename", type=str, default="gnn_experiment", help="File name for results document")
    parser.add_argument("--note", type=str, default="", help="Note for the experiment")
    parser.add_argument("--copy_old", type=lambda x: x.lower() == "true", default=True, help="Whether to backup old experiment data (true/false).")
    
    parser.add_argument("--top_pcs", type=int, default=3, help="Number of principal components for PCA")
    parser.add_argument("--top_features", type=int, default=2, help="Number of top features per PC")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # Load dataset
    loader = GraphDatasetLoader()
    data = loader.load_dataset(args.dataset)

    # Feature selection using PCA
    pca_selector = PCAFeatureSelector(top_n_pcs=args.top_pcs, top_n_features_per_pc=args.top_features)
    pca_selector.fit(data.x.cpu().numpy())
    imp_features = pca_selector.get_top_features()

    # Modify graph using selected features
    modifier = GraphModifier(data)
    modified_graphs = modifier.modify_graph(imp_features)

    # Select model
    model_class = GCN2 if args.model == "GCN2" else GCN3

    # Get the trial number
    logger = ExperimentLogger(save_dir=args.result_dir, file_name=args.result_filename, move_old=args.copy_old)
    trial_number = logger.get_next_trial_number()

    # Train GNN model
    trainer = GNNTrainer(dataset_name=args.dataset, data=modified_graphs, model_class=model_class, 
                         trial_number=trial_number, epochs=args.epochs, lr=args.lr, 
                         weight_decay=args.weight_decay, save_model_dir=args.model_dir, 
                         save_plot_dir=args.plot_dir, copy_old=args.copy_old)
    result = trainer.run()

    # Save experiment results
    logger.log_experiment(args.dataset, result)
