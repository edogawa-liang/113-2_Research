import argparse
import torch
import os
from data.dataset_loader import GraphDatasetLoader
from data.data_modifier import GraphModifier
from subgraph_selector.utils.feat_sel import FeatureSelector
from models.basic_GCN import GCN2Classifier, GCN3Classifier, GCN2Regressor, GCN3Regressor
from trainer.gnn_trainer import GNNClassifierTrainer, GNNRegressorTrainer
from utils.save_result import ExperimentLogger




def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN model with specified parameters.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    
    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for binary classification")
    
    # result settings
    parser.add_argument("--run_mode", type=str, default="try", help="Experiment run mode: 'try' (testing) or formal runs")
    parser.add_argument("--result_filename", type=str, default="result", help="File name for results document")
    parser.add_argument("--note", type=str, default="", help="Note for the experiment")
    parser.add_argument("--copy_old", type=lambda x: x.lower() == "true", default=True, help="Whether to backup old experiment data (true/false).")
    
    # Feature selection parameters
    parser.add_argument("--use_original_label", type=lambda x: x.lower() == "true", default=True, help="Use original labels (true/false)")
    parser.add_argument("--feature_selection_method", type=str, default="pca", choices=["pca", "svd", "tree", "mutual_info"], help="Feature selection method")
    parser.add_argument("--top_n", type=int, default=6, help="Number of top features to select")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    os.environ['TORCH'] = torch.__version__
    print(f"Using torch version: {torch.__version__}")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    loader = GraphDatasetLoader()
    data, num_features, num_classes = loader.load_dataset(args.dataset)
    data = data.to(device)

    if args.use_original_label is False:
        print(f"Performing feature selection using {args.feature_selection_method.upper()}...")

        # 初始化 FeatureSelector
        selector = FeatureSelector(
            method=args.feature_selection_method,
            top_n=args.top_n,
            top_n_features_per_pc=2
        )

        # 若方法需要 labels，則提供 labels
        if args.feature_selection_method in ["tree", "mutual_info"]:
            if data.y is None:
                raise ValueError(f"{args.feature_selection_method} feature selection requires labels (y).")
            selector.fit(data.x.cpu().numpy(), labels=data.y.cpu().numpy())
        else:
            selector.fit(data.x.cpu().numpy())

        imp_features = selector.get_top_features()
        print(f"Selected important features: {imp_features}")

        # Modify graph using selected features
        modifier = GraphModifier(data)
        modified_graphs = modifier.modify_graph(imp_features)  # List of graphs
        print(f"Graph modified into {len(modified_graphs)} graphs.")

        # One feature is removed from the original dataset to label
        num_features = num_features - 1


    else:
        modified_graphs = [data]  
        print("Using original dataset without feature selection.")
        modified_graphs[0].task_type = "classification"


    # Initialize logger
    logger = ExperimentLogger(file_name=args.result_filename, note=args.note, copy_old=args.copy_old, run_mode=args.run_mode)

    # Loop over all modified graphs
    for i, graph in enumerate(modified_graphs):
        print(f"\nTraining on Graph {i+1}/{len(modified_graphs)} - Task: {graph.task_type}")
        label_source = "Original Label" if args.use_original_label else f"Feature {imp_features[i]}"

        if graph.task_type == "regression":
            trial_number = logger.get_next_trial_number(args.dataset + "_regression")
            print(f"Training Regression Model - Trial {trial_number}")
            trainer = GNNRegressorTrainer(dataset_name=args.dataset, data=graph, 
                                        num_features=num_features, 
                                        model_class=GCN2Regressor if args.model == "GCN2" else GCN3Regressor,
                                        trial_number=trial_number, device=device,
                                        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                        run_mode=args.run_mode)
            result = trainer.run()
            logger.log_experiment(args.dataset + "_regression", result, label_source, feat_sel_method=args.feature_selection_method)

        
        elif graph.task_type == "classification": 
            trial_number = logger.get_next_trial_number(args.dataset + "_classification")
            print(f"Training Classification Model - Trial {trial_number}")
            trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=graph, 
                                        num_features=num_features, num_classes=num_classes,  
                                        model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                        trial_number=trial_number, device=device,
                                        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, 
                                        run_mode=args.run_mode, threshold=args.threshold)
            result = trainer.run()
            logger.log_experiment(args.dataset + "_classification", result, label_source, feat_sel_method=args.feature_selection_method)

