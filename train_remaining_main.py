import torch
import os
import argparse
from data.dataset_loader import GraphDatasetLoader

from subgraph_selector.random_selector import RandomSubgraphSelector
# from subgraph_selector.explainer_selector import ExplainerSubgraphSelector
from subgraph_selector.remaining_graph import RemainingGraphConstructor

from models.basic_GCN import GCN2Classifier, GCN3Classifier
from trainer.gnn_trainer import GNNClassifierTrainer
from utils.save_result import ExperimentLogger

# python train_remaining_main.py --dataset GitHub --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult --note basic_node_cls --selector_type random --fraction 0.1
# python train_remaining_main.py --dataset FacebookPagePage --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult --note basic_node_cls --selector_type random --fraction 0.1
# python train_remaining_main.py --dataset Cora --model GCN2 --epochs 300 --lr 0.01 --run_mode baselineResult --note basic_node_cls --selector_type random --fraction 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN after removing a selected subgraph.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, default="GCN2", choices=["GCN2", "GCN3"], help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of edges to remove for the subgraph")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--selector_type", type=str, default="random", choices=["random", "explainer"], help="Subgraph selector type")
    parser.add_argument("--run_mode", type=str, default="try", help="Run mode")
    parser.add_argument("--filename", type=str, default="result", help="File name for saving results")
    parser.add_argument("--note", type=str, default="", help="Note for the experiment")
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

    # Select subgraph
    if args.selector_type == "random":
        print("Use random selector")
        selector = RandomSubgraphSelector(data, fraction=args.fraction, seed=args.seed)
    elif args.selector_type == "explainer":
        print("Use explainer selector")
        # selector = ExplainerSubgraphSelector(data)

    selected_edges = selector.select_subgraph()

    # Remove subgraph from the original graph
    remaining_graph_constructor = RemainingGraphConstructor(data, selected_edges)
    remaining_graph = remaining_graph_constructor.get_remaining_graph()

    # Train GNN on the remaining graph
    print("\nTraining GNN on the remaining graph after removing subgraph...")

    trial_number = 1  # Example trial number
    trainer = GNNClassifierTrainer(dataset_name=args.dataset, data=remaining_graph, 
                                   num_features=num_features, num_classes=num_classes, 
                                   model_class=GCN2Classifier if args.model == "GCN2" else GCN3Classifier,
                                   trial_number=trial_number, device=device,
                                   epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                   run_mode=args.run_mode)

    result = trainer.run()

    # Save experiment results
    logger = ExperimentLogger(file_name=args.filename, note=args.note, copy_old=True, run_mode=args.run_mode)
    logger.log_experiment(args.dataset + "_remaining_graph", result, label_source="Original", selector_type=args.selector_type, fraction=args.fraction)

    print("Experiment finished and results saved.")
