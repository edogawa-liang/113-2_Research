import argparse
import pandas as pd
from subgraph_selector.utils.feat_sel import SensitiveAttributeEvaluator
from utils.dataset_loader import GraphDatasetLoader

def main(dataset_name, a, b, c):
    loader = GraphDatasetLoader()
    if dataset_name not in loader.datasets:
        print(f"錯誤: {dataset_name} 不在可用數據集 {list(loader.datasets.keys())} 之內。")
        return
    
    dataset, data = loader.load_dataset(dataset_name)
    node_features = data.x.cpu().numpy() if data.x is not None else None
    
    if node_features is not None:
        df_features = pd.DataFrame(node_features)
        evaluator = SensitiveAttributeEvaluator(df_features, a=a, b=b, c=c)
        sensitivity_df, _ = evaluator.compute_sensitivity()
        print(sensitivity_df)
    else:
        print("未找到節點特徵。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="計算圖數據的敏感性分數")
    parser.add_argument("--dataset", type=str, required=True, help="選擇要載入的數據集")
    parser.add_argument("--a", type=float, default=1.0, help="敏感性計算的參數 a")
    parser.add_argument("--b", type=float, default=1.0, help="敏感性計算的參數 b")
    parser.add_argument("--c", type=float, default=1.0, help="敏感性計算的參數 c")
    args = parser.parse_args()
    
    main(args.dataset, args.a, args.b, args.c)
