import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ClassificationEvaluator:
    """
    Evaluates a trained GNN model and computes various performance metrics.
    """

    @staticmethod
    @torch.no_grad()
    def evaluate(model, data, threshold):
        """
        Evaluates the model on validation and test sets and returns performance metrics.

        :param model: Trained GNN model.
        :param data: Graph data object.
        :param threshold: Threshold for binary classification.
        """
        model.eval()
        out = model(data.x, data.edge_index)

        # 安全地過濾掉 feature node
        is_ori = data.is_original_node if hasattr(data, "is_original_node") else torch.ones(data.y.shape[0], dtype=torch.bool, device=data.y.device)

        num_nodes = data.y.shape[0]  # 原始節點數
        out = out[:num_nodes]        # 僅取原始節點的輸出 (feature to node 時要注意)

        prob = F.softmax(out, dim=-1)  # get class probabilities
        pred = prob.argmax(dim=-1)  # get predicted class
        
        val_mask = data.val_mask[:num_nodes] & is_ori
        test_mask = data.test_mask[:num_nodes] & is_ori
        

        # Compute accuracy
        acc_val = (pred[val_mask] == data.y[val_mask]).sum().item() / val_mask.sum().item()
        acc_test = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()

        # Extract true & predicted values
        y_true_val = data.y[val_mask].cpu().numpy()
        y_pred_val = prob[val_mask].cpu().numpy()
        y_true_test = data.y[test_mask].cpu().numpy()
        y_pred_test = prob[test_mask].cpu().numpy()

        # only use the positive class's probability for binary classification!!!
        num_classes = out.shape[1]
        if num_classes == 2:
            y_pred_val = y_pred_val[:, 1]
            y_pred_test = y_pred_test[:, 1]
            use_threshold = threshold
        else:
            use_threshold = None 

        # Compute AUC 
        val_auc = roc_auc_score(y_true_val, y_pred_val, multi_class="ovr", average="macro") if num_classes > 2 else roc_auc_score(y_true_val, y_pred_val)
        test_auc = roc_auc_score(y_true_test, y_pred_test, multi_class="ovr", average="macro") if num_classes > 2 else roc_auc_score(y_true_test, y_pred_test)

        # Get predicted class
        # if binary classification, use 0.5 as threshold
        y_pred_class_val = y_pred_val.argmax(axis=1) if num_classes > 2 else (y_pred_val > use_threshold).astype(int)
        y_pred_class_test = y_pred_test.argmax(axis=1) if num_classes > 2 else (y_pred_test > use_threshold).astype(int)

        # Compute precision, recall, F1-score
        val_precision = precision_score(y_true_val, y_pred_class_val, average="macro", zero_division=0)
        test_precision = precision_score(y_true_test, y_pred_class_test, average="macro", zero_division=0)

        val_recall = recall_score(y_true_val, y_pred_class_val, average="macro", zero_division=0)
        test_recall = recall_score(y_true_test, y_pred_class_test, average="macro", zero_division=0)

        val_f1 = f1_score(y_true_val, y_pred_class_val, average="macro", zero_division=0)
        test_f1 = f1_score(y_true_test, y_pred_class_test, average="macro", zero_division=0)

        # Compute confusion matrix
        cm = confusion_matrix(y_true_test, y_pred_class_test).tolist()

        return {
            "Threshold": threshold,
            "Val_AUC": val_auc,
            "Test_AUC": test_auc,
            "Val_Acc": acc_val,
            "Test_Acc": acc_test,
            "Val_Pr": val_precision,
            "Test_Pr": test_precision,
            "Val_Re": val_recall,
            "Test_Re": test_recall,
            "Val_F1": val_f1,
            "Test_F1": test_f1,
            "CM": cm,
            "Threshold": use_threshold
        }
    

class RegressionEvaluator:
    """
    Evaluates a trained GNN model for node regression.
    """

    @staticmethod
    @torch.no_grad()
    def evaluate(model, data):
        """
        Evaluates the model on validation and test sets and returns regression metrics.

        :param model: Trained GNN model.
        :param data: Graph data object.
        """
        model.eval()
        out = model(data.x, data.edge_index)  # output continuous value
        
        num_nodes = data.y.shape[0]  # 原始節點數
        out = out[:num_nodes]        # 僅取原始節點的輸出 (feature to node 時要注意)

        val_mask = data.val_mask[:num_nodes]
        test_mask = data.test_mask[:num_nodes]

        # Extract true & predicted values
        y_true_val = data.y[val_mask].cpu().numpy()
        y_pred_val = out[val_mask].cpu().numpy()
        y_true_test = data.y[test_mask].cpu().numpy()
        y_pred_test = out[test_mask].cpu().numpy()

        # MSE、MAE、R²
        val_mse = mean_squared_error(y_true_val, y_pred_val)
        test_mse = mean_squared_error(y_true_test, y_pred_test)

        val_mae = mean_absolute_error(y_true_val, y_pred_val)
        test_mae = mean_absolute_error(y_true_test, y_pred_test)

        val_r2 = r2_score(y_true_val, y_pred_val)
        test_r2 = r2_score(y_true_test, y_pred_test)

        return {
            "Val_MSE": val_mse,
            "Test_MSE": test_mse,
            "Val_MAE": val_mae,
            "Test_MAE": test_mae,
            "Val_R2": val_r2,
            "Test_R2": test_r2,
        }
