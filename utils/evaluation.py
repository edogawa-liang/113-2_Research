import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

class Evaluator:
    """
    Evaluates a trained GNN model and computes various performance metrics.
    """

    @staticmethod
    def evaluate(model, data):
        """
        Evaluates the model on validation and test sets and returns performance metrics.
        """
        out, pred = model.predict(data) 

        # Compute accuracy
        acc_val = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        acc_test = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

        # Extract true & predicted values
        y_true_val = data.y[data.val_mask].cpu().numpy()
        y_pred_val = pred[data.val_mask].cpu().numpy()
        y_true_test = data.y[data.test_mask].cpu().numpy()
        y_pred_test = pred[data.test_mask].cpu().numpy()

        # Compute AUC (only if multiple classes exist)
        val_auc = roc_auc_score(y_true_val, y_pred_val, multi_class="ovr") if len(set(y_true_val)) > 1 else 0
        test_auc = roc_auc_score(y_true_test, y_pred_test, multi_class="ovr") if len(set(y_true_test)) > 1 else 0

        # Compute precision, recall, F1-score
        val_precision = precision_score(y_true_val, y_pred_val, average="macro", zero_division=0)
        test_precision = precision_score(y_true_test, y_pred_test, average="macro", zero_division=0)

        val_recall = recall_score(y_true_val, y_pred_val, average="macro", zero_division=0)
        test_recall = recall_score(y_true_test, y_pred_test, average="macro", zero_division=0)

        val_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)
        test_f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division=0)

        # Compute confusion matrix
        cm = confusion_matrix(y_true_test, y_pred_test).tolist()

        return {
            "Loss": float(F.cross_entropy(out[data.train_mask], data.y[data.train_mask])),
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
            "CM": cm
        }