import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator:
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

        # Extract true & predicted values
        y_true_val = data.y[data.val_mask].cpu().numpy()
        y_pred_val = out[data.val_mask].cpu().numpy()

        y_true_test = data.y[data.test_mask].cpu().numpy()
        y_pred_test = out[data.test_mask].cpu().numpy()

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
