import torch
import inspect

def extract_model_init_args(model):
    """
    從模型實例自動提取 __init__() 中的初始化參數。
    假設模型在 __init__ 中有 self.xxx = xxx 的寫法。
    """
    sig = inspect.signature(model.__class__.__init__)
    arg_names = [p for p in sig.parameters if p != 'self']
    config = {}
    for name in arg_names:
        if hasattr(model, name):
            config[name] = getattr(model, name)
    return config


def save_model_and_config(model, model_path, config_path, training_params=None):
    """
    儲存模型權重與對應的 config（含初始化參數 + 訓練參數）。
    """
    torch.save(model.state_dict(), model_path)

    config = extract_model_init_args(model)
    if training_params:
        config.update(training_params)  # 加上 epochs / lr 等資訊

    torch.save(config, config_path)
    print(f"\nModel saved at {model_path}")
    print(f"Config saved at {config_path}")
