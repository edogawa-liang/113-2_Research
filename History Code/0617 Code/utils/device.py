# utils/device.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using device: {DEVICE}")
