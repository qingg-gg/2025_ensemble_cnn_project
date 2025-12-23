"""
設定要進行訓練的地點
"""

import torch
import torchvision

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def check_environment():
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)

    mps_built = torch.backends.mps.is_built()
    mps_availible = torch.backends.mps.is_available()
    print("mps built:", mps_built)
    print("mps available:", mps_availible)

    device = get_device()
    print("Using device:", device)

if __name__ == "__main__":
    check_environment()