"""
每個 Snapshot 分別進行預測，並以得票率最高者最為最後預測結果
"""

import torch

class EnsembleDefense:
    def __init__(self, models, device):
        self.models = models
        self.device = device

    # 集成預測
    def predict(self, images):
        images = images.to(self.device)
        outputs = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(images)
                outputs.append(output)

        ensemble_output = torch.stack(outputs).mean(dim = 0)
        _, predicted = ensemble_output.max(1)

        return predicted

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from src.utils.set_device import *

    device = get_device()

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32 * 32 * 3, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

    models = [
        DummyModel().to(device),
        DummyModel().to(device),
        DummyModel().to(device)
    ]

    ensemble = EnsembleDefense(models, device)
    images = torch.randn(1, 3, 32, 32)
    predictions = ensemble.predict(images)
    print("Ensemble predictions: ", predictions.item())