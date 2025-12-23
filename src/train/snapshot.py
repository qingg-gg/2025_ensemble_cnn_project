"""
透過 PyTorch 中的 optim 訓練 ResNet 模型
共訓練 180 回合，將其以 60 個為一單位分成週期，每個週期結束時儲存當下的模型權重（Snapshot），共 3 個
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
from tqdm import tqdm

class SnapshotEnsembleTrainer:
    def __init__(self, model_fn, device, save_dir = None, num_snapshots = 3):
        self.model_fn = model_fn
        self.model = model_fn().to(device)
        self.device = device
        self.save_dir = save_dir
        self.num_snapshots = num_snapshots
        self.snapshots = []

    # Train（一回合內行為）
    def _train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc = "Training", leave = False, delay = 0.5)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # 前向傳播
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            # 反向傳播
            loss.backward()
            optimizer.step()

            # 統計
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        pbar.close()
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        return train_loss, train_accuracy

    # 整個訓練流程（Train + Snapshot）
    def train_with_snapshots(self, train_loader, num_epochs = 180, lr = 0.1, momentum = 0.9, weight_decay = 5e-4):
        # Loss function 與優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

        # 週期與學習曲線
        epochs_per_cycle = num_epochs // self.num_snapshots
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = epochs_per_cycle, T_mult= 1, eta_min = 0.0001)

        snapshot_count = 0
        training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "learning_rate": [],
            'snapshot_epochs': []
        }
        for epoch in range(num_epochs):
            # 訓練
            train_loss, train_accuracy = self._train_epoch(train_loader, optimizer, criterion)
            scheduler.step(epoch + 1)
            current_learning_rate = optimizer.param_groups[0]['lr']

            training_history["train_loss"].append(train_loss)
            training_history["train_accuracy"].append(train_accuracy)
            training_history["learning_rate"].append(current_learning_rate)

            # 每 60 個回合儲存一次 Snapshot
            if (epoch + 1) % epochs_per_cycle == 0 and snapshot_count < self.num_snapshots:
                snapshot_path = os.path.join(self.save_dir, f"snapshot_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, snapshot_path)
                self.snapshots.append(snapshot_path)
                training_history["snapshot_epochs"].append(epoch + 1)
                snapshot_count += 1

            # 印出結果
            tqdm.write(f"Epoch: {epoch + 1} / {num_epochs} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Train accuracy: {train_accuracy:.4f} | "
                  f"Learning rate: {current_learning_rate:.4f}"
                       )

        print(f"Training completed.")
        return training_history

    # 載入所有模型（For ensemble)
    def load_snapshot(self):
        models = []
        for snapshot_path in self.snapshots:
            model_copy = self.model_fn().to(self.device)
            checkpoint = torch.load(snapshot_path, map_location = self.device)
            model_copy.load_state_dict(checkpoint['model_state_dict'])
            model_copy.eval()
            models.append(model_copy)

        return models

if __name__ == "__main__":
    from src.model.resnet import ResNet18
    from src.utils.set_device import *

    device = get_device()
    model = ResNet18(num_classes = 10)
    trainer = SnapshotEnsembleTrainer(
        model_fn = lambda: ResNet18(num_classes = 10),
        device = device,
        num_snapshots = 3)

    print("Trainer initialized successfully.")