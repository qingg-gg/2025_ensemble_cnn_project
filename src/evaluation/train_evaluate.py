"""
評估模型與集成模型的預測準確率
"""

import torch
import torch.nn as nn
from tqdm import tqdm

class TrainEvaluator:
    def __init__(self, device):
        self.device = device
        self.default_criterion = nn.CrossEntropyLoss()

    # 評估模型
    def evaluate_model(self, model, data_loader, criterion = None, return_loss = False):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        if criterion is None:
            criterion = self.default_criterion

        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating single model", leave = False, delay = 0.5)

            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        loss = running_loss / len(data_loader)
        accuracy = 100.0 * correct / total

        if return_loss:
            return loss, accuracy
        else:
            return accuracy

    # 評估集成模型
    def evaluate_ensemble(self, ensemble, data_loader, return_details = False):
        correct = 0
        total = 0

        per_model_correct = None
        if return_details:
            per_model_correct = [0] * len(ensemble.models)

        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating ensemble model", leave = False, delay = 0.5)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predictions = ensemble.predict(inputs)
                correct += predictions.eq(labels).sum().item()
                total += labels.size(0)

                if return_details:
                    for i, model in enumerate(ensemble.models):
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        per_model_correct[i] += predicted.eq(labels).sum().item()

        ensemble_accuracy = 100.0 * correct / total
        if return_details:
            per_model_acc = [100.0 * c / total for c in per_model_correct]
            return ensemble_accuracy, per_model_acc

        return ensemble_accuracy

    def compare_models(self, models, data_loader, model_names = None):
        if model_names is None:
            model_names = [f"Model {i + 1}" for i in range(len(models))]

        results = {}
        for model, name in zip(models, model_names):
            accuracy = self.evaluate_model(model, data_loader)
            results[name] = accuracy
            print(f"{name:<10}: {accuracy:.2f}%")

        best_model = max(results, key = results.get)
        print(f"Best model: {best_model} | Accuracy: {results[best_model]:.2f}%")

        return results