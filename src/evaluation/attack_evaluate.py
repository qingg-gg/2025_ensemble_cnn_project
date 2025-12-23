"""
評估模型對抗攻擊的效果
"""

import torch
from tqdm import tqdm

class AttackEvaluator:
    def __init__(self, device):
        self.device = device

    # 測試單一模型
    def evaluate_single_model(self, model, attack, data_loader):
        model.eval()
        correct_clean = 0
        correct_adversarial = 0
        total = 0

        pbar = tqdm(data_loader, desc="Single model", leave = False, delay = 0.5)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # 測試原始資料
            with torch.no_grad():
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_clean += predicted.eq(labels).sum().item()

            # 測試攻擊資料
            perturbed_images = attack.generate_attack(images, labels, self.device)
            perturbed_images = perturbed_images.detach()
            with torch.no_grad():
                outputs = model(perturbed_images)
                _, predicted = outputs.max(1)
                correct_adversarial += predicted.eq(labels).sum().item()

            total += images.size(0)

        clean_accuracy = 100 * correct_clean / total
        adversarial_accuracy = 100 * correct_adversarial / total

        return clean_accuracy, adversarial_accuracy

    # 測試集成模型
    def evaluate_ensemble_models(self, ensemble, attack, data_loader):
        correct = 0
        total = 0

        pbar = tqdm(data_loader, desc = "Ensemble models", leave = False, delay = 0.5)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # 測試攻擊資料
            perturbed_images = attack.generate_attack(images, labels, self.device)
            perturbed_images = perturbed_images.detach()
            with torch.no_grad():
                predictions = ensemble.predict(perturbed_images)
                correct += predictions.eq(labels).sum().item()

            total += images.size(0)

        accuracy = 100 * correct / total

        return accuracy

    # 測試魯棒性
    def compare_robustness(self, model, ensemble, attack_list, data_loader):
        results = {
            'epsilons': [],
            'single': [],
            'ensemble': [],
            'improvements': []
        }

        for attack in attack_list:
            _, single_adversarial_accuracy = self.evaluate_single_model(model, attack, data_loader)
            ensemble_adversarial_accuracy = self.evaluate_ensemble_models(ensemble, attack, data_loader)
            improvement = ensemble_adversarial_accuracy - single_adversarial_accuracy

            results['epsilons'].append(attack.epsilon)
            results['single'].append(single_adversarial_accuracy)
            results['ensemble'].append(ensemble_adversarial_accuracy)
            results['improvements'].append(improvement)

            print(f"epsilons: {attack.epsilon:.3f}",
                  f"single: {single_adversarial_accuracy:.2f}",
                  f"ensemble: {ensemble_adversarial_accuracy:.2f}",
                  f"improvements: {improvement:.2f}",
                  sep = " | ")

        average_improvement = sum(results['improvements']) / len(results['improvements'])
        print(f"Average improvement: {average_improvement:.2f}")

        return results