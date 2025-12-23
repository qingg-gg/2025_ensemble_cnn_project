"""
集成模型成功對抗案例
輸出結合「原始影像、攻擊影像、擾動大小、預測結果」的表格
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.model.resnet import ResNet18, ResNet34
from src.attack.fgsm import FGSMAttack
from src.train.ensemble import EnsembleDefense

class AdversarialExampleVisualizer:
    def __init__(self, model_name = 'resnet18', snapshot_dir = None, device = None):
        self.model_name = model_name
        self.device = device
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.single_model = self._load_single_model(snapshot_dir)
        self.ensemble_model = self._load_ensemble_model(snapshot_dir)

    # 載入單一模型
    def _load_single_model(self, snapshot_dir):
        model_fn = ResNet18 if self.model_name == 'resnet18' else ResNet34
        model = model_fn(num_classes = 10).to(self.device)

        snapshots = sorted((Path(snapshot_dir) / self.model_name).glob("snapshot_*.pth"))
        if snapshots:
            snapshot_path = snapshots[-1]
        else:
            raise FileNotFoundError(f"No snapshot found of {self.model_name}")

        checkpoint = torch.load(snapshot_path, map_location = self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model

    # 載入集成模型
    def _load_ensemble_model(self, snapshot_dir):
        model_fn = ResNet18 if self.model_name == 'resnet18' else ResNet34

        snapshot_paths = sorted((Path(snapshot_dir) / self.model_name).glob('snapshot_*.pth'))
        ensemble_models = []
        for path in snapshot_paths:
            model = model_fn(num_classes = 10).to(self.device)
            checkpoint = torch.load(path, map_location = self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            ensemble_models.append(model)

        return EnsembleDefense(ensemble_models, self.device)

    # 尋找成功範例
    def find_successful_defense_examples(self, data_loader, epsilon = 0.03, num_examples = 10):
        attack = FGSMAttack(self.single_model, epsilon = epsilon)
        examples = []

        for images, labels in data_loader:
            if len(examples) >= num_examples:
                break

            # 載入影像、製造擾動
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                single_output_clean = self.single_model(images)
                _, single_prediction_clean = single_output_clean.max(1)
                ensemble_prediction_clean = self.ensemble_model.predict(images)

            perturbed_images = attack.generate_attack(images, labels, self.device)
            with torch.no_grad():
                single_output_adversarial = self.single_model(perturbed_images)
                _, single_prediction_adversarial = single_output_adversarial.max(1)
                ensemble_prediction_adversarial = self.ensemble_model.predict(perturbed_images)

            # 進行預測、儲存範例
            for i in range(len(labels)):
                if len(examples) >= num_examples:
                    break

                if (single_prediction_clean[i] == labels[i] and ensemble_prediction_clean[i] == labels[i] and
                single_prediction_adversarial[i] != labels[i] and ensemble_prediction_adversarial[i] == labels[i]):
                    example = {
                        'original_image': images[i].cpu(),
                        'adversarial_image': perturbed_images[i].cpu(),
                        'true_label': labels[i].item(),
                        'single_prediction_clean': single_prediction_clean[i].item(),
                        'single_prediction_adversarial': single_prediction_adversarial[i].item(),
                        'ensemble_prediction_clean': ensemble_prediction_clean[i].item(),
                        'ensemble_prediction_adversarial': ensemble_prediction_adversarial[i].item(),
                        'epsilon': epsilon
                    }
                    examples.append(example)

                    print(f"Example {len(examples)} / {num_examples}:")
                    print(f"True label: {self.classes[example['true_label']]}",
                          f"Single prediction label: {self.classes[single_prediction_adversarial[i]]}",
                          f"Ensemble prediction label: {self.classes[ensemble_prediction_adversarial[i]]}",
                          sep = '\n')

        return examples

    # 展示案例
    def visualize_example(self, example, save_path = None):
        fig = plt.figure(figsize = (16, 5))

        # 載入影像
        def denormalize(img):
            if img.requires_grad:
                img = img.detach()
            img = img.cpu().numpy().transpose(1, 2, 0)
            img = img * 0.5 + 0.5
            return np.clip(img, 0, 1)

        original = denormalize(example['original_image'])
        adversarial = denormalize(example['adversarial_image'])
        perturbation = adversarial - original

        true_label = self.classes[example['true_label']]
        single_prediction_adversarial = self.classes[example['single_prediction_adversarial']]
        ensemble_prediction_adversarial = self.classes[example['ensemble_prediction_adversarial']]

        # 繪製圖表
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(original)
        ax1.set_title(f"Original Image\nTrue Label: {true_label}", fontsize = 12, fontweight = 'bold')
        ax1.axis('off')

        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(adversarial)
        ax2.set_title(f"Adversarial Image\nEpsilon: {example['epsilon']:.3f}", fontsize = 12, fontweight = 'bold')
        ax2.axis('off')

        ax3 = plt.subplot(1, 4, 3)
        perturbation_enhanced = np.abs(perturbation) * 10
        ax3.imshow(perturbation_enhanced)
        ax3.set_title("Perturbation\n(Enhanced 10 times)", fontsize = 12, fontweight = 'bold')
        ax3.axis('off')

        ax4 = plt.subplot(1, 4, 4)
        ax4.axis('off')
        result_text = (
            f"True Label: {true_label}\n\n"
            f"Single Model:\n"
            f"1. Original Image   : {self.classes[example['single_prediction_clean']]} -> Correct\n"
            f"2. Adversarial Image: {single_prediction_adversarial} -> Wrong\n\n"
            f"Ensemble Model:\n"
            f"1. Original Image   : {self.classes[example['ensemble_prediction_clean']]} -> Correct\n"
            f"2. Adversarial Image: {ensemble_prediction_adversarial} -> Correct"
        )

        ax4.text(0.1, 0.5, result_text, fontsize = 11, verticalalignment = 'center',
                 bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5),
                 family = 'monospace')
        plt.tight_layout()

        # 儲存影像
        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            plt.close(fig)

        return fig


def main():
    import argparse
    from src.utils.set_device import get_device
    from src.utils.data_loader import dataloader

    # 設定輸入參數
    parser = argparse.ArgumentParser(description = 'Visualize adversarial examples.')
    parser.add_argument('--model', type = str, default = 'resnet18', choices = ['resnet18', 'resnet34'], help = 'Model to use.')
    parser.add_argument('--snapshot_dir', type = str, default = './snapshots', help = 'Path to the snapshot directory.')
    parser.add_argument('--epsilon', type = float, default = 0.03, help = 'Epsilon of adversarial example.')
    parser.add_argument('--num_examples', type = int, default = 10, help = 'Number of examples to find.')
    parser.add_argument('--output_dir', type = str, default = './results', help = 'Path to the output directory.')
    parser.add_argument('--data_dir', type = str, default = './data', help = 'Path to the data directory.')

    args = parser.parse_args()

    print("Start running.")

    # 建立視覺化器
    device = get_device()
    visualizer = AdversarialExampleVisualizer(
        model_name=args.model,
        snapshot_dir=args.snapshot_dir,
        device=device
    )

    # 尋找案例
    _, testloader = dataloader(data_dir=args.data_dir, batch_size=100, num_workers=0)
    examples = visualizer.find_successful_defense_examples(
        testloader,
        epsilon=args.epsilon,
        num_examples=args.num_examples
    )
    if not examples:
        print("No defense examples found.")
        return

    # 儲存影像
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    count = 1
    for example in range(len(examples)):
        visualizer.visualize_example(
            examples[example],
            save_path = output_dir / f'example_{args.model}_{count}.png'
        )
        count += 1

    print("Finished.")

if __name__ == "__main__":
    main()