"""
使用 Fast gradient sign method 的方式產生對抗影像
公式：x_adversarial = x_original + ε * sign(∇_x L(θ, x, y))，ε 為擾動強度、L 為損失函數、x 為影像、y 為真實標籤
"""

import torch
import torch.nn as nn

class FGSMAttack:
    def __init__(self, model, epsilon = 0.03):
        self.model = model
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()

    def generate_attack(self, images, labels, device):
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad = True

        # 前向傳播、計算損失
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # 反向傳播、計算梯度
        self.model.zero_grad()
        loss.backward()

        # 生成對抗式樣本
        data_gradient = images.grad.data
        sign_data_gradient = data_gradient.sign()
        perturbed_images = images + self.epsilon * sign_data_gradient
        perturbed_images = torch.clamp(perturbed_images, -1, 1)

        return perturbed_images

if __name__ == '__main__':
    from src.model.resnet import ResNet18
    from src.utils.set_device import *

    device = get_device()
    model = ResNet18(num_classes = 10).to(device)

    attack = FGSMAttack(model, epsilon = 0.03)
    print("FGSM attack initialized successfully.")

    dummy_images = torch.randn(4, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (4,))
    adversarial_images = attack.generate_attack(dummy_images, dummy_labels, device)
    print(f"Generated adversarial images: {adversarial_images.shape}")