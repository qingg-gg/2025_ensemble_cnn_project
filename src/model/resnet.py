"""
使用 PyTorch 中的 nn.Module 建立 ResNet-18 與 ResNet-34 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet 的基本殘差區塊
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()

        # 第一層卷積
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二層卷積
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )

    # 前向傳播
    def forward(self, x):
        # 主要路徑
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Shortcut
        out += self.shortcut(x)
        out = F.relu(out)

        return out

# ResNet 的基本架構
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 一個卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)

        # 四個殘差層
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)

        # 一個全聯接層
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # 建立殘差層
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

            return nn.Sequential(*layers)

    # 前向傳播
    def forward(self, x):
        # Convolution
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Pooling
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

# ResNet-18 模型
def ResNet18(num_classes = 10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)

# ResNet-34 模型
def ResNet34(num_classes = 10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

if __name__ == '__main__':
    model = ResNet18(num_classes = 10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")