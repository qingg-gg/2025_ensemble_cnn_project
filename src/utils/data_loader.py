"""
使用 PyTorch 中的 Dataloader 載入 CIFAR-10 資料集
此資料集已經為使用者切好「Train」與「Test」兩部分
"""

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.utils.set_device import *

def dataloader(
        data_dir: str = None,
        batch_size: int = 128,
        num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    # 針對單一影像的處理流程
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ])

    # 取得資料集
    train_set = torchvision.datasets.CIFAR10(
        root = data_dir, train = True, download = True, transform = transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root = data_dir, train = False, download = True, transform = transform
    )

    # 建立 Dataloader：要如何取得 Dataset 中的資料
    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )

    return train_loader, test_loader

if __name__ == "__main__":
    device = get_device()
    print("Using device: ", device)

    train_loader, test_loader = dataloader()
    images, labels = next(iter(train_loader))
    print("Train batch images: ", images.shape)
    print("Train batch labels: ", labels.shape)
    print("Labels example: ", labels[:10].tolist())