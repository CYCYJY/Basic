# dataset.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset_name="cifar10", data_dir="./data", batch_size=128, num_workers=4, val_split=0.1):
    """
    数据加载中枢：
    1. 下载/读取指定数据集
    2. 将图像 Resize 到 224x224 (满足 ImageNet 骨干网络尺寸)
    3. 将原始训练集划分为 Train 集和 Validation 集 (超参数调优)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 训练集数据增强与预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # 验证集与测试集预处理
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # 数据集加载逻辑扩展
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        full_train = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
        num_classes = 100
    elif dataset_name == "custom":
        # 读取本地 ImageFolder 标准分类格式
        full_train = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=val_transform)
        num_classes = len(full_train.classes)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 划分训练集与验证集 (Question 4 落地)
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保划分为固定集合
    )

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes
if __name__ == "__main__":
    print("开始加载数据集...", flush=True)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset_name="cifar10",
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        val_split=0.1,
    )
    print("num_classes:", num_classes, flush=True)
    print("训练集 batch 数:", len(train_loader), flush=True)
    print("验证集 batch 数:", len(val_loader), flush=True)
    print("测试集 batch 数:", len(test_loader), flush=True)

    for imgs, labels in train_loader:
        print("一个 batch 的 shape:", imgs.shape, flush=True)
        print("labels[:8]:", labels[:8], flush=True)
        break

    print("数据加载测试完成。", flush=True)
    #