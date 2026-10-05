# models/alexnet.py
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    仅通过 nn.Module 纯手写搭建的 AlexNet 网络
    """

    def __init__(self, num_classes=10, dropout=0.5):
        super(AlexNet, self).__init__()

        # 1. 特征提取层 (Convolutional Backbone)
        self.features = nn.Sequential(
            # Conv1: 输入 3 通道 -> 输出 64 通道，11x11 卷积核，步长 4
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: 输出 192 通道，5x5 卷积核
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: 输出 384 通道
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 输出 256 通道
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 输出 256 通道
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 2. 自适应全局池化 (保证输出大小固定为 6x6，增强不同分辨率输入的兼容性)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 3. 全连接分类头 (Classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 从 channel 维往后展平成 1D 向量
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Kaiming / Normal 权重初始化规范"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)