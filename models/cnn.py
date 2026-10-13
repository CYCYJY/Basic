import torch.nn as nn

'''Convolutional Neural Network'''
class CNN(nn.Module):
    def __init__(self, train_shape, category):
        super(CNN, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(512*train_shape[-1], category)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

    # example：dataset uci --model cnn
    # ==================================================  【张量转换】  ===================================================

    # x_train_tensor shape: torch.Size([7352, 1, 128, 9])        (number, depth, hidth/Timesteps, width/channels)
    # x_test_tensor shape: torch.Size([2947, 1, 128, 9])
    # Category num: 6
    # nn.Conv2d(1, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  参数:形状 kernel只包含时间步, 通道间隔离。
    # self.ada_pool = nn.AdaptiveAvgPool2d(output_size=(1, train_shape[-1])) 参数:池化后输出尺寸 时间步压缩到1, 通道间隔离。