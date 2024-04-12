import torch
import torch.nn as nn


class CatClassificationModel(nn.Module):
    def __init__(self, n_class):  # 添加 n_class 作为参数
        super(CatClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # 注意，这里的全连接层输入尺寸取决于你的数据预处理和网络结构
        self.fc1 = nn.Linear(2097152, 128)  # 假设这是根据实际特
        self.fc2 = nn.Linear(128, n_class)  # 使用传入的 n_class 设置输出层

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # print(x.shape)  # 打印卷积层输出的尺寸，以核实
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print(CatClassificationModel)
