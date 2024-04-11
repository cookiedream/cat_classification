import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size, padding=0, stride=1, with_relu=True):
    """
    創建一個卷積塊，包含卷積層、批量標準化和ReLU激活函數。
    """
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
              nn.BatchNorm2d(out_channels)]
    if with_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class CatClassificationModel(nn.Module):
    def __init__(self, n_class):
        super(CatClassificationModel, self).__init__()
        self.n_class = n_class

        # 使用conv_block來簡化卷積層的創建
        self.conv1 = conv_block(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = conv_block(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 進行特徵維度提升的卷積層
        self.conv6 = conv_block(512, 4096, kernel_size=7, padding=3)
        self.conv7 = conv_block(4096, 4096, kernel_size=1)

        # 分數層和上採樣層
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)

    def forward(self, x):
        # 特徵提取
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x4 = self.pool4(self.conv4(x))
        x5 = self.pool5(self.conv5(x4))

        # 特征维度提升
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        score_fr = self.score_fr(x7)

        score3 = self.score_pool3(x)

        # 在计算up2之前先计算score4
        score4 = self.score_pool4(x4)

        # 使用score_fr的大小作为上采样的目标尺寸
        up2 = F.interpolate(score_fr, size=score4.size()[
                            2:], mode='bilinear', align_corners=False)

        # up2和score4相加得到up4
        up4 = up2 + score4

        # 確保up4的尺寸與score3匹配
        up4 = F.interpolate(up4, size=score3.size()[
                            2:], mode='bilinear', align_corners=False)

        # 將up4和score3相加，並進行最終的上採樣
        out = self.upscore8(up4 + score3)

        return out
