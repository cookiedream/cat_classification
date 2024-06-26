import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np
import torchvision

random.seed(42)


with open('train.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


class CatsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): 資料夾路徑，包含所有貓的品種資料夾。
            transform (callable, optional): 一個可選的轉換函數，用於對樣本進行處理。
            self.classes: 一個列表，包含所有貓的品種名稱。
            self.images: 一個列表，包含所有圖片的路徑。
            self.labels: 一個列表，包含所有圖片的標籤。
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.classes.sort()
        self.images = []
        self.labels = []

        total_images_count = 0  # 總圖片數量計數器

        for idx, breed in enumerate(self.classes):
            breed_dir = os.path.join(data_dir, breed)
            breed_images = [os.path.join(breed_dir, img)
                            for img in os.listdir(breed_dir)]
            images_count = len(breed_images)  # 計算當前品種下的圖片數量
            total_images_count += images_count  # 累加到總圖片數量
            self.images.extend(breed_images)
            self.labels.extend([idx] * images_count)

            # 打印當前品種和其圖片數量
            # print(f"{breed}: {images_count} 張圖片")

        # print(f"Total {len(self.classes)} classes")  # 打印品種總數和部分品種名稱
        # print("breed", self.classes[:20])  # 僅打印前五個品種名稱作為示例
        # print(f"Total {len(self.images)} img")  # 打印圖片總數

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 資料轉換，可以根據需求自定義
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])
"""
這裡是使用網路上預設圖片處理的方法，這裡的處理方法包括：

1. `transforms.Resize(256)`：将图像大小调整为 256x256 像素。这通常是为了确保所有输入图像有统一的尺寸，以便于网络处理。
2. `transforms.ColorJitter()`：随机改变图像的亮度、对比度和饱和度。这有助于模型泛化，因为真实世界中的图像可能在这些属性上有所不同。
3. `transforms.RandomCrop(224)`：从图像中随机裁剪出 224x224 像素的区域。这是一种形式的数据增强，通过从原始图像的不同位置裁剪出图像，模型可以学习到更多关于图像不同部分的信息。
4. `transforms.RandomHorizontalFlip()`：以 50% 的概率水平翻转图像。这也是一种数据增强技术，有助于模型学习对于方向变化不敏感。
5. `transforms.Resize(128)`：将图像再次缩放到 128x128 像素。这可能是为了符合特定的网络输入要求，或者减少计算负担。
6. `transforms.ToTensor()`：将 PIL 图像或 NumPy ndarray 转换成 FloatTensor，并且把图像的像素值范围从 [0, 255] 归一化到 [0.0, 1.0]。这一步是准备图像数据喂入神经网络必须的步骤，因为 PyTorch 中的张量是处理图像的基本单位。
"""

# 創建 Dataset
data_dir = 'data'  # 這裡填入你的資料夾根路徑
cats_dataset = CatsDataset(data_dir=data_dir, transform=transform)

# 切分訓練和驗證資料
dataset_size = len(cats_dataset)
train_size = dataset_size
# 計算驗證集的大小
val_size = int(dataset_size * config["TRAINING"]["VAL_RATIO"])
# 計算訓練集的大小
train_size = dataset_size - val_size

# 使用計算出的大小分割數據集
train_dataset, val_dataset = random_split(cats_dataset, [train_size, val_size])


# 創建 DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=config["TRAINING"]["BATCH_SIZE"], shuffle=config["TRAINING"]["SHUFFLE"])
val_loader = DataLoader(
    val_dataset, batch_size=config["TRAINING"]["BATCH_SIZE"], shuffle=config["TRAINING"]["SHUFFLE"])


# samples, labels = next(iter(train_loader))
# plt.figure(figsize=(16, 24))
# grid_imgs = torchvision.utils.make_grid(samples[:24])
# np_grid_imgs = grid_imgs.numpy()
# # in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
# plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))
# plt.savefig('./fig/fig.png')
