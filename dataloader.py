import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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
            print(f"{breed}: {images_count} 張圖片")

        print(f"Total {len(self.classes)} classes")  # 打印品種總數和部分品種名稱
        print("breed", self.classes[:20])  # 僅打印前五個品種名稱作為示例
        print(f"Total {len(self.images)} img")  # 打印圖片總數

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 創建 Dataset
data_dir = 'data'  # 這裡填入你的資料夾根路徑
cats_dataset = CatsDataset(data_dir=data_dir, transform=transform)

# 創建 DataLoader
data_loader = DataLoader(cats_dataset, batch_size=32, shuffle=True)
