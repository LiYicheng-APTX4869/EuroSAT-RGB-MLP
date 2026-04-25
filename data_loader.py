"""
data_loader.py — 数据加载与预处理模块

EuroSAT_RGB 数据集加载器：
- 遍历10个类别子目录，读取所有 .jpg 图像
- 展平为 12288 维向量（64×64×3）
- 零均值单位方差归一化
- 划分 train/val/test
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# EuroSAT_RGB 的10个类别（与文件夹名对应）
CLASSES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake',
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


class EuroSATDataLoader:
    """
    EuroSAT_RGB 数据集加载器

    参数：
        data_dir (str): EuroSAT_RGB 文件夹路径
        val_ratio (float): 验证集比例（从训练集中划分）
        test_ratio (float): 测试集比例
        seed (int): 随机种子，保证可复现
        image_size (int): 图像尺寸（默认64，原始大小）
    """

    def __init__(self, data_dir, val_ratio=0.15, test_ratio=0.15, seed=42, image_size=64):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.image_size = image_size
        self.mean = None  # 训练集均值（归一化用）
        self.std = None   # 训练集标准差（归一化用）

    def load_data(self):
        """
        加载所有图像并划分数据集

        返回：
            X_train, y_train: 训练集特征和标签
            X_val, y_val:     验证集特征和标签
            X_test, y_test:   测试集特征和标签

        X 的形状: (N, 12288)，dtype=float64
        y 的形状: (N,)，dtype=int64，值域 [0, 9]
        """
        print("正在加载 EuroSAT_RGB 数据集...")
        X, y = self._load_all_images()
        print(f"共加载 {len(X)} 张图像，形状 {X.shape}")

        # 划分测试集
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=y  # 保持各类别比例
        )

        # 从剩余数据中划分验证集
        val_ratio_adjusted = self.val_ratio / (1.0 - self.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_adjusted,
            random_state=self.seed,
            stratify=y_trainval
        )

        # 用训练集均值/标准差归一化（避免数据泄露）
        X_train, X_val, X_test = self._normalize(X_train, X_val, X_test)

        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _load_all_images(self):
        """遍历所有类别文件夹，读取图像并展平"""
        X_list = []
        y_list = []

        for class_name in CLASSES:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"警告：找不到类别目录 {class_dir}，跳过")
                continue

            label = CLASS_TO_IDX[class_name]
            img_files = [f for f in os.listdir(class_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            print(f"  {class_name}: {len(img_files)} 张图像")

            for fname in img_files:
                fpath = os.path.join(class_dir, fname)
                try:
                    img = Image.open(fpath).convert('RGB')
                    # 确保尺寸为 64×64
                    if img.size != (self.image_size, self.image_size):
                        img = img.resize((self.image_size, self.image_size),
                                         Image.BILINEAR)
                    arr = np.array(img, dtype=np.float64)  # (64, 64, 3)
                    flat = arr.flatten()  # (12288,)
                    X_list.append(flat)
                    y_list.append(label)
                except Exception as e:
                    print(f"    警告：无法读取 {fpath}：{e}")

        X = np.array(X_list, dtype=np.float64)  # (N, 12288)
        y = np.array(y_list, dtype=np.int64)     # (N,)
        return X, y

    def _normalize(self, X_train, X_val, X_test):
        """
        零均值单位方差归一化

        用训练集统计量归一化所有集合（防止数据泄露）
        先除以255 将像素缩放到 [0,1]，再做标准化
        """
        # 先归一化到 [0, 1]
        X_train = X_train / 255.0
        X_val   = X_val   / 255.0
        X_test  = X_test  / 255.0

        # 计算训练集均值和标准差（逐特征）
        self.mean = X_train.mean(axis=0)           # (12288,)
        self.std  = X_train.std(axis=0) + 1e-8    # 防止除零

        X_train = (X_train - self.mean) / self.std
        X_val   = (X_val   - self.mean) / self.std
        X_test  = (X_test  - self.mean) / self.std

        return X_train, X_val, X_test

    def save_stats(self, path):
        """保存归一化统计量，供测试时使用"""
        np.savez(path, mean=self.mean, std=self.std,
                 classes=CLASSES)
        print(f"归一化统计量已保存至 {path}")

    def load_stats(self, path):
        """加载归一化统计量"""
        data = np.load(path, allow_pickle=True)
        self.mean = data['mean']
        self.std  = data['std']
        print(f"归一化统计量已从 {path} 加载")


def get_class_names():
    """返回类别名称列表"""
    return CLASSES.copy()


if __name__ == '__main__':
    # 快速测试
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './EuroSAT_RGB'
    loader = EuroSATDataLoader(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()
    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print(f"标签范围: [{y_train.min()}, {y_train.max()}]")
    print(f"训练集均值: {X_train.mean():.4f}, 标准差: {X_train.std():.4f}")
