import os
import torch
import torchvision.datasets as datasets

class SUN397:
    def __init__(self, preprocess, location='~/data', batch_size=32, num_workers=16):
        # 加载完整数据集
        full_dataset = datasets.SUN397(
            root=location,
            download=True,
            transform=preprocess
        )

        # 定义训练集和测试集的划分比例
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # 使用 random_split 进行划分
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        # 创建 DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        idx_to_class = dict((v, k)
                            for k, v in full_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]
