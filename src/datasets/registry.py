import sys
import inspect
import random
import torch
import copy

import os
from torch.utils.data.dataset import random_split
from torchvision import datasets
from torchvision.datasets import Flowers102, FashionMNIST

from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT
from src.datasets.gtsrb import GTSRB
from src.datasets.imagenet import ImageNet
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.stl10 import STL10
from src.datasets.svhn import SVHN
from src.datasets.sun397 import SUN397

# registry = {
#     name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
# }

import sys
import torch
from torch.utils.data import DataLoader, random_split

# 假设你已经有 TV_DATASETS 字典
TV_DATASETS = {
    'Flowers102':   (datasets.Flowers102, 20, True),
    'Caltech101':   (datasets.Caltech101, 10, True),
    'Food101':      (datasets.Food101, 15, True),
    'FashionMNIST': (datasets.FashionMNIST, 5, False),
    'STL10':        (datasets.STL10, 8, False),
    'CIFAR10':      (datasets.CIFAR10, 8, False),
    'CIFAR100':     (datasets.CIFAR100, 8, False),
}


# 哪些数据集有官方split
SPLIT_DATASETS = {
    'Flowers102': ['train', 'val', 'test'],
    'Food101': ['train', 'test'],
    'Caltech101': ['train', 'test'],
    # 你可以继续补充
}

flowers102_classnames = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily", "common tulip", "wild rose"
]

prj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_location = os.path.join(prj_root, "datasets")
# 1. 定义通用封装类
class TorchvisionDatasetWrapper:
    def __init__(self, dataset_cls, preprocess, location=data_location,
                 batch_size=32, num_workers=16, train_split=0.8, seed=0, has_official_split=False, **kwargs):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 1. 有官方split的数据集
        if has_official_split:
            # Flowers102需要特殊处理
            if dataset_cls is datasets.Flowers102:
                self.train_dataset = dataset_cls(root=location, split='train', download=False, transform=preprocess, **kwargs)
                self.val_dataset   = dataset_cls(root=location, split='test', download=False, transform=preprocess, **kwargs)
                self.test_dataset  = dataset_cls(root=location, split='val', download=False, transform=preprocess, **kwargs)
                self.classnames = flowers102_classnames
            else:
                # 尝试加载train/val/test
                self.train_dataset = dataset_cls(root=location, split='train', download=False, transform=preprocess, **kwargs)
                # Caltech101没有val split，这里用test作为val
                try:
                    self.val_dataset   = dataset_cls(root=location, split='val', download=False, transform=preprocess, **kwargs)
                except:
                    self.val_dataset = None
                try:
                    self.test_dataset  = dataset_cls(root=location, split='test', download=False, transform=preprocess, **kwargs)
                except:
                    self.test_dataset = None

                # 处理类别名
                idx_to_class = getattr(self.train_dataset, 'classes', None)
                if idx_to_class is not None:
                    self.classnames = list(idx_to_class)
                elif hasattr(self.train_dataset, 'class_to_idx'):
                    idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
                    self.classnames = [idx_to_class[i] for i in range(len(idx_to_class))]
                else:
                    self.classnames = None

            # DataLoader
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            # val_loader
            if self.val_dataset is not None:
                self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            else:
                self.val_loader = None

            # test_loader 修正逻辑：如果 test split 没 label或 test_dataset 为 None，则 fallback 用 val_loader
            # Flowers102 test split 没 label，Caltech101 没有 val split
            if self.test_dataset is not None and len(getattr(self.test_dataset, 'targets', [])) > 0:
                self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            elif self.val_dataset is not None:
                # fallback 用 val split 做 test
                self.test_dataset = self.val_dataset
                self.test_loader = self.val_loader
            else:
                # fallback 用 train split 做 test（极端情况）
                self.test_dataset = self.train_dataset
                self.test_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 2. 没有官方 split 的
        else:
            # 兼容FashionMNIST等
            if hasattr(dataset_cls, 'train') and 'train' in inspect.signature(dataset_cls).parameters:
                full_dataset = dataset_cls(root=location, train=True, download=False, transform=preprocess, **kwargs)
                test_dataset = dataset_cls(root=location, train=False, download=False, transform=preprocess, **kwargs)
            else:
                full_dataset = dataset_cls(root=location, download=False, transform=preprocess, **kwargs)
                test_dataset = None

            train_size = int(train_split * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
            )
            self.test_dataset = test_dataset

            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            if self.test_dataset is not None:
                self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            else:
                # fallback 用 val split 做 test
                self.test_dataset = self.val_dataset
                self.test_loader = self.val_loader

            # 类别名
            idx_to_class = getattr(full_dataset, 'classes', None)
            if idx_to_class is not None:
                self.classnames = list(idx_to_class)
            elif hasattr(full_dataset, 'class_to_idx'):
                idx_to_class = dict((v, k) for k, v in full_dataset.class_to_idx.items())
                self.classnames = [idx_to_class[i] for i in range(len(idx_to_class))]
            else:
                self.classnames = None




# 2. 动态注册
import sys


def register_tv_datasets(TV_DATASETS):
    current_module = sys.modules[__name__]
    for name, (dataset_cls, default_epoch, has_official_split) in TV_DATASETS.items():
        # 创建一个工厂函数来避免闭包问题
        def create_init(cls,  has_official_split):
            def __init__(self, preprocess, location=data_location, batch_size=32, num_workers=16, **kwargs):
                TorchvisionDatasetWrapper.__init__(
                    self, cls, preprocess, location, batch_size, num_workers,has_official_split=has_official_split, **kwargs
                )

            return __init__

        wrapper_class = type(
            name,
            (TorchvisionDatasetWrapper,),
            {'__init__': create_init(dataset_cls,  has_official_split)}
        )
        setattr(current_module, name, wrapper_class)


# 3. 在主程序入口调用
register_tv_datasets(TV_DATASETS)

def build_registry():
    return {
        name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    }

registry = build_registry()

def check_registry():
    registry = build_registry()
    print("Registered datasets:")
    for name, cls in registry.items():
        if name in TV_DATASETS:
            expected_cls = TV_DATASETS[name][0]
            print(f"  {name}: {cls.__name__} (Expected to use {expected_cls.__name__})")
        else:
            print(f"  {name}: {cls.__name__}")

# 调用检查函数
check_registry()


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044


    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16, val_fraction=0.1, max_val_samples=5000):
    print(dataset_name)
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    return dataset