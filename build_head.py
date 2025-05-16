import os
import time
import torch
import argparse

import tqdm

from src import utils
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate, eval_single_dataset
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head

import src.datasets as datasets

# Number of classes for new datasets
NUM_CLASSES_DICT = {
    'Flowers102': 102,
    'Caltech101': 102,  # Caltech101 contains 102 classes (including background)
    'FashionMNIST': 10,
    'Food101': 101,
    'STL10': 10,
    'CIFAR100': 100,
}

# Epochs for each dataset
TV_DATASETS = {
    'Flowers102': 20,
    'Caltech101': 10,
    'FashionMNIST': 5,
    'Food101': 15,
    'STL10': 8,
    'CIFAR100': 8,
}


def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)
    headdir = os.path.join(args.save, f'head_{train_dataset}.pt')

    # Ensure save directory exists
    os.makedirs(ckpdir, exist_ok=True)

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    # Ensure num_classes is set in args
    if train_dataset in NUM_CLASSES_DICT:
        args.num_classes = NUM_CLASSES_DICT[train_dataset]
        print(f"Using predefined num_classes={args.num_classes} for {train_dataset}")

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)
    torch.save(model.classification_head, headdir)

if __name__ == '__main__':
    data_location = 'datasets'
    models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']

    # New datasets to train
    datasets_to_train = ['FashionMNIST', 'Food101', 'Flowers102', 'STL10', 'CIFAR100']

    epochs = {
        'Flowers102': 20,
        'Caltech101': 10,
        'FashionMNIST': 1,
        'Food101': 15,
        'STL10': 8,
        'CIFAR100': 8,
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'ImageNet': 4
    }

    for model in models:
        for dataset in datasets_to_train:
            try:
                print('=' * 100)
                print(f'Finetuning {model} on {dataset}')
                print('=' * 100)
                args = parse_arguments()
                args.lr = 1e-4
                args.epochs = epochs[dataset]
                args.data_location = data_location
                args.train_dataset = dataset
                args.batch_size = 128
                args.model = model
                args.save = f'checkpoints/{model}'

                # Add num_classes parameter
                if dataset in NUM_CLASSES_DICT:
                    args.num_classes = NUM_CLASSES_DICT[dataset]

                finetune(args)
            except Exception as e:
                print(f'error in dataset {dataset}, {e}')
                continue
