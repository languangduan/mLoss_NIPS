

import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np

pretrained_checkpoint = f'checkpoints/ViT-L-14/zeroshot.pt'
# 加载模型的 state_dict
pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

# 打印每一层的名称和形状
for name, param in pretrained_state_dict.items():
    print(f"Parameter Name: {name}, Shape: {param.shape}")



