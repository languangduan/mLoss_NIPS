import torch
import torch.nn as nn
import torch.optim as optim
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np

# Config
datasets = ['MNIST', 'DTD', 'EuroSAT', 'GTSRB', 'SUN397', 'SVHN']
#datasets = ['SVHN']
test_datasets = ['CIFAR10']
model = 'ViT-B-32'
args = parse_arguments()
args.data_location = 'datasets'
args.model = model
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'


# 定义元模型（Meta-model）
class MetaModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# 训练元模型
def train_meta_model(task_vectors, similarities, meta_model, optimizer, criterion):
    meta_model.train()

    # 计算任务相似度的输入（可能是两个向量的拼接或者其他处理方法）
    input_data = torch.tensor(similarities, dtype=torch.float32)  # 任务相似度
    target_weights = torch.tensor(task_vectors, dtype=torch.float32)  # 每个客户端的真实权重

    optimizer.zero_grad()

    # 元模型前向传播
    output_weights = meta_model(input_data)

    # 计算损失
    loss = criterion(output_weights, target_weights)

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss


# 模型融合方法
def model_merging(task_vectors, meta_model, similarities):
    meta_model.eval()

    # 生成的权重
    generated_weights = meta_model(torch.tensor(similarities, dtype=torch.float32))

    # 将生成的权重应用到模型融合上
    merged_model = torch.zeros_like(task_vectors[0])  # 初始化一个空的模型
    total_weight = 0

    # 基于权重进行融合
    for i, weight in enumerate(generated_weights):
        merged_model += task_vectors[i] * weight
        total_weight += weight

    # 归一化模型
    merged_model /= total_weight

    return merged_model


# task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]

similarities = [[torch.cosine_similarity(v1, v2, dim=0).item() for v2 in task_vectors] for v1 in
                task_vectors]  # 计算任务向量之间的相似度

# 初始化元模型
input_dim = len(task_vectors)  # 输入维度为任务向量的数量
output_dim = 1  # 输出维度为每个客户端的权重
meta_model = MetaModel(input_dim, output_dim)

# 优化器和损失函数
optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练元模型
for epoch in range(1000):
    loss = train_meta_model(task_vectors, similarities, meta_model, optimizer, criterion)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 使用训练好的元模型进行模型融合
merged_model = model_merging(task_vectors, meta_model, similarities)
print("Merged Model:", merged_model)
