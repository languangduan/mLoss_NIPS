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


# Define Meta-model
class MetaModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# Train the meta-model
def train_meta_model(task_vectors, similarities, meta_model, optimizer, criterion):
    meta_model.train()

    # Compute the input for task similarity (could be concatenation of two vectors or other methods)
    input_data = torch.tensor(similarities, dtype=torch.float32)  # Task similarities
    target_weights = torch.tensor(task_vectors, dtype=torch.float32)  # True weights for each client

    optimizer.zero_grad()

    # Meta-model forward pass
    output_weights = meta_model(input_data)

    # Compute loss
    loss = criterion(output_weights, target_weights)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss


# Model merging method
def model_merging(task_vectors, meta_model, similarities):
    meta_model.eval()

    # Generated weights
    generated_weights = meta_model(torch.tensor(similarities, dtype=torch.float32))

    # Apply generated weights to model merging
    merged_model = torch.zeros_like(task_vectors[0])  # Initialize an empty model
    total_weight = 0

    # Merge based on weights
    for i, weight in enumerate(generated_weights):
        merged_model += task_vectors[i] * weight
        total_weight += weight

    # Normalize the model
    merged_model /= total_weight

    return merged_model


# task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]

similarities = [[torch.cosine_similarity(v1, v2, dim=0).item() for v2 in task_vectors] for v1 in
                task_vectors]  # Calculate similarities between task vectors

# Initialize meta-model
input_dim = len(task_vectors)  # Input dimension is the number of task vectors
output_dim = 1  # Output dimension is the weight for each client
meta_model = MetaModel(input_dim, output_dim)

# Optimizer and loss function
optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the meta-model
for epoch in range(1000):
    loss = train_meta_model(task_vectors, similarities, meta_model, optimizer, criterion)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Use the trained meta-model for model merging
merged_model = model_merging(task_vectors, meta_model, similarities)
print("Merged Model:", merged_model)
