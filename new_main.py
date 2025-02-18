import torch
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

def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.view(-1)
    tensor2 = tensor2.view(-1)
    cos_sim = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
    return cos_sim.item()

def orthogonal_projection(tensor1, tensor2):
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)
    dot_prod = torch.dot(tensor1_flat, tensor2_flat)
    norm_sq = torch.dot(tensor2_flat, tensor2_flat)
    projection_flat = tensor1_flat - (dot_prod / norm_sq) * tensor2_flat
    return projection_flat.view_as(tensor1)

def apply_l1_regularization(tensor, lambda_coef):
    """Applies L1 regularization to a tensor."""
    with torch.no_grad():
        return torch.sign(tensor) * torch.clamp(torch.abs(tensor) - lambda_coef, min=0)

def combine_task_vectors_with_layer_split(task_vectors, weights, lambda_coef, private_layers):
    """
    Aggregate Task Vectors while preserving private layers (classification layers) for each client,
    and share all other layers.

    Args:
    - task_vectors: List of TaskVectors for each client.
    - weights: List of weights corresponding to each client.
    - lambda_coef: L1 regularization coefficient.
    - private_layers: List of layer names to be preserved for each client (only contains classification layers).

    Returns:
    - combined_shared_vector: Aggregated TaskVector for shared layers.
    - private_task_vectors: List of TaskVectors for the preserved layers of each client.
    """
    vectors = [tv.vector for tv in task_vectors]
    keys = vectors[0].keys()

    # Separate shared layers and private layers based on the provided keys
    shared_layers = [key for key in keys if key not in private_layers]
    print(f"shared_layers: {(shared_layers)}, private: {private_layers}")

    # Initialize a vector dictionary for shared layers
    #print("Task Vector keys:", list(task_vectors[0].vector.keys()))

    # Initialize a vector dictionary for shared layers
    shared_vector = {key: torch.zeros_like(next(iter(vectors[0].values()))) for key in shared_layers if key in keys}

    # Initialize a list of vector dictionaries for the private layers of each client
    private_vectors = [
        {key: torch.zeros_like(task_vectors[0].vector[key]) for key in private_layers if key in keys}
        for _ in range(len(task_vectors))
    ]

    num_vectors = len(vectors)
    projection_log = []

    # Aggregate the shared layers
    for key in shared_layers:
        if key not in keys:
            print(f"Skipping layer {key} as it is not found in task vector keys.")
            continue

        # Retrieve the vectors for the current layer from all clients
        vector_list = [v[key] for v in vectors]
        # Apply L1 regularization to each vector after projection
        vector_list = [apply_l1_regularization(vec, lambda_coef) for vec in vector_list]

        num_vectors = len(vector_list)

        # 1. 计算每对向量的余弦相似度
        cos_sim_list = []
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                cos_sim = cosine_similarity(vector_list[i], vector_list[j])
                cos_sim_list.append((cos_sim, i, j))

        # 2. 根据余弦相似度从大到小排序
        cos_sim_list.sort(reverse=True, key=lambda x: x[0])
        #print('cos',cos_sim_list)

        # 3. 根据排序顺序进行正交投影
        for cos_sim, i, j in cos_sim_list:
            if cos_sim < 0:  # 余弦相似度小于 0 时进行正交投影
                vector_list[j] = orthogonal_projection(vector_list[j], vector_list[i])
                # 记录投影日志
                projection_log.append({
                    'key': key,
                    'task_1': datasets[i],
                    'task_2': datasets[j],
                    'cosine_similarity': cos_sim
                })

        # Use weighted sum to aggregate the shared layer vectors
        shared_vector[key] = sum(w * vec for w, vec in zip(weights, vector_list))

    # Separate and preserve the private layer parameters for each client
    for i, task_vector in enumerate(task_vectors):
        for key in private_layers:
            if key not in keys:
                print(f"Skipping private layer {key} as it is not found in task vector keys.")
                continue
            private_vectors[i][key] = task_vector.vector[key]

    # Save the aggregated shared layers and private layers into TaskVector objects
    combined_shared_vector = TaskVector(vector=shared_vector)
    private_task_vectors = [TaskVector(vector=private_vector) for private_vector in private_vectors]

    with open('projection_log.json', 'w') as f:
        json.dump(projection_log, f, indent=4)

    return combined_shared_vector, private_task_vectors

# Function to calculate pairwise cosine similarities between task vectors
def calculate_similarity_matrix(task_vectors):
    num_tasks = len(task_vectors)
    similarity_matrix = torch.zeros(num_tasks, num_tasks)

    # Get the keys from the first task vector (assuming all task vectors have the same keys)
    keys = list(task_vectors[0].vector.keys())

    for i in range(num_tasks):
        for j in range(i, num_tasks):
            total_cos_sim = 0
            count = 0

            # Compute cosine similarity for each key (tensor)
            for key in keys:
                tensor1 = task_vectors[i].vector[key]
                tensor2 = task_vectors[j].vector[key]
                cos_sim = cosine_similarity(tensor1, tensor2)
                total_cos_sim += cos_sim
                count += 1

            # Average cosine similarity over all tensors
            avg_cos_sim = total_cos_sim / count
            similarity_matrix[i, j] = avg_cos_sim
            similarity_matrix[j, i] = avg_cos_sim  # Matrix is symmetric

    return similarity_matrix


# Function to derive weights from similarity matrix
def calculate_weights_from_similarity(similarity_matrix, target_weight=1/3,sigma = 0.3):
    total_similarity = similarity_matrix.sum(dim=1)
    print('sim',total_similarity)
    # Invert the similarity matrix to get "importance" (lower similarity -> higher weight)
    #inverse_similarities =  similarity_matrix
    # Ensure no negative values
    similarities = torch.clamp(similarity_matrix, min=0)

    # Apply Gaussian smoothing to inverse similarities
    gaussian_weights = torch.exp(-similarities ** 2 / (2 * sigma ** 2))

    # Sum up the Gaussian weights to get total importance
    gaussian_weights = gaussian_weights.sum(dim=1)

    # Normalize the importance values
    normalized_weights = gaussian_weights / gaussian_weights.sum()

    # Scale the weights to ensure they reflect the target weight
    weights = normalized_weights * target_weight / normalized_weights.mean()

    return weights


#private_layers = ['model.ln_final.weight', 'model.ln_final.bias']
private_layers = ['model.visual.ln_post.weight', 'model.visual.ln_post.bias', 'model.token_embedding.weight',
                  'model.ln_final.weight', 'model.ln_final.bias']
                  #'model.visual.transformer.resblocks.0.ln_1.weight', 'model.visual.transformer.resblocks.0.ln_1.bias', 'model.visual.transformer.resblocks.0.attn.in_proj_weight', 'model.visual.transformer.resblocks.0.attn.in_proj_bias', 'model.visual.transformer.resblocks.0.attn.out_proj.weight', 'model.visual.transformer.resblocks.0.attn.out_proj.bias', 'model.visual.transformer.resblocks.0.ln_2.weight', 'model.visual.transformer.resblocks.0.ln_2.bias', 'model.visual.transformer.resblocks.0.mlp.c_fc.weight', 'model.visual.transformer.resblocks.0.mlp.c_fc.bias', 'model.visual.transformer.resblocks.0.mlp.c_proj.weight', 'model.visual.transformer.resblocks.0.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.1.ln_1.weight', 'model.visual.transformer.resblocks.1.ln_1.bias', 'model.visual.transformer.resblocks.1.attn.in_proj_weight', 'model.visual.transformer.resblocks.1.attn.in_proj_bias', 'model.visual.transformer.resblocks.1.attn.out_proj.weight', 'model.visual.transformer.resblocks.1.attn.out_proj.bias', 'model.visual.transformer.resblocks.1.ln_2.weight', 'model.visual.transformer.resblocks.1.ln_2.bias', 'model.visual.transformer.resblocks.1.mlp.c_fc.weight', 'model.visual.transformer.resblocks.1.mlp.c_fc.bias', 'model.visual.transformer.resblocks.1.mlp.c_proj.weight', 'model.visual.transformer.resblocks.1.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.2.ln_1.weight', 'model.visual.transformer.resblocks.2.ln_1.bias', 'model.visual.transformer.resblocks.2.attn.in_proj_weight', 'model.visual.transformer.resblocks.2.attn.in_proj_bias', 'model.visual.transformer.resblocks.2.attn.out_proj.weight', 'model.visual.transformer.resblocks.2.attn.out_proj.bias', 'model.visual.transformer.resblocks.2.ln_2.weight', 'model.visual.transformer.resblocks.2.ln_2.bias', 'model.visual.transformer.resblocks.2.mlp.c_fc.weight', 'model.visual.transformer.resblocks.2.mlp.c_fc.bias', 'model.visual.transformer.resblocks.2.mlp.c_proj.weight', 'model.visual.transformer.resblocks.2.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.3.ln_1.weight', 'model.visual.transformer.resblocks.3.ln_1.bias', 'model.visual.transformer.resblocks.3.attn.in_proj_weight', 'model.visual.transformer.resblocks.3.attn.in_proj_bias', 'model.visual.transformer.resblocks.3.attn.out_proj.weight', 'model.visual.transformer.resblocks.3.attn.out_proj.bias', 'model.visual.transformer.resblocks.3.ln_2.weight', 'model.visual.transformer.resblocks.3.ln_2.bias', 'model.visual.transformer.resblocks.3.mlp.c_fc.weight', 'model.visual.transformer.resblocks.3.mlp.c_fc.bias', 'model.visual.transformer.resblocks.3.mlp.c_proj.weight', 'model.visual.transformer.resblocks.3.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.4.ln_1.weight', 'model.visual.transformer.resblocks.4.ln_1.bias', 'model.visual.transformer.resblocks.4.attn.in_proj_weight', 'model.visual.transformer.resblocks.4.attn.in_proj_bias', 'model.visual.transformer.resblocks.4.attn.out_proj.weight', 'model.visual.transformer.resblocks.4.attn.out_proj.bias', 'model.visual.transformer.resblocks.4.ln_2.weight', 'model.visual.transformer.resblocks.4.ln_2.bias', 'model.visual.transformer.resblocks.4.mlp.c_fc.weight', 'model.visual.transformer.resblocks.4.mlp.c_fc.bias', 'model.visual.transformer.resblocks.4.mlp.c_proj.weight', 'model.visual.transformer.resblocks.4.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.5.ln_1.weight', 'model.visual.transformer.resblocks.5.ln_1.bias', 'model.visual.transformer.resblocks.5.attn.in_proj_weight', 'model.visual.transformer.resblocks.5.attn.in_proj_bias', 'model.visual.transformer.resblocks.5.attn.out_proj.weight', 'model.visual.transformer.resblocks.5.attn.out_proj.bias', 'model.visual.transformer.resblocks.5.ln_2.weight', 'model.visual.transformer.resblocks.5.ln_2.bias', 'model.visual.transformer.resblocks.5.mlp.c_fc.weight', 'model.visual.transformer.resblocks.5.mlp.c_fc.bias', 'model.visual.transformer.resblocks.5.mlp.c_proj.weight', 'model.visual.transformer.resblocks.5.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.6.ln_1.weight', 'model.visual.transformer.resblocks.6.ln_1.bias', 'model.visual.transformer.resblocks.6.attn.in_proj_weight', 'model.visual.transformer.resblocks.6.attn.in_proj_bias', 'model.visual.transformer.resblocks.6.attn.out_proj.weight', 'model.visual.transformer.resblocks.6.attn.out_proj.bias', 'model.visual.transformer.resblocks.6.ln_2.weight', 'model.visual.transformer.resblocks.6.ln_2.bias', 'model.visual.transformer.resblocks.6.mlp.c_fc.weight', 'model.visual.transformer.resblocks.6.mlp.c_fc.bias', 'model.visual.transformer.resblocks.6.mlp.c_proj.weight', 'model.visual.transformer.resblocks.6.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.7.ln_1.weight', 'model.visual.transformer.resblocks.7.ln_1.bias', 'model.visual.transformer.resblocks.7.attn.in_proj_weight', 'model.visual.transformer.resblocks.7.attn.in_proj_bias', 'model.visual.transformer.resblocks.7.attn.out_proj.weight', 'model.visual.transformer.resblocks.7.attn.out_proj.bias', 'model.visual.transformer.resblocks.7.ln_2.weight', 'model.visual.transformer.resblocks.7.ln_2.bias', 'model.visual.transformer.resblocks.7.mlp.c_fc.weight', 'model.visual.transformer.resblocks.7.mlp.c_fc.bias', 'model.visual.transformer.resblocks.7.mlp.c_proj.weight', 'model.visual.transformer.resblocks.7.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.8.ln_1.weight', 'model.visual.transformer.resblocks.8.ln_1.bias', 'model.visual.transformer.resblocks.8.attn.in_proj_weight', 'model.visual.transformer.resblocks.8.attn.in_proj_bias', 'model.visual.transformer.resblocks.8.attn.out_proj.weight', 'model.visual.transformer.resblocks.8.attn.out_proj.bias', 'model.visual.transformer.resblocks.8.ln_2.weight', 'model.visual.transformer.resblocks.8.ln_2.bias', 'model.visual.transformer.resblocks.8.mlp.c_fc.weight', 'model.visual.transformer.resblocks.8.mlp.c_fc.bias', 'model.visual.transformer.resblocks.8.mlp.c_proj.weight', 'model.visual.transformer.resblocks.8.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.9.ln_1.weight', 'model.visual.transformer.resblocks.9.ln_1.bias', 'model.visual.transformer.resblocks.9.attn.in_proj_weight', 'model.visual.transformer.resblocks.9.attn.in_proj_bias', 'model.visual.transformer.resblocks.9.attn.out_proj.weight', 'model.visual.transformer.resblocks.9.attn.out_proj.bias', 'model.visual.transformer.resblocks.9.ln_2.weight', 'model.visual.transformer.resblocks.9.ln_2.bias', 'model.visual.transformer.resblocks.9.mlp.c_fc.weight', 'model.visual.transformer.resblocks.9.mlp.c_fc.bias', 'model.visual.transformer.resblocks.9.mlp.c_proj.weight', 'model.visual.transformer.resblocks.9.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.10.ln_1.weight', 'model.visual.transformer.resblocks.10.ln_1.bias', 'model.visual.transformer.resblocks.10.attn.in_proj_weight', 'model.visual.transformer.resblocks.10.attn.in_proj_bias', 'model.visual.transformer.resblocks.10.attn.out_proj.weight', 'model.visual.transformer.resblocks.10.attn.out_proj.bias', 'model.visual.transformer.resblocks.10.ln_2.weight', 'model.visual.transformer.resblocks.10.ln_2.bias', 'model.visual.transformer.resblocks.10.mlp.c_fc.weight', 'model.visual.transformer.resblocks.10.mlp.c_fc.bias', 'model.visual.transformer.resblocks.10.mlp.c_proj.weight', 'model.visual.transformer.resblocks.10.mlp.c_proj.bias',
                  #'model.visual.transformer.resblocks.11.ln_1.weight', 'model.visual.transformer.resblocks.11.ln_1.bias', 'model.visual.transformer.resblocks.11.attn.in_proj_weight', 'model.visual.transformer.resblocks.11.attn.in_proj_bias', 'model.visual.transformer.resblocks.11.attn.out_proj.weight', 'model.visual.transformer.resblocks.11.attn.out_proj.bias', 'model.visual.transformer.resblocks.11.ln_2.weight', 'model.visual.transformer.resblocks.11.ln_2.bias', 'model.visual.transformer.resblocks.11.mlp.c_fc.weight', 'model.visual.transformer.resblocks.11.mlp.c_fc.bias', 'model.visual.transformer.resblocks.11.mlp.c_proj.weight', 'model.visual.transformer.resblocks.11.mlp.c_proj.bias']

# task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]


similarity_matrix = calculate_similarity_matrix(task_vectors)
print('Similarity Matrix:', similarity_matrix)
weights = calculate_weights_from_similarity(similarity_matrix, target_weight=1/3)
print('Weights:', weights)

lambda_coef = 0.00005

combined_shared_vector, private_task_vectors = combine_task_vectors_with_layer_split(
    task_vectors, weights, lambda_coef, private_layers
)


image_encoder = combined_shared_vector.apply_to(pretrained_checkpoint, scaling_coef=1)


for dataset in datasets:
    eval_single_dataset(image_encoder, dataset, args)
        #eval_single_dataset(image_encoder, dataset, test_dataset, args)
