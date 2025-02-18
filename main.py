import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
import torch.nn.functional as F
from tqdm import tqdm
import json

# Config
datasets = ['MNIST','DTD','EuroSAT','GTSRB','SUN397','SVHN']
#datasets = ['MNIST']
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


def combine_task_vectors(task_vectors, weights, lambda_coef):
    vectors = [tv.vector for tv in task_vectors]
    keys = vectors[0].keys()

    #combined_vector = {key: torch.zeros_like(vectors[0][key]) for key in keys}
    combined_vector = {key: torch.zeros_like(next(iter(vectors[0].values()))) for key in keys}

    num_vectors = len(vectors)
    projection_log = []

    for key in keys:
        vector_list = [v[key] for v in vectors]
        # Apply L1 regularization to each vector after projection
        vector_list = [apply_l1_regularization(vec, lambda_coef) for vec in vector_list]

        for i in range(num_vectors):

            for j in range(i + 1, num_vectors):
                cos_sim = cosine_similarity(vector_list[i], vector_list[j])
                if cos_sim < 0:  # cos_sim < 0 implies an angle > 90 degrees
                    vector_list[j] = orthogonal_projection(vector_list[j], vector_list[i])
                    projection_log.append({
                        'key': key,
                        'task_1': datasets[i],
                        'task_2': datasets[j],
                        'cosine_similarity': cos_sim
                    })


        # Sum vectors
        combined_vector[key] = sum(w * vec for w, vec in zip(weights, vector_list))

    with open('projection_log.json', 'w') as f:
        json.dump(projection_log, f, indent=4)

    return TaskVector(vector=combined_vector)

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
def calculate_weights_from_similarity(similarity_matrix, target_weight=1/3):
    '''
    num_tasks = similarity_matrix.size(0)
    weights = torch.zeros(num_tasks)

    # Normalize the similarity values to derive weights
    for i in range(num_tasks):
        weights[i] = similarity_matrix[i].sum()  # Sum similarities for each task

    # Normalize weights to sum to 1
    normalized_weights = weights / weights.sum()
    scaling_factor = target_weight / normalized_weights.mean()
    weights = normalized_weights * scaling_factor
    return weights
'''

    # Inverse the similarity values to get "importance" (lower similarity -> higher weight)
    inverse_similarities = 1 / (similarity_matrix + 1e-6)  # Add small epsilon to avoid division by zero

    # Sum up the inverse similarities for each task to get the total "importance"
    importance = inverse_similarities.sum(dim=1)

    # Normalize the importance values
    normalized_weights = importance / importance.sum()

    # Adjust the weights to be around the target_weight (e.g., 1/3)
    scaling_factor = target_weight / normalized_weights.mean()
    weights = normalized_weights * scaling_factor

    return weights

# Create the task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]

similarity_matrix = calculate_similarity_matrix(task_vectors)
print('sim',similarity_matrix)
weights = calculate_weights_from_similarity(similarity_matrix, target_weight=1/3)
print('weight',weights)
lambda_coef = 0.00001
# Combine task vectors with orthogonal projection
task_vector_combined =combine_task_vectors(task_vectors, weights, lambda_coef)
#task_vector_combined =combine_task_vectors_ignore_last_layer(task_vectors, weights, lambda_coef)
# Apply the resulting task vector
image_encoder = task_vector_combined.apply_to(pretrained_checkpoint, scaling_coef= 1 / len(datasets))

# Evaluate
for dataset in datasets:
    eval_single_dataset(image_encoder, dataset, args)