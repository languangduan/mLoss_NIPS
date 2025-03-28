import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np
import logging
import random
# Config
#datasets = ['MNIST', 'DTD', 'EuroSAT', 'GTSRB', 'SUN397', 'SVHN', 'Cars', 'RESISC45']
# datasets = ['SVHN']
#test_datasets = ['CIFAR10']
args = parse_arguments()
model = args.model
datasets = args.eval_datasets
K = args.k
SEED = args.seed

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Setting random seed to {SEED} for reproducibility.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
#args.data_location = 'datasets'
#args.model = model
#args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

# 加载预训练模型的 state_dict
pretrained_model = torch.load(pretrained_checkpoint)
pretrained_state_dict = pretrained_model.state_dict()



def random_trim_task_vector(task_vector, k, rescale = True):
    """
    Trims the task vector by randomly keeping k proportion of values and setting the rest to zero.

    Args:
        task_vector (dict): A model state dict, where keys are parameter names and values are tensors.
        k (float): Proportion of values to keep (0 < k <= 1).

    Returns:
        dict: A trimmed task vector with the same keys but randomly modified tensor values.
    """
    trimmed_vector = {}
    for key, tensor in task_vector.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # Generate a random mask with probability k to keep each element
        mask = torch.bernoulli(torch.full(tensor.shape, k)).to(tensor.device)

        # Apply the mask to keep k proportion of the elements
        if rescale == False:
            trimmed_vector[key] = tensor * mask
        else:
            trimmed_vector[key] = tensor * mask *(1/k)


    return trimmed_vector




def elect_sign_vector(task_vectors):
    """
    Elects the aggregated sign vector by resolving disagreements across task vectors.

    Args:
        task_vectors (list[dict]): A list of model state dicts, where each dict contains parameter tensors.

    Returns:
        dict: A model state dict containing the elected sign for each parameter.
    """
    elected_sign = {}

    # Iterate over keys in the task vector
    for key in task_vectors[0].keys():
        # Stack all task vector tensors for the current parameter
        stacked_tensors = torch.stack([task_vector[key] for task_vector in task_vectors])  # Shape: (n, ...)
        
        # Calculate positive and negative mass
        positive_mass = stacked_tensors.clamp(min=0).sum(dim=0)  # Sum of positive values
        negative_mass = (-stacked_tensors).clamp(min=0).sum(dim=0)  # Sum of negative values
        
        # Elect the sign based on the greater mass
        elected_sign[key] = torch.where(positive_mass >= negative_mass, 1.0, -1.0)
    
    return elected_sign



# Disjoint Merge Step

def disjoint_merge(task_vectors, elected_sign):
    """
    Combines task vectors by keeping only the parameters whose signs match the elected sign.
    Performs element-wise scaling based on the number of task vectors contributing to each element.

    Args:
        task_vectors (list[dict]): A list of trimmed model state dicts.
        elected_sign (dict): A model state dict containing the elected sign for each parameter.

    Returns:
        dict: A merged model state dict with element-wise averaged contributions.
    """
    merged_vector = {}

    # Iterate over keys in the task vectors
    for key in task_vectors[0].keys():
        # Initialize tensors for the merged result and contribution counts
        merged_tensor = torch.zeros_like(task_vectors[0][key])
        contribution_counts = torch.zeros_like(task_vectors[0][key], dtype=torch.float)

        # Iterate over each task vector to accumulate contributions and counts
        for task_vector in task_vectors:
            # Create a mask where the sign matches the elected sign
            match_mask = torch.sign(task_vector[key]) == elected_sign[key]
            
            # Apply the mask to get the contributing values
            contributing_values = torch.where(match_mask, task_vector[key], torch.zeros_like(task_vector[key]))
            
            # Accumulate the contributing values
            merged_tensor += contributing_values
            
            # Update the contribution counts (1 where contributing, 0 otherwise)
            contribution_counts += match_mask.float()

        # Avoid division by zero by setting zero counts to one (these will result in zero after division)
        contribution_counts = torch.where(contribution_counts == 0, torch.ones_like(contribution_counts), contribution_counts)
        
        # Perform element-wise division to average the contributions
        averaged_tensor = merged_tensor / contribution_counts
        
        # Assign the averaged tensor to the merged vector
        merged_vector[key] = averaged_tensor

    return merged_vector




# Main Function to Perform TIES-MERGING
def dare_merging(task_vectors, k):
    """
    Executes the DARE-MERGING process using the three steps: Random Trim, Elect, and Disjoint Merge.

    Args:
        task_vectors (list[dict]): A list of model state dicts.
        k (float): Proportion of values to keep in the trimming step.

    Returns:
        dict: A merged model state dict.
    """
    # Step 1: Trim (Randomly)
    trimmed_task_vectors = [random_trim_task_vector(tv, k, rescale = True) for tv in task_vectors]

    # Step 2: Elect
    elected_sign = elect_sign_vector(trimmed_task_vectors)

    # Step 3: Disjoint Merge
    merged_task_vector = disjoint_merge(trimmed_task_vectors, elected_sign)

    return merged_task_vector



# Load Task Vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt').vector
    for dataset in datasets
]

# Perform TIES-MERGING
k = args.k  # Keep top k values
merged_task_vector = dare_merging(task_vectors, k)

# Add to Initial Parameters and Scale
scaling_hyperparameter = 1.0
final_parameters = {key: scaling_hyperparameter * merged_task_vector[key]
                    for key in merged_task_vector.keys()}

# Apply to Model and Evaluate
image_encoder = TaskVector(vector=final_parameters).apply_to(pretrained_checkpoint, scaling_coef=1)

evaluation_results = {}

for dataset in datasets:
    print(f"Evaluating on {dataset}...")
    result = eval_single_dataset(image_encoder, dataset, args)
    evaluation_results[dataset] = result

# Calculate the average accuracy across all datasets
accuracies = [result['top1'] for result in evaluation_results.values()]
average_accuracy = sum(accuracies) / len(accuracies)

# Add the average accuracy to the evaluation results
evaluation_results['average_accuracy'] = average_accuracy

# Save the evaluation results to the specified JSON path
with open(args.results_db, 'w') as f:
    json.dump(evaluation_results, f, indent=4)

# Log the average accuracy
print(f"Average accuracy across datasets: {average_accuracy * 100:.2f}%")
print(f"Average accuracy across datasets: {average_accuracy * 100:.2f}%")

print("Evaluation results saved to:", args.results_db)
