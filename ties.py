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
# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Example of logging
logger.info("Logger initialized successfully.")

args = parse_arguments()
SEED = args.seed
logger.info(f"Setting random seed to {SEED} for reproducibility.")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

    
model = args.model
datasets = args.eval_datasets
K = args.k
if args.layerwise == "True":
    layerwise = True
else:
    layerwise = False
#args.data_location = 'datasets'
#args.model = model
#args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

# 加载预训练模型的 state_dict
pretrained_model = torch.load(pretrained_checkpoint)
pretrained_state_dict = pretrained_model.state_dict()



#layer_wise_trim step
def layerwise_trim_task_vector(task_vector, k):
    """
    Trims the task vector (model state dict) by keeping only the top-k% values based on magnitude for each parameter.
    Replaces torch.quantile with torch.topk to handle large tensors efficiently.
    
    Args:
        task_vector (dict): A model state dict, where keys are parameter names and values are tensors.
        k (float): Percentage of values to keep (0 < k <= 1).
    
    Returns:
        dict: A trimmed task vector with the same keys but modified tensor values.
    """
    trimmed_vector = {}
    for key, tensor in task_vector.items():
        # Skip non-tensor entries (if any)
        if not isinstance(tensor, torch.Tensor):
            continue
        
        # Flatten the tensor to 1D for processing
        flat_tensor = tensor.view(-1)
        num_elements = flat_tensor.numel()
        num_keep = max(1, int(num_elements * k))  # Ensure at least one element is kept
        
        if num_keep >= num_elements:
            # If k >= 1, keep all elements
            trimmed_vector[key] = tensor.clone()
            continue
        
        # Use torch.topk to find the top-k% elements by magnitude
        try:
            topk = torch.topk(flat_tensor.abs(), num_keep, largest=True, sorted=False)
            threshold = topk.values.min()
        except RuntimeError as e:
            print(f"Error processing key '{key}': {e}")
            # In case of error, skip trimming for this key
            trimmed_vector[key] = tensor.clone()
            continue
        
        # Create a mask where elements >= threshold
        mask = tensor.abs() >= threshold
        
        # Apply the mask to keep top-k% elements, set others to zero
        trimmed_tensor = tensor * mask
        
        # Reshape back to original shape
        trimmed_vector[key] = trimmed_tensor.view_as(tensor)
    
    return trimmed_vector


# Global Trim Step
def trim_task_vector(task_vector, k):
    """
    Trims the task vector by keeping only the top-k% values based on the global magnitude
    across all parameters in the state dict.

    Args:
        task_vector (dict): A model state dict, where keys are parameter names and values are tensors.
        k (float): Percentage of values to keep (0 < k <= 1).

    Returns:
        dict: A trimmed task vector with the same keys but modified tensor values.
    """
    # Step 1: Flatten and collect all values into a list
    all_values = torch.cat([param.view(-1).abs() for param in task_vector.values() if isinstance(param, torch.Tensor)])
    
    # Step 2: Calculate the number of values to discard (bottom (1-k)%)
    num_values_to_discard = int((1 - k) * all_values.numel())
    
    # Step 3: Sort and find the threshold for the top-k% values
    threshold = torch.kthvalue(all_values, num_values_to_discard).values

    # Step 4: Create a trimmed task vector
    trimmed_vector = {}
    for key, tensor in task_vector.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        # Mask values below the threshold
        magnitude = tensor.abs()
        trimmed_vector[key] = torch.where(magnitude >= threshold, tensor, torch.zeros_like(tensor))
    
    return trimmed_vector





#task_vector为字典的trim
# def trim_task_vector(task_vector, k):
#     """
#     Trims the task vector by keeping only the top-k% values based on magnitude for each key.
#     """
#     trimmed_vector = {}
#     for key, tensor in task_vector.items():
#         # 计算每个张量的绝对值
#         magnitude = tensor.abs()
#         # 根据百分比计算阈值
#         threshold = torch.quantile(magnitude, 1 - k)
#         # 保留大于或等于阈值的值，其余置为0
#         trimmed_vector[key] = torch.where(magnitude >= threshold, tensor, torch.zeros_like(tensor))
#     return trimmed_vector


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


'''old method, without counting A_p'''
# def disjoint_merge(task_vectors, elected_sign):
#     """
#     Combines task vectors by keeping only the parameters whose signs match the elected sign.

#     Args:
#         task_vectors (list[dict]): A list of model state dicts.
#         elected_sign (dict): A model state dict containing the elected sign for each parameter.

#     Returns:
#         dict: A merged model state dict.
#     """
#     merged_vector = {}

#     # Iterate over keys in the task vector
#     for key in task_vectors[0].keys():
#         # Initialize a tensor for the merged result
#         merged_tensor = torch.zeros_like(task_vectors[0][key])
        
#         # Merge task vectors whose signs match the elected sign
#         for task_vector in task_vectors:
#             match_mask = torch.sign(task_vector[key]) == elected_sign[key]
#             merged_tensor += torch.where(match_mask, task_vector[key], torch.zeros_like(task_vector[key]))
        
#         # Normalize the merged tensor by the number of task vectors
#         merged_vector[key] = merged_tensor / len(task_vectors)
    
#     return merged_vector



# Main Function to Perform TIES-MERGING
def ties_merging(task_vectors, k, layerwise = False):
    """
    Executes the TIES-MERGING process using the three steps: Trim, Elect, and Disjoint Merge.

    Args:
        task_vectors (list[dict]): A list of model state dicts.
        k (float): Percentage of values to keep in the trimming step.

    Returns:
        dict: A merged model state dict.
    """
    # Step 1: Trim
    if layerwise == False:
        trimmed_task_vectors = [trim_task_vector(tv, k) for tv in task_vectors]
    else:
        trimmed_task_vectors = [layerwise_trim_task_vector(tv, k) for tv in task_vectors]

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
k = args.k  # Keep top 20% values
merged_task_vector = ties_merging(task_vectors, k,layerwise=layerwise)

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

