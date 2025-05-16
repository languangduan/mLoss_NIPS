import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
import numpy as np
import random
import logging
import json

# --------------------------
# Config and Initialization
# --------------------------
args = parse_arguments()
model_name = args.model
datasets = args.eval_datasets
SEED = args.seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Setting random seed to {SEED} for reproducibility.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load pretrained model
pretrained_checkpoint = f'checkpoints/{model_name}/zeroshot.pt'
pretrained_model = torch.load(pretrained_checkpoint)
pretrained_state_dict = pretrained_model.state_dict()

# Load task vectors (difference between fine-tuned and pretrained)
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model_name}/{dataset}/finetuned.pt').vector
    for dataset in datasets
]

# --------------------------
# EMR-MERGING Functions
# --------------------------

def elect_unified_task_vector(task_vectors):
    """
    Elect a unified task vector τ_uni from all task vectors.
    For each parameter key:
      1. Stack all task vectors: shape (n, ...)
      2. Sum across tasks and take sign (if 0, set to +1)
      3. For each element, among all task vectors, select the maximum absolute value that matches the unified sign
      4. Return τ_uni = gamma_uni * epsilon_uni
    """
    unified = {}
    keys = task_vectors[0].keys()
    for key in keys:
        tensor_stack = torch.stack([tv[key] for tv in task_vectors], dim=0)
        sum_tensor = tensor_stack.sum(dim=0)
        gamma_uni = torch.where(sum_tensor == 0, torch.ones_like(sum_tensor), torch.sign(sum_tensor))
        gamma_uni_expanded = gamma_uni.unsqueeze(0).expand_as(tensor_stack)
        sign_mask = (torch.sign(tensor_stack) == gamma_uni_expanded).float()
        abs_stack = torch.abs(tensor_stack)
        masked_abs = abs_stack * sign_mask
        epsilon_uni, _ = masked_abs.max(dim=0)
        unified[key] = gamma_uni * epsilon_uni
    return unified

def compute_mask(task_vector, unified_tau):
    """
    For a single task vector, generate a binary mask M = (τ_i * τ_uni > 0)
    Only keep locations where the task vector and the unified task vector have the same sign.
    """
    mask = {}
    for key in task_vector.keys():
        mask[key] = (task_vector[key] * unified_tau[key] > 0).float()
    return mask

def compute_rescaler(task_vector, mask, unified_tau, eps=1e-8):
    """
    Compute the task-specific scaling factor λ:
      λ = (sum(abs(τ_i))) / (sum(abs(M * τ_uni)) + eps)
    """
    total_task = 0.0
    total_masked = 0.0
    for key in task_vector.keys():
        total_task += torch.sum(torch.abs(task_vector[key])).item()
        total_masked += torch.sum(torch.abs(mask[key] * unified_tau[key])).item()
    return total_task / (total_masked + eps)

def modulate_unified(task_vector, unified_tau):
    """
    For a single task, compute mask, then rescaler,
    and return the modulated task vector: λ * (M * τ_uni).
    Also return λ and mask for debugging.
    """
    mask = compute_mask(task_vector, unified_tau)
    lambda_i = compute_rescaler(task_vector, mask, unified_tau)
    modulated = {}
    for key in task_vector.keys():
        modulated[key] = lambda_i * mask[key] * unified_tau[key]
    return modulated, lambda_i, mask

def merge_model(pretrained_state, modulated_tau):
    """
    Add the modulated task vector to the pretrained model to obtain the merged model parameters.
    """
    merged_state = {}
    for key in pretrained_state.keys():
        if key in modulated_tau:
            merged_state[key] = pretrained_state[key] + modulated_tau[key]
        else:
            merged_state[key] = pretrained_state[key]
    return merged_state

def emr_merging(task_vectors, pretrained_state):
    """
    Full EMR-MERGING process:
      1. Elect a unified task vector τ_uni from all task vectors.
      2. For each task vector:
         - Compute task-specific mask M and scaling factor λ
         - Obtain modulated task vector: λ * (M * τ_uni)
         - Final model parameters: Ŵ = W_pre + modulated task vector
    Return the list of merged model state_dicts for each task, and the unified vector for debugging.
    """
    unified_tau = elect_unified_task_vector(task_vectors)
    merged_models = []
    for tv in task_vectors:
        modulated_tau, lambda_i, mask = modulate_unified(tv, unified_tau)
        merged_state = merge_model(pretrained_state, modulated_tau)
        merged_models.append(merged_state)
        logger.info(f"Computed lambda: {lambda_i:.4f}")
    return merged_models, unified_tau

# --------------------------
# Main Execution
# --------------------------
logger.info("Starting EMR-MERGING process...")

# EMR-MERGING to get merged models for each task
merged_state_dicts, unified_tau = emr_merging(task_vectors, pretrained_state_dict)

# Evaluate each merged model on its corresponding dataset
evaluation_results = {}
for dataset, merged_state in zip(datasets, merged_state_dicts):
    logger.info(f"Evaluating merged model on {dataset}...")
    merged_model = TaskVector(vector=merged_state).apply_to(pretrained_checkpoint, scaling_coef=1)
    result = eval_single_dataset(merged_model, dataset, args)
    evaluation_results[dataset] = result

# Compute average accuracy across all tasks
accuracies = [result['top1'] for result in evaluation_results.values()]
average_accuracy = sum(accuracies) / len(accuracies)
evaluation_results['average_accuracy'] = average_accuracy

# Save results to JSON
with open(args.results_db, 'w') as f:
    json.dump(evaluation_results, f, indent=4)

logger.info(f"Average accuracy across datasets: {average_accuracy * 100:.2f}%")
logger.info(f"Evaluation results saved to: {args.results_db}")
