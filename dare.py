import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
import numpy as np
import random
import logging
import json

# Argument parsing and config
args = parse_arguments()
model = args.model
datasets = args.eval_datasets
K = args.k
SEED = args.seed

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Setting random seed to {SEED} for reproducibility.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load pretrained model
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
logger.info(f"Loading pretrained model from: {pretrained_checkpoint}")
pretrained_model = torch.load(pretrained_checkpoint, map_location=args.device)
pretrained_state_dict = pretrained_model.state_dict()

def random_trim_task_vector(task_vector, k, rescale=True):
    """
    Randomly keep k proportion of each tensor's elements in the task vector, set others to zero.
    Optionally rescale the kept values by 1/k.
    """
    trimmed_vector = {}
    for key, tensor in task_vector.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        mask = torch.bernoulli(torch.full(tensor.shape, k, device=tensor.device))
        if not rescale:
            trimmed_vector[key] = tensor * mask
        else:
            trimmed_vector[key] = tensor * mask * (1.0 / k)
    return trimmed_vector

def elect_sign_vector(task_vectors):
    """
    For each parameter, elect the sign (+1 or -1) by majority mass across task vectors.
    """
    elected_sign = {}
    for key in task_vectors[0].keys():
        stacked_tensors = torch.stack([tv[key] for tv in task_vectors])
        positive_mass = stacked_tensors.clamp(min=0).sum(dim=0)
        negative_mass = (-stacked_tensors).clamp(min=0).sum(dim=0)
        elected_sign[key] = torch.where(positive_mass >= negative_mass, 1.0, -1.0)
    return elected_sign

def disjoint_merge(task_vectors, elected_sign):
    """
    For each parameter, keep only the elements whose sign matches the elected sign.
    Average the contributions for each element.
    """
    merged_vector = {}
    for key in task_vectors[0].keys():
        merged_tensor = torch.zeros_like(task_vectors[0][key])
        contribution_counts = torch.zeros_like(task_vectors[0][key], dtype=torch.float)
        for tv in task_vectors:
            match_mask = torch.sign(tv[key]) == elected_sign[key]
            merged_tensor += torch.where(match_mask, tv[key], torch.zeros_like(tv[key]))
            contribution_counts += match_mask.float()
        contribution_counts = torch.where(contribution_counts == 0, torch.ones_like(contribution_counts), contribution_counts)
        merged_vector[key] = merged_tensor / contribution_counts
    return merged_vector

def dare_merging(task_vectors, k):
    """
    DARE-MERGING: random trim, elect sign, and disjoint merge.
    """
    trimmed_task_vectors = [random_trim_task_vector(tv, k, rescale=True) for tv in task_vectors]
    elected_sign = elect_sign_vector(trimmed_task_vectors)
    merged_task_vector = disjoint_merge(trimmed_task_vectors, elected_sign)
    return merged_task_vector

# Load task vectors (from finetuned checkpoints)
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt').vector
    for dataset in datasets
]

# Perform DARE merging
k = args.k
merged_task_vector = dare_merging(task_vectors, k)

# Apply scaling and create merged model
scaling_hyperparameter = 1.0
final_parameters = {key: scaling_hyperparameter * merged_task_vector[key]
                    for key in merged_task_vector.keys()}
image_encoder = TaskVector(vector=final_parameters).apply_to(pretrained_checkpoint, scaling_coef=1)

# Evaluate on each dataset
evaluation_results = {}
for dataset in datasets:
    logger.info(f"Evaluating on {dataset}...")
    result = eval_single_dataset(image_encoder, dataset, args)
    evaluation_results[dataset] = result

accuracies = [result['top1'] for result in evaluation_results.values()]
average_accuracy = sum(accuracies) / len(accuracies)
evaluation_results['average_accuracy'] = average_accuracy

with open(args.results_db, 'w') as f:
    json.dump(evaluation_results, f, indent=4)

logger.info(f"Average accuracy across datasets: {average_accuracy * 100:.2f}%")
logger.info(f"Evaluation results saved to: {args.results_db}")
