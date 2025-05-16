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





# Config
#datasets = ['MNIST', 'DTD', 'EuroSAT', 'GTSRB', 'SUN397', 'SVHN']
# currently no 'Cars' and 'RESISC45'
# datasets = ['SVHN']
#test_datasets = ['CIFAR10']
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
#args.data_location = 'datasets'
#args.model = model
#args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

pretrained_model = torch.load(pretrained_checkpoint)
pretrained_state_dict = pretrained_model.state_dict()

#simply avg merge the task vectors
#can also work for whole model parameter (not only task vec)
def avg_merge(task_vectors):
    """
    Args:
        task_vectors (list[dict]): A list of model state dicts.
    Returns:
        dict: A merged model state dict.
    """
    merged_vector = {}

    # Iterate over keys in the task vector
    for key in task_vectors[0].keys():
        # Initialize a tensor for the merged result
        merged_tensor = torch.zeros_like(task_vectors[0][key])
        
        # Merge task vectors whose signs match the elected sign
        for task_vector in task_vectors:
            merged_tensor += task_vector[key]
        
        # Normalize the merged tensor by the number of task vectors
        merged_vector[key] = merged_tensor / len(task_vectors)
    
    return merged_vector


# Load Task Vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt').vector
    for dataset in datasets
]

# Perform task arithmetic
merged_task_vector = avg_merge(task_vectors)

# Add to Initial Parameters and Scale
scaling_hyperparameter = args.scaling
final_parameters = {key: scaling_hyperparameter * merged_task_vector[key]
                    for key in merged_task_vector.keys()}

# Apply to Model and Evaluate
image_encoder = TaskVector(vector=final_parameters).apply_to(pretrained_checkpoint)

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
