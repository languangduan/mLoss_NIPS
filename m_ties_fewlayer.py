import os
import torch
import json
import random
import logging
import numpy as np
import copy
import time
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from src.args import parse_arguments
from src.datasets.registry import get_dataset
from src.datasets.common import maybe_dictionarize
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from old_history_file.llfcAnalyzerOld import LLFCAnalyzer  
import PIL
from PIL import Image
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

###############################################################################
# 1. Utility: Build Unlabeled Data Combined
###############################################################################
def build_unlabeled_data_combined(dataset_names, args, proportion=0.01, max_samples=500, min_samples=100):
    logger.info("Building an unlabeled combined DataLoader.")
    all_subsets = []
    for ds_name in dataset_names:
        logger.info(f"Loading dataset '{ds_name}' for unlabeled sampling.")
        dataset_obj = get_dataset(
            ds_name,
            preprocess=None,  
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        train_data = dataset_obj.train_dataset
        total_len = len(train_data)
        sample_len = int(total_len * proportion)
        sample_len = max(sample_len, min_samples)
        sample_len = min(sample_len, max_samples)
        if sample_len <= 0:
            logger.warning(f"Sample length for dataset '{ds_name}' is <= 0. Skipping.")
            continue
        indices = list(range(total_len))
        random.shuffle(indices)
        subset_indices = indices[:sample_len]
        subset_data = Subset(train_data, subset_indices)
        all_subsets.append(subset_data)
        logger.info(f"Dataset '{ds_name}': Chosen {len(subset_data)} samples out of {total_len}.")
    if not all_subsets:
        msg = "No subsets created; check your proportion/min_samples/max_samples settings."
        logger.error(msg)
        raise ValueError(msg)
    combined_dataset = ConcatDataset(all_subsets)
    logger.info(f"Total combined unlabeled samples: {len(combined_dataset)}")
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    logger.info("Unlabeled DataLoader created successfully.")
    return combined_loader

to_tensor = transforms.ToTensor()
resize_and_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),        
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
])

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [resize_and_to_tensor(img) if isinstance(img, Image.Image) else img for img in inputs]
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels)
    return inputs, labels

###############################################################################
# 2. m_TIES Merging (Trims Task Vectors, Not Raw Weights)
###############################################################################
def m_ties_merging(
    pretrained_checkpoint,
    finetuned_checkpoints,
    sample_dataloader,
    k=0.2,
    e=0.1,
    rowwise_target_layers=None,
    analyzer=LLFCAnalyzer(device=args.device),
    scaling=1.0,
    args=args
):
    logger.info("Starting m_TIES merging process.")
    logger.info(f"Base keep ratio k={k}, vibration coefficient e={e}.")
    logger.info(f"Loading pretrained model from '{pretrained_checkpoint}'")
    pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu')
    merged_model = copy.deepcopy(pretrained_model)
    models = [merged_model]
    for ckpt in finetuned_checkpoints:
        models.append(torch.load(ckpt, map_location='cpu'))
    pretrained_sd = pretrained_model.state_dict()

    all_task_vectors = []
    for ft_ckpt in finetuned_checkpoints:
        logger.info(f"Loading finetuned checkpoint '{ft_ckpt}'")
        tv = TaskVector(pretrained_checkpoint, ft_ckpt).vector
        all_task_vectors.append(tv)

    param_keys = list(pretrained_sd.keys())
    merged_vector = {}
    detected_layers = []

    fixed_weights = [1.0 / (len(models) - 1)] * (len(models) - 1)
    data_iter = itertools.cycle(sample_dataloader)

    for param_key in param_keys:
        param_tvs = [tv[param_key] for tv in all_task_vectors if param_key in tv]
        if not param_tvs:
            logger.debug(f"No task vectors found for parameter '{param_key}'. Skipping.")
            continue

        is_rowwise_candidate = (rowwise_target_layers and 
                                any(rl in param_key for rl in rowwise_target_layers) and 
                                param_key.endswith(".weight"))
        if is_rowwise_candidate:
            logger.info(f"Parameter '{param_key}' is a rowwise target. Applying rowwise robust trimming.")
            N = (len(models) - 1) * args.sampling_size
            M_l_sum = None
            layer_key = param_key[:-len('.weight')]
            for _ in range(N):
                inputs, _ = next(data_iter)
                M_l = analyzer.compute_llfc_loss_by_row(inputs, models, fixed_weights, layer_key)
                M_l_sum = M_l.clone() if M_l_sum is None else M_l_sum + M_l
            M_l = M_l_sum / N
            trimmed_tvs = rowwise_robust_trim_task_vector(param_tvs, k, e, sample_dataloader, M_l)
            detected_layers.append(param_key)
        else:
            trimmed_tvs = [ties_trim_vector(ptv, k) for ptv in param_tvs]

        merged_param = ties_elect_and_merge(trimmed_tvs)
        merged_vector[param_key] = merged_param
        pretrained_sd[param_key].add_(scaling * merged_param)

    logger.info("m_TIES merging process completed.")
    return merged_vector, detected_layers

###############################################################################
# 3. Example Helper Functions
###############################################################################
def ties_trim_vector(tensor, k):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    flat = tensor.view(-1).abs()
    num_keep = int(len(flat) * k)
    if num_keep <= 0:
        return torch.zeros_like(tensor)
    if num_keep >= len(flat):
        return tensor.clone()
    topk_vals, _ = torch.topk(flat, num_keep, largest=True, sorted=False)
    threshold = topk_vals.min()
    mask = tensor.abs() >= threshold
    return tensor * mask

def robust_scale_losses(row_losses: torch.Tensor) -> torch.Tensor:
    median = torch.median(row_losses)
    q75 = torch.quantile(row_losses, 0.75)
    q25 = torch.quantile(row_losses, 0.25)
    iqr = q75 - q25
    if iqr == 0:
        return torch.zeros_like(row_losses)
    scaled = (row_losses - median) / iqr
    min_val, max_val = scaled.min(), scaled.max()
    return (scaled - min_val) / (max_val - min_val)

def rank_scale(row_losses: torch.Tensor) -> torch.Tensor:
    device = row_losses.device
    sorted_indices = row_losses.argsort()
    ranks = torch.zeros_like(sorted_indices, dtype=torch.float32, device=device)
    ranks[sorted_indices] = torch.arange(len(row_losses), dtype=torch.float32, device=device)
    return ranks / (len(row_losses) - 1)

def log_scale(row_losses: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scaled = torch.log(row_losses + eps)
    min_val, max_val = scaled.min(), scaled.max()
    return (scaled - min_val) / (max_val - min_val)

def rowwise_robust_trim_task_vector(param_tvs, k, e, sample_dataloader, M_l=None):
    logger.info("Rowwise robust trimming for param_tvs. Using M_l metric.")
    shape = param_tvs[0].shape
    out_dim = shape[0]
    if M_l is None:
        M_l = torch.rand(out_dim)
    row_keep_ratios = k - e * rank_scale(M_l)
    logger.info(f"Row keep ratios: {row_keep_ratios}")
    trimmed_tvs = []
    for ptv in param_tvs:
        w_clone = ptv.clone()
        for r_i in range(out_dim):
            keep_r = row_keep_ratios[r_i].item()
            if len(w_clone.shape) == 1:
                w_clone[r_i] = ties_trim_vector(w_clone[r_i], keep_r)
            else:
                row_abs = w_clone[r_i, :].abs()
                row_len = row_abs.numel()
                num_keep = int(row_len * keep_r)
                if num_keep <= 0:
                    w_clone[r_i, :] = 0
                    continue
                if num_keep >= row_len:
                    continue
                topk_vals, _ = torch.topk(row_abs, num_keep, largest=True, sorted=False)
                thr = topk_vals.min()
                row_mask = row_abs >= thr
                w_clone[r_i, :] = w_clone[r_i, :] * row_mask
        trimmed_tvs.append(w_clone)
    return trimmed_tvs

def ties_elect_and_merge(trimmed_tensors):
    if not trimmed_tensors:
        return None
    stk = torch.stack(trimmed_tensors, dim=0)
    pos_mass = stk.clamp(min=0).sum(dim=0)
    neg_mass = (-stk).clamp(min=0).sum(dim=0)
    elected_sign = torch.where(pos_mass >= neg_mass, 1.0, -1.0)
    merged_tensor = torch.zeros_like(elected_sign)
    contribution_counts = torch.zeros_like(elected_sign, dtype=torch.float)
    for t in trimmed_tensors:
        match_mask = torch.sign(t) == elected_sign
        merged_tensor += torch.where(match_mask, t, torch.zeros_like(t))
        contribution_counts += match_mask.float()
    contribution_counts = torch.where(contribution_counts == 0, torch.ones_like(contribution_counts), contribution_counts)
    return merged_tensor / contribution_counts

###############################################################################
# 4. Full Demo Script "main"
###############################################################################
def main():
    start_time = time.time() 
    logger.info("Starting m_TIES script.")
    args = parse_arguments()

    SEED = args.seed
    logger.info(f"Setting random seed to {SEED} for reproducibility.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seeds set successfully.")

    model_name = args.model
    dataset_names = args.eval_datasets  
    pretrained_checkpoint = f'checkpoints/{model_name}/zeroshot.pt'
    finetuned_ckpts = [f'checkpoints/{model_name}/{ds}/finetuned.pt' for ds in dataset_names]

    unlabeled_loader = build_unlabeled_data_combined(
        dataset_names,
        args,
        proportion=0.01,
        max_samples=500,
        min_samples=100
    )
    logger.info("Unlabeled data loader built successfully.")
    if model_name == "ViT-B-32":
        rowwise_target_layers = [f"model.visual.transformer.resblocks.{i}.mlp.c_fc.weight" for i in [8,9,10]]
    elif model_name == "ViT-L-14":
        rowwise_target_layers = [f"model.visual.transformer.resblocks.{i}.mlp.c_fc.weight" for i in range(24)]
    merged_vector, detected_layers = m_ties_merging(
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoints=finetuned_ckpts,
        sample_dataloader=unlabeled_loader,
        k=args.k,
        e=args.e,
        rowwise_target_layers=rowwise_target_layers
    )

    from src.task_vectors import TaskVector
    final_tv = TaskVector(vector=merged_vector)
    merged_model = final_tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    logger.info("Merged model constructed successfully.")

    logger.info("Evaluating merged model on each dataset.")
    evaluation_results = {}
    total_accuracy = 0.0
    num_datasets = 0
    for ds in dataset_names:
        logger.info(f"Evaluating dataset '{ds}'.")
        result = eval_single_dataset(merged_model, ds, args)
        evaluation_results[ds] = result
        if "top1" in result:
            total_accuracy += float(result["top1"])
            num_datasets += 1
        logger.info(f"Results for '{ds}': {result}")
    if detected_layers:
        logger.info(f"Detected rowwise-trimmed layers: {detected_layers}")
        evaluation_results["rowwise_trimmed_layers"] = detected_layers
    else:
        warning_msg = "Warning: No target layers were detected for row-wise trimming."
        logger.warning(warning_msg)
        evaluation_results["rowwise_trimmed_layers"] = warning_msg
    if num_datasets > 0:
        avg_acc = total_accuracy / num_datasets
        evaluation_results["avg_accuracy"] = avg_acc
        logger.info(f"Average accuracy across {num_datasets} datasets: {avg_acc:.4f}")
    else:
        logger.warning("No 'top1' keys found in evaluation results; cannot compute average accuracy.")
    evaluation_results["random_seed"] = SEED

    end_time = time.time()
    runtime_seconds = end_time - start_time
    evaluation_results["runtime_seconds"] = runtime_seconds

    evaluation_results["hyperparameters"] = vars(args)

    logger.info(f"Random seed used: {SEED}")
    logger.info(f"Total runtime: {runtime_seconds:.2f} seconds")
    with open(args.results_db, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    logger.info(f"Evaluation results saved to {args.results_db}")
    logger.info("m_TIES script completed successfully.")

if __name__ == "__main__":
    main()
