import os
import torch
import json
import random
import logging
import numpy as np
import copy
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

###############################################################################
# Select GPU = cuda:1 if available
###############################################################################
device = torch.device("cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else "cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

###############################################################################
# 1. Utility: Build Unlabeled Data Combined
###############################################################################
def build_unlabeled_data_combined(dataset_names, args, proportion=0.01, max_samples=500, min_samples=100):
    """
    Samples a small proportion (or up to max_samples) of training data from each dataset
    and combines them into a single unlabeled DataLoader for distribution-based row-wise scaling.
    """
    logger.info("Building an unlabeled combined DataLoader.")
    all_subsets = []
    for ds_name in dataset_names:
        logger.info(f"Loading dataset '{ds_name}' for unlabeled sampling.")
        dataset_obj = get_dataset(
            ds_name,
            preprocess=None,  # No transform for unlabeled data
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        train_data = dataset_obj.train_dataset
        total_len = len(train_data)

        # First compute the proportion
        sample_len = int(total_len * proportion)
        # Enforce min_samples and max_samples
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

    # Merge into one dataset
    combined_dataset = ConcatDataset(all_subsets)
    logger.info(f"Total combined unlabeled samples: {len(combined_dataset)}")

    # When running on CUDA, pin_memory=True can help speed up host-to-GPU copies
    pin_memory = (device.type == "cuda")

    # Create DataLoader (unlabeled)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory
    )
    logger.info("Unlabeled DataLoader created successfully.")
    return combined_loader

to_tensor = transforms.ToTensor()
resize_and_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),         # 转换为张量
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # 如果是灰度图，扩展到 3 通道
])

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [resize_and_to_tensor(img) if isinstance(img, Image.Image) else img for img in inputs]
    inputs = torch.stack(inputs)  # 堆叠成张量
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
    analyzer=LLFCAnalyzer(),
    scaling=1.0
):
    """
    Modified TIES method that merges multiple task vectors layer by layer.
    """
    logger.info("Starting m_TIES merging process.")
    logger.info(f"Base keep ratio k={k}, vibration coefficient e={e}.")

    # 1. Load the pretrained model directly on our selected device
    logger.info(f"Loading pretrained model from '{pretrained_checkpoint}'")
    pretrained_model = torch.load(pretrained_checkpoint, map_location=device)
    models = [pretrained_model]
    for i in finetuned_checkpoints:
        # Load each finetuned model on the same device
        models.append(torch.load(i, map_location=device))

    pretrained_sd = pretrained_model.state_dict()
    # copy.deepcopy() the pretrained model on device
    merged_model = copy.deepcopy(pretrained_model)

    # 2. Create the list of task vectors
    all_task_vectors = []
    for ft_ckpt in finetuned_checkpoints:
        logger.info(f"Loading finetuned checkpoint '{ft_ckpt}'")
        tv = TaskVector(pretrained_checkpoint, ft_ckpt).vector
        all_task_vectors.append(tv)

    # 3. Merge each parameter key
    param_keys = list(pretrained_sd.keys())
    merged_vector = {}
    detected_layers = []

    for param_key in param_keys:
        param_tvs = [tv[param_key] for tv in all_task_vectors if param_key in tv]
        if not param_tvs:
            logger.debug(f"No task vectors found for parameter '{param_key}'. Skipping.")
            continue

        # Check rowwise
        is_rowwise_candidate = (
            rowwise_target_layers
            and any(rl in param_key for rl in rowwise_target_layers)
            and param_key.endswith(".weight")
        )

        if is_rowwise_candidate:
            logger.info(f"Parameter '{param_key}' is a rowwise target. Applying rowwise robust trimming.")

            data_iter = iter(sample_dataloader)
            N = len(models) - 1
            M_l_sum = None

            for _ in range(N):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(sample_dataloader)
                    batch = next(data_iter)

                inputs, labels = batch
                # If you want to place them on the same device:
                inputs = inputs.to(device)
                weights = [1 / (len(models) - 1) for _ in range(len(models) - 1)]
                layer_key = param_key[:-len('.weight')]

                M_l = analyzer.compute_llfc_loss_by_row(inputs, models, weights, layer_key)

                if M_l_sum is None:
                    M_l_sum = M_l.clone()
                else:
                    M_l_sum += M_l

            M_l = M_l_sum / N
            trimmed_tvs = rowwise_robust_trim_task_vector(param_tvs, k, e, sample_dataloader, M_l)
            detected_layers.append(param_key)
        else:
            # Standard TIES trimming
            trimmed_tvs = [ties_trim_vector(ptv, k) for ptv in param_tvs]

        # Elect sign + disjoint merge
        merged_param = ties_elect_and_merge(trimmed_tvs)
        merged_vector[param_key] = merged_param
        # Add to merged_model
        merged_model.state_dict()[param_key] += scaling * merged_param.to(device)

    logger.info("m_TIES merging process completed.")
    return merged_vector, detected_layers

###############################################################################
# 3. Example Helper Functions
###############################################################################
def ties_trim_vector(tensor, k):
    """
    Standard TIES trimming logic on a single Tensor (the task vector).
    Keep top-k% by magnitude, set others to 0.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor  # In case it's not a Tensor

    flat = tensor.view(-1).abs()
    num_keep = int(len(flat) * k)
    if num_keep <= 0:
        return torch.zeros_like(tensor)
    if num_keep >= len(flat):
        return tensor.clone()

    topk_vals, _ = torch.topk(flat, num_keep, largest=True, sorted=False)
    threshold = topk_vals.min()
    mask = tensor.abs() >= threshold
    trimmed = tensor * mask
    return trimmed

def robust_scale_losses(row_losses: torch.Tensor) -> torch.Tensor:
    # unchanged
    median = torch.median(row_losses)
    q75 = torch.quantile(row_losses, 0.75)
    q25 = torch.quantile(row_losses, 0.25)
    iqr = q75 - q25
    if iqr == 0:
        return torch.zeros_like(row_losses)
    scaled = (row_losses - median) / iqr
    min_val = scaled.min()
    max_val = scaled.max()
    normalized = (scaled - min_val) / (max_val - min_val)
    return normalized

def rank_scale(row_losses: torch.Tensor) -> torch.Tensor:
    """基于排序的缩放 (支持 GPU 和 CPU)"""
    device = row_losses.device  # 获取输入张量所在的设备

    # 排序并生成排名
    sorted_indices = row_losses.argsort()
    ranks = torch.zeros_like(sorted_indices, dtype=torch.float32, device=device)  # 确保 ranks 在相同设备上
    ranks[sorted_indices] = torch.arange(len(row_losses), dtype=torch.float32, device=device)  # 将 ranks 移动到正确设备

    # 归一化到 [0, 1]
    normalized = ranks / (len(row_losses) - 1)
    return normalized

def log_scale(row_losses: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # unchanged
    row_losses = row_losses + eps
    scaled = torch.log(row_losses)
    min_val = scaled.min()
    max_val = scaled.max()
    normalized = (scaled - min_val) / (max_val - min_val)
    return normalized

def rowwise_robust_trim_task_vector(param_tvs, k, e, sample_dataloader, M_l=None):
    # unchanged logic
    logging.info(" rowwise robust trimming for param_tvs. Using M_l metric.")
    shape = param_tvs[0].shape
    out_dim = shape[0]
    # if M_l is None:
    #     M_l = torch.rand(out_dim)

    row_keep_ratios = k - e * rank_scale(M_l)
    logging.info(f"Row keep ratios: {row_keep_ratios}")

    trimmed_tvs = []
    for idx, ptv in enumerate(param_tvs):
        w_clone = ptv.clone()
        out_dim, in_dim = w_clone.shape
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
                row_mask = (row_abs >= thr)
                w_clone[r_i, :] = w_clone[r_i, :] * row_mask
        trimmed_tvs.append(w_clone)
    return trimmed_tvs

def ties_elect_and_merge(trimmed_tensors):
    # unchanged
    if not trimmed_tensors:
        return None
    stk = torch.stack(trimmed_tensors, dim=0)
    pos_mass = stk.clamp(min=0).sum(dim=0)
    neg_mass = (-stk).clamp(min=0).sum(dim=0)
    elected_sign = torch.where(pos_mass >= neg_mass, 1.0, -1.0)

    merged_tensor = torch.zeros_like(elected_sign)
    contribution_counts = torch.zeros_like(elected_sign, dtype=torch.float)

    num_models = len(trimmed_tensors)
    for i in range(num_models):
        match_mask = (torch.sign(trimmed_tensors[i]) == elected_sign)
        merged_tensor += torch.where(match_mask, trimmed_tensors[i], torch.zeros_like(trimmed_tensors[i]))
        contribution_counts += match_mask.float()

    contribution_counts = torch.where(contribution_counts == 0, torch.ones_like(contribution_counts), contribution_counts)
    merged_param = merged_tensor / contribution_counts
    return merged_param

###############################################################################
# 4. Full Demo Script "main"
###############################################################################
def main():
    logger.info("Starting m_TIES script.")
    args = parse_arguments()

    # ------------------------------
    # Set fixed random seeds for reproducibility
    # ------------------------------
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

    # ------------------------------
    model_name = args.model
    dataset_names = args.eval_datasets
    pretrained_checkpoint = f'checkpoints/{model_name}/zeroshot.pt'
    finetuned_ckpts = [f'checkpoints/{model_name}/{ds}/finetuned.pt' for ds in dataset_names]

    # 1. Build unlabeled combined dataloader (with min_samples=100, max_samples=500)
    unlabeled_loader = build_unlabeled_data_combined(
        dataset_names,
        args,
        proportion=0.01,
        max_samples=500,
        min_samples=100
    )
    logger.info("Unlabeled data loader built successfully.")

    # 2. Perform m_TIES:
    rowwise_target_layers = [
        f"model.visual.transformer.resblocks.{i}.mlp.c_fc.weight" for i in range(24)
    ]
    merged_vector, detected_layers = m_ties_merging(
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoints=finetuned_ckpts,
        sample_dataloader=unlabeled_loader,
        k=args.k,
        e=args.e,
        rowwise_target_layers=rowwise_target_layers
    )

    # 3. Apply merged task vector
    from src.task_vectors import TaskVector
    final_tv = TaskVector(vector=merged_vector)
    merged_model = final_tv.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    logger.info("Merged model constructed successfully.")

    # 4. Evaluate on each dataset
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

    # 5a. Add layer detection info
    if detected_layers:
        logger.info(f"Detected rowwise-trimmed layers: {detected_layers}")
        evaluation_results["rowwise_trimmed_layers"] = detected_layers
    else:
        warning_msg = "Warning: No target layers were detected for row-wise trimming."
        logger.warning(warning_msg)
        evaluation_results["rowwise_trimmed_layers"] = warning_msg

    # 5b. Compute average accuracy
    if num_datasets > 0:
        avg_acc = total_accuracy / num_datasets
        evaluation_results["avg_accuracy"] = avg_acc
        logger.info(f"Average accuracy across {num_datasets} datasets: {avg_acc:.4f}")
    else:
        logger.warning("No 'top1' keys found in evaluation results; cannot compute average accuracy.")

    evaluation_results["random_seed"] = SEED
    logger.info(f"Random seed used: {SEED}")

    # 6. Save results
    with open(args.results_db, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    logger.info(f"Evaluation results saved to {args.results_db}")

    logger.info("m_TIES script completed successfully.")

if __name__ == "__main__":
    main()
