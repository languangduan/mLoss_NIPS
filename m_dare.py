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

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

###############################################################################
# 1. Utility: Build Unlabeled Data Combined
###############################################################################
def build_unlabeled_data_combined(dataset_names, args, proportion=0.01, max_samples=500, min_samples=100):
    """
    Samples a small proportion (or up to max_samples) of training data from each dataset
    and combines them into a single unlabeled DataLoader for distribution-based row-wise scaling.

    Args:
        dataset_names (list[str]): A list of dataset names (e.g., ['MNIST','SVHN',...]).
        args: Command-line arguments (device, data_location, batch_size, etc.).
        proportion (float): Fraction of each dataset's train split to sample.
        max_samples (int): Caps the maximum number of samples per dataset (default: 500).
        min_samples (int): Ensures a minimum number of samples per dataset (default: 100).

    Returns:
        DataLoader: A DataLoader containing combined unlabeled samples from all datasets, ignoring labels.
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

    # Create DataLoader (unlabeled)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn  # 添加自定义 collate_fn
    )
    logger.info("Unlabeled DataLoader created successfully.")
    return combined_loader



to_tensor = transforms.ToTensor()
resize_and_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),        # 转换为张量
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
    analyzer = LLFCAnalyzer(device = args.device),
    scaling = 1.0
):
    """
    Modified TIES method that merges multiple task vectors layer by layer, trimming
    the *task vectors* (finetuned - pretrained) instead of trimming the raw weights.
    In rowwise_target_layers, we do the row-wise robust scaling logic *only for the .weight param*
    if the layer is in rowwise_target_layers. For biases, we use standard TIES trimming.

    Returns:
        merged_parameters (dict): The final merged task vector dictionary,
            which can be applied to the pretrained checkpoint to get the merged model.

        detected_layers (list[str]): The list of param_key (layer names) that received row-wise trimming.
    """
    logger.info("Starting m_TIES merging process.")
    logger.info(f"Base keep ratio k={k}, vibration coefficient e={e}.")

    # 1. Load the pretrained model
    logger.info(f"Loading pretrained model from '{pretrained_checkpoint}'")
    pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu')
    models = [pretrained_model]
    for i in finetuned_checkpoints:
        models.append(torch.load(i,map_location = 'cpu'))
    pretrained_sd = pretrained_model.state_dict()
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
    
    # Keep track of which layers actually received row-wise trimming
    detected_layers = []

    for param_key in param_keys:
        param_tvs = [tv[param_key] for tv in all_task_vectors if param_key in tv]
        if not param_tvs:
            logger.debug(f"No task vectors found for parameter '{param_key}'. Skipping.")
            continue

        # Check if this param_key is in rowwise_target_layers AND is a .weight param
        is_rowwise_candidate = (
            rowwise_target_layers
            and any(rl in param_key for rl in rowwise_target_layers)
            and param_key.endswith(".weight")
        )

        if is_rowwise_candidate:
            logger.info(f"Parameter '{param_key}' is a rowwise target. Applying rowwise robust trimming.")




            data_iter = iter(sample_dataloader)
            N = len(models)-1  # 使用的批次数量

            # 初始化累积变量
            M_l_sum = None  # 确保形状与 M_l 的返回值相同

            # 循环获取多个批次数据
            for _ in range(N):
                try:
                    batch = next(data_iter)  # 获取一个批次数据
                except StopIteration:
                    # 如果 DataLoader 的数据不足 N 个批次，则重新初始化迭代器
                    data_iter = iter(sample_dataloader)
                    batch = next(data_iter)

                inputs, labels = batch
                weights = [1 / (len(models) - 1) for _ in range(len(models) - 1)]
                layer_key = param_key[:-len('.weight')]

                # 计算当前批次的 M_l
                M_l = analyzer.compute_llfc_loss_by_row(inputs, models, weights, layer_key)

                # 累积 M_l
                if M_l_sum == None:
                    M_l_sum = M_l.clone()
                else:
                    M_l_sum += M_l

            # 计算最终的 M_l 均值
            M_l = M_l_sum / N

            trimmed_tvs = rowwise_robust_trim_task_vector(param_tvs, k, e, sample_dataloader,M_l)
            detected_layers.append(param_key)
        else:
            # For biases or other parameters, do standard TIES trimming
            trimmed_tvs = [dare_trim_vector(ptv, k) for ptv in param_tvs]

        # Elect sign + disjoint merge
        merged_param = ties_elect_and_merge(trimmed_tvs)
        merged_vector[param_key] = merged_param
        merged_model.state_dict()[param_key] += scaling* merged_param

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


def dare_trim_vector(tensor, k):
    """
    Now using DARE random trimming on a single Tensor (the task vector).
    Randomly keep a fraction k of elements (Bernoulli mask), set the rest to zero,
    and rescale the kept portion by 1/k so that the expected magnitude is unchanged.
    """
    # Optional: log or print a message to indicate we're using random trimming
    

    if not isinstance(tensor, torch.Tensor):
        return tensor  # In case it's not a Tensor

    # Generate a random mask with probability k to keep each element
    mask = torch.bernoulli(torch.full_like(tensor, k))  # same shape as tensor, each entry ~ Bernoulli(k)

    # Apply the mask
    trimmed = tensor * mask

    # Rescale the kept entries by 1/k
    # (Guard against division by zero if k is extremely small)
    if k > 1e-12:
        trimmed = trimmed * (1.0 / k)

    return trimmed

def robust_scale_losses(row_losses: torch.Tensor) -> torch.Tensor:
    """使用鲁棒缩放标准化损失值 (输入和输出均为 Tensor)

    Args:
        row_losses (torch.Tensor): 原始损失值张量

    Returns:
        torch.Tensor: 标准化后的损失值张量
    """
    # 计算中位数
    median = torch.median(row_losses)
    
    # 计算四分位数
    q75 = torch.quantile(row_losses, 0.75)
    q25 = torch.quantile(row_losses, 0.25)
    iqr = q75 - q25  # 四分位距

    if iqr == 0:
        # 如果 IQR 为 0，则所有值相同，返回全 0 张量
        return torch.zeros_like(row_losses)

    # 鲁棒缩放
    scaled = (row_losses - median) / iqr

    # 可选：限制缩放后的值范围
    #scaled = torch.clamp(scaled, -3, 3)

    # 归一化到 [0, 1]
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
    """对数缩放"""
    row_losses = row_losses + eps  # 防止 log(0)
    scaled = torch.log(row_losses)

    # 归一化到 [0, 1]
    min_val = scaled.min()
    max_val = scaled.max()
    normalized = (scaled - min_val) / (max_val - min_val)

    return normalized



def rowwise_robust_trim_task_vector(param_tvs, k, e, sample_dataloader, M_l = None):
    """
    Placeholder for row-wise robust scaling approach on *task vectors* for a single parameter.
    In actual usage, partial forward computations for pre-activation values would happen here.
    """
    logging.info(" rowwise robust trimming for param_tvs. Using M_l metric.")
    shape = param_tvs[0].shape
    out_dim = shape[0]
    in_dim = shape[1] if len(shape) > 1 else 1

    if M_l == None:
        M_l = torch.rand(out_dim)
    row_keep_ratios = k-e*rank_scale(M_l)

    # if out_dim == 1:
    #     row_keep_ratios[0] = k
    # else:
    #     for i in range(out_dim):
    #         alpha = ranks[i].item() / (out_dim - 1)
    #         row_keep = (k + e) - 2.0 * e * alpha
    #         row_keep = min(1.0, max(0.0, row_keep))
    #         row_keep_ratios[i] = row_keep

    logging.info(f"Row keep ratios: {row_keep_ratios}")

    trimmed_tvs = []
    for idx, ptv in enumerate(param_tvs):
        w_clone = ptv.clone()
        for r_i in range(out_dim):
            keep_r = row_keep_ratios[r_i].item()
            if len(w_clone.shape) == 1:
                # For biases or single-dimension parameters
                w_clone[r_i] = dare_trim_vector(w_clone[r_i], keep_r)
            else:
                # For weight matrices
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
    """
    After trimming, do sign election + disjoint merge across multiple task vectors
    to produce a single merged tensor.
    """
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
        match_mask = torch.sign(trimmed_tensors[i]) == elected_sign
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
    # Ensure deterministic behavior in PyTorch
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
    #    Only row-wise trim the 'c_fc.weight' for each of the 12 resblocks.
    #    The matching pattern is the substring "mlp.c_fc.weight".
    rowwise_target_layers = [
        f"model.visual.transformer.resblocks.{i}.mlp.c_fc.weight" for i in range(12)
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

        # Suppose result['top1'] is the accuracy (0.0 ~ 1.0).
        if "top1" in result:
            total_accuracy += float(result["top1"])
            num_datasets += 1

        logger.info(f"Results for '{ds}': {result}")

    # 5a. Add layer detection info to the final JSON output
    if detected_layers:
        logger.info(f"Detected rowwise-trimmed layers: {detected_layers}")
        evaluation_results["rowwise_trimmed_layers"] = detected_layers
    else:
        warning_msg = "Warning: No target layers were detected for row-wise trimming."
        logger.warning(warning_msg)
        evaluation_results["rowwise_trimmed_layers"] = warning_msg

    # 5b. Compute average accuracy across all evaluated test sets
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

    logger.info("m_DARE script completed successfully.")
  


if __name__ == "__main__":
    main()
