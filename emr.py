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

# --------------------------
# 配置与初始化
# --------------------------
args = parse_arguments()
model_name = args.model
datasets = args.eval_datasets  # 每个任务对应一个数据集
SEED = args.seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Setting random seed to {SEED} for reproducibility.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 加载预训练模型
pretrained_checkpoint = f'checkpoints/{model_name}/zeroshot.pt'
pretrained_model = torch.load(pretrained_checkpoint)
pretrained_state_dict = pretrained_model.state_dict()

# 加载各任务的任务向量（注意：TaskVector 类计算的是 fine-tuned 模型与预训练模型的差异）
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model_name}/{dataset}/finetuned.pt').vector
    for dataset in datasets
]

# --------------------------
# EMR-MERGING各步骤函数定义
# --------------------------

def elect_unified_task_vector(task_vectors):
    """
    对所有任务向量执行 Elect 操作，构造统一任务向量 τ_uni。
    对于每个参数 key：
      1. 堆叠所有任务向量：形状 (n, *tensor.shape)
      2. 计算各元素的总和，并取符号（若为 0 则设为 +1）
      3. 对于每个位置，在所有任务向量中选择与统一符号相同的最大绝对值
      4. 返回 τ_uni = gamma_uni ⊙ epsilon_uni
    """
    unified = {}
    keys = task_vectors[0].keys()  # 假设所有任务向量具有相同的 key
    for key in keys:
        # 堆叠所有任务向量
        tensor_stack = torch.stack([tv[key] for tv in task_vectors], dim=0)  # shape: (n, ...)
        # 求和并计算符号（若为0则设为1）
        sum_tensor = tensor_stack.sum(dim=0)
        gamma_uni = torch.where(sum_tensor == 0, torch.ones_like(sum_tensor), torch.sign(sum_tensor))
        
        # 扩展 gamma_uni 使其与 tensor_stack 形状一致
        gamma_uni_expanded = gamma_uni.unsqueeze(0).expand_as(tensor_stack)
        # 仅保留与 gamma_uni 符号一致的部分
        sign_mask = (torch.sign(tensor_stack) == gamma_uni_expanded).float()
        abs_stack = torch.abs(tensor_stack)
        masked_abs = abs_stack * sign_mask
        # 对每个位置取最大绝对值
        epsilon_uni, _ = masked_abs.max(dim=0)
        
        unified[key] = gamma_uni * epsilon_uni
    return unified

def compute_mask(task_vector, unified_tau):
    """
    对于单个任务向量，生成二值掩码 M = (τ_i ⊙ τ_uni > 0)
    即只有当任务向量和统一任务向量在相同位置的乘积大于0时，该位置保留1，否则为0。
    """
    mask = {}
    for key in task_vector.keys():
        mask[key] = (task_vector[key] * unified_tau[key] > 0).float()
    return mask

def compute_rescaler(task_vector, mask, unified_tau, eps=1e-8):
    """
    计算任务特定的缩放因子 λ：
      λ = (sum(abs(τ_i))) / (sum(abs(M ⊙ τ_uni)) + eps)
    """
    total_task = 0.0
    total_masked = 0.0
    for key in task_vector.keys():
        total_task += torch.sum(torch.abs(task_vector[key])).item()
        total_masked += torch.sum(torch.abs(mask[key] * unified_tau[key])).item()
    return total_task / (total_masked + eps)

def modulate_unified(task_vector, unified_tau):
    """
    针对单个任务，先计算掩码，再计算 rescaler，
    最后返回调制后的任务向量：λ * (M ⊙ τ_uni)
    同时返回计算得到的 λ 和掩码，便于调试。
    """
    mask = compute_mask(task_vector, unified_tau)
    lambda_i = compute_rescaler(task_vector, mask, unified_tau)
    modulated = {}
    for key in task_vector.keys():
        modulated[key] = lambda_i * mask[key] * unified_tau[key]
    return modulated, lambda_i, mask

def merge_model(pretrained_state, modulated_tau):
    """
    将调制后的任务向量加到预训练模型上，得到最终合并模型的参数。
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
    完整的 EMR-MERGING 流程：
      1. 根据所有任务向量构造统一任务向量 τ_uni（Elect）。
      2. 对于每个任务向量：
         - 计算任务特定的掩码 M 和缩放因子 λ
         - 得到调制后的任务向量：λ * (M ⊙ τ_uni)
         - 最终模型参数：Ŵ = W_pre + 调制后的任务向量
    返回每个任务对应的合并模型 state_dict 列表，以及统一任务向量（便于调试）。
    """
    unified_tau = elect_unified_task_vector(task_vectors)
    merged_models = []
    # 为每个任务分别计算调制后的模型
    for tv in task_vectors:
        modulated_tau, lambda_i, mask = modulate_unified(tv, unified_tau)
        merged_state = merge_model(pretrained_state, modulated_tau)
        merged_models.append(merged_state)
        logger.info(f"Computed lambda: {lambda_i:.4f}")
    return merged_models, unified_tau

# --------------------------
# 主程序
# --------------------------
logger.info("Starting EMR-MERGING process...")

# 进行 EMR-MERGING 得到每个任务的合并模型
merged_state_dicts, unified_tau = emr_merging(task_vectors, pretrained_state_dict)

# 评估每个合并模型在对应任务上的表现
evaluation_results = {}
for dataset, merged_state in zip(datasets, merged_state_dicts):
    logger.info(f"Evaluating merged model on {dataset}...")
    # 这里假设 TaskVector.apply_to 方法能够将 state_dict 应用到预训练模型中
    merged_model = TaskVector(vector=merged_state).apply_to(pretrained_checkpoint, scaling_coef=1)
    result = eval_single_dataset(merged_model, dataset, args)
    evaluation_results[dataset] = result

# 计算所有任务上的平均准确率
accuracies = [result['top1'] for result in evaluation_results.values()]
average_accuracy = sum(accuracies) / len(accuracies)
evaluation_results['average_accuracy'] = average_accuracy

# 将评估结果保存到JSON文件中
with open(args.results_db, 'w') as f:
    json.dump(evaluation_results, f, indent=4)

logger.info(f"Average accuracy across datasets: {average_accuracy * 100:.2f}%")
logger.info(f"Evaluation results saved to: {args.results_db}")
