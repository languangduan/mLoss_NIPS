import os
import torch
import json
import random
import logging
import numpy as np
from tqdm import tqdm

from src.args import parse_arguments
from src.eval import eval_single_dataset

import torch.nn as nn

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 或 logging.DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FinalOutputEnsembleEncoder(nn.Module):
    def __init__(self, models):
        """
        初始化集成编码器，传入多个 finetuned 模型检查点。假设各模型的 forward 方法返回最终输出（例如 logits）。
        同时从第一个模型复制 train_preprocess 和 val_preprocess 属性，
        以满足下游评估代码对图像预处理属性的要求。
        
        Args:
            models (list[nn.Module]): 加载好的多个 finetuned 模型检查点。
        """
        super(FinalOutputEnsembleEncoder, self).__init__()
        self.models = models
        # 设置所有模型为评估模式，禁用 dropout、梯度计算等
        for model in self.models:
            model.eval()
        # 复制预处理属性（如果存在）从第一个模型
        self.train_preprocess = getattr(models[0], "train_preprocess", None)
        self.val_preprocess = getattr(models[0], "val_preprocess", None)

    def forward(self, x):
        """
        对输入 x 分别通过各个模型计算最终输出，并对所有输出进行平均。
        
        Args:
            x (Tensor): 输入数据
        
        Returns:
            Tensor: 各模型最终输出的平均值
        """
        outputs = []
        for model in self.models:
            with torch.no_grad():
                out = model(x)  # 计算最终输出
                outputs.append(out)
        # 将所有模型输出沿新维度堆叠后求均值
        avg_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return avg_output

def main():
    logger.info("Starting final output ensemble evaluation script.")
    args = parse_arguments()
    
    # 固定随机种子，确保结果可重复
    SEED = args.seed if args.seed is not None else 42
    logger.info(f"Setting random seed to {SEED} for reproducibility.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = args.model
    dataset_names = args.eval_datasets  # 假设此参数为列表

    # 只加载 finetuned 检查点模型（不包含预训练模型）
    finetuned_ckpts = [f'checkpoints/{model_name}/{ds}/finetuned.pt' for ds in dataset_names]

    models = []
    for ckpt in finetuned_ckpts:
        logger.info(f"Loading finetuned checkpoint from: {ckpt}")
        model = torch.load(ckpt, map_location=args.device)
        models.append(model)

    if not models:
        logger.error("No finetuned models loaded. Please check your checkpoint paths.")
        return

    # 构造集成编码器：对各 finetuned 模型的最终输出求平均
    ensemble_encoder = FinalOutputEnsembleEncoder(models)
    logger.info("Final output ensemble encoder constructed successfully.")

    # 使用 ensemble_encoder 对各个数据集进行评估
    evaluation_results = {}
    total_accuracy = 0.0
    num_datasets = 0

    for ds in dataset_names:
        logger.info(f"Evaluating on dataset '{ds}' using ensemble encoder.")
        result = eval_single_dataset(ensemble_encoder, ds, args)
        evaluation_results[ds] = result
        if "top1" in result:
            total_accuracy += float(result["top1"])
            num_datasets += 1
        logger.info(f"Results for '{ds}': {result}")

    if num_datasets > 0:
        avg_acc = total_accuracy / num_datasets
        evaluation_results["avg_accuracy"] = avg_acc
        logger.info(f"Average accuracy across {num_datasets} datasets: {avg_acc:.4f}")
    else:
        logger.warning("No accuracy results found; cannot compute average accuracy.")

    evaluation_results["random_seed"] = SEED

    # 保存评估结果到 JSON 文件
    if args.results_db is not None:
        results_path = args.results_db
        dirname = os.path.dirname(results_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        logger.info(f"Evaluation results saved to {results_path}")
    else:
        logger.info("Results not saved (use --results_db to specify a path).")

if __name__ == "__main__":
    main()
