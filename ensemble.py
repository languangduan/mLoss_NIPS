import os
import torch
import json
import random
import logging
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from PIL import Image

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.datasets.common import maybe_dictionarize

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 或 logging.DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定义伪模型类：用于将多个检查点模型隐藏层输出进行平均
import torch.nn as nn


class PseudoEnsembleEncoder(nn.Module):
    def __init__(self, models, layer_name=None):
        """
        初始化伪模型，保存多个检查点模型，并指定提取隐藏层的层名（可选）
        Args:
            models (list[nn.Module]): 加载好的多个检查点模型
            layer_name (str, 可选): 指定提取隐藏层的名称，默认为 None，则取最后一层隐藏状态
        """
        super(PseudoEnsembleEncoder, self).__init__()
        self.models = models
        self.layer_name = layer_name

        # 从第一个模型继承所有属性
        self.base_model = models[0]

        self.train_preprocess = models[0].train_preprocess
        self.val_preprocess = models[0].val_preprocess
        # 将第一个模型的所有属性复制到当前模型
        for attr_name in dir(self.base_model):
            # 跳过私有属性、内置方法和forward方法
            if not attr_name.startswith('_') and attr_name != 'forward' and not callable(
                    getattr(self.base_model, attr_name)):
                try:
                    setattr(self, attr_name, getattr(self.base_model, attr_name))
                except AttributeError:
                    pass  # 如果无法设置某些属性，跳过

        # 设置所有模型为评估模式，避免计算梯度
        for model in self.models:
            model.eval()

    def forward(self, x):
        """
        对输入 x 依次通过各个模型，提取隐藏层输出，并对所有输出进行平均
        Args:
            x (Tensor): 输入图像数据
        Returns:
            Tensor: 平均后的隐藏层表示
        """
        hidden_states = []
        for model in self.models:
            with torch.no_grad():
                # 直接使用模型的前向传播
                output = model(x)
                hidden_states.append(output)

        # 对各模型的输出沿新维度堆叠后求均值
        hidden_avg = torch.mean(torch.stack(hidden_states, dim=0), dim=0)
        return hidden_avg


# 自定义 collate_fn，用于数据加载时处理图像
to_tensor = transforms.ToTensor()
resize_and_to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),           # 转换为张量
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # 灰度图扩展到3通道
])

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [resize_and_to_tensor(img) if isinstance(img, Image.Image) else img for img in inputs]
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels)
    return inputs, labels

def main():
    logger.info("Starting ensemble evaluation script using hidden layer averaging.")
    args = parse_arguments()
    args.seed = 42

    # ------------------------------
    # 固定随机种子，确保结果可重复
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

    # ------------------------------
    # 加载多个检查点模型
    # ------------------------------
    model_name = args.model
    dataset_names = args.eval_datasets

    # 这里假设我们同时加载 zeroshot（预训练）和各数据集对应的 finetuned 检查点
    pretrained_checkpoint = f'checkpoints/{model_name}/zeroshot.pt'
    finetuned_ckpts = [f'checkpoints/{model_name}/{ds}/finetuned.pt' for ds in dataset_names]

    logger.info(f"Loading pretrained model from: {pretrained_checkpoint}")
    # 加载预训练模型，并移动到指定设备
    pretrained_model = torch.load(pretrained_checkpoint, map_location=args.device)
    models = [pretrained_model]

    for ckpt in finetuned_ckpts:
        logger.info(f"Loading finetuned checkpoint from: {ckpt}")
        model = torch.load(ckpt, map_location=args.device)
        models.append(model)

    # 构造伪模型，对所有模型隐藏层进行平均
    ensemble_encoder = PseudoEnsembleEncoder(models, layer_name=None)
    logger.info("Ensemble encoder constructed successfully.")

    # ------------------------------
    # 使用 ensemble_encoder 进行数据集评估
    # ------------------------------
    evaluation_results = {}
    total_accuracy = 0.0
    num_datasets = 0

    for ds in dataset_names:
        logger.info(f"Evaluating on dataset '{ds}' using ensemble encoder.")
        # 调用 eval_single_dataset，内部会将 image_encoder 与分类头组合构成 ImageClassifier
        result = eval_single_dataset(ensemble_encoder, ds, args)
        evaluation_results[ds] = result

        if "top1" in result:
            total_accuracy += float(result["top1"])
            num_datasets += 1

        logger.info(f"Results for '{ds}': {result}")

    # 计算平均准确率
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
