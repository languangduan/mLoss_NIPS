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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import torch.nn as nn

class PseudoEnsembleEncoder(nn.Module):
    def __init__(self, models, layer_name=None):
        """
        Ensemble encoder that averages the hidden layer outputs of multiple models.
        Args:
            models (list[nn.Module]): List of loaded checkpoint models.
            layer_name (str, optional): Name of the hidden layer to extract. If None, use the last layer output.
        """
        super(PseudoEnsembleEncoder, self).__init__()
        self.models = models
        self.layer_name = layer_name

        self.base_model = models[0]
        self.train_preprocess = models[0].train_preprocess
        self.val_preprocess = models[0].val_preprocess
        # Copy all non-private, non-callable, non-forward attributes from base model
        for attr_name in dir(self.base_model):
            if not attr_name.startswith('_') and attr_name != 'forward' and not callable(getattr(self.base_model, attr_name)):
                try:
                    setattr(self, attr_name, getattr(self.base_model, attr_name))
                except AttributeError:
                    pass

        for model in self.models:
            model.eval()

    def forward(self, x):
        """
        Pass input x through each model, extract hidden layer outputs, and average them.
        Args:
            x (Tensor): Input image data.
        Returns:
            Tensor: Averaged hidden layer representation.
        """
        hidden_states = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                hidden_states.append(output)
        hidden_avg = torch.mean(torch.stack(hidden_states, dim=0), dim=0)
        return hidden_avg

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

def main():
    logger.info("Starting ensemble evaluation script using hidden layer averaging.")
    args = parse_arguments()
    args.seed = 42

    SEED = args.seed
    logger.info(f"Setting random seed to {SEED} for reproducibility.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = args.model
    dataset_names = args.eval_datasets

    pretrained_checkpoint = f'checkpoints/{model_name}/zeroshot.pt'
    finetuned_ckpts = [f'checkpoints/{model_name}/{ds}/finetuned.pt' for ds in dataset_names]

    logger.info(f"Loading pretrained model from: {pretrained_checkpoint}")
    pretrained_model = torch.load(pretrained_checkpoint, map_location=args.device)
    models = [pretrained_model]

    for ckpt in finetuned_ckpts:
        logger.info(f"Loading finetuned checkpoint from: {ckpt}")
        model = torch.load(ckpt, map_location=args.device)
        models.append(model)

    ensemble_encoder = PseudoEnsembleEncoder(models, layer_name=None)
    logger.info("Ensemble encoder constructed successfully.")

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
