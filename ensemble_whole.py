import os
import torch
import json
import random
import logging
import numpy as np

from src.args import parse_arguments
from src.eval import eval_single_dataset

import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FinalOutputEnsembleEncoder(nn.Module):
    def __init__(self, models):
        """
        Ensemble encoder that averages the final outputs (e.g., logits) of multiple models.
        Args:
            models (list[nn.Module]): List of loaded finetuned checkpoint models.
        """
        super(FinalOutputEnsembleEncoder, self).__init__()
        self.models = models
        for model in self.models:
            model.eval()
        self.train_preprocess = getattr(models[0], "train_preprocess", None)
        self.val_preprocess = getattr(models[0], "val_preprocess", None)

    def forward(self, x):
        """
        Pass input x through each model and average the final outputs.
        Args:
            x (Tensor): Input data
        Returns:
            Tensor: Averaged final outputs of all models
        """
        outputs = []
        for model in self.models:
            with torch.no_grad():
                out = model(x)
                outputs.append(out)
        avg_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return avg_output

def main():
    logger.info("Starting final output ensemble evaluation script.")
    args = parse_arguments()
    
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
    dataset_names = args.eval_datasets

    finetuned_ckpts = [f'checkpoints/{model_name}/{ds}/finetuned.pt' for ds in dataset_names]

    models = []
    for ckpt in finetuned_ckpts:
        logger.info(f"Loading finetuned checkpoint from: {ckpt}")
        model = torch.load(ckpt, map_location=args.device)
        models.append(model)

    if not models:
        logger.error("No finetuned models loaded. Please check your checkpoint paths.")
        return

    ensemble_encoder = FinalOutputEnsembleEncoder(models)
    logger.info("Final output ensemble encoder constructed successfully.")

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
