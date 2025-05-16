# README

## Overview

This repository implements the M-loss framework, a novel evaluation metric that quantifies the compatibility of merging source models using limited unlabeled data. M-loss measures the discrepancy between parameter averaging and model ensembling at both layer and node levels, facilitating more effective merging strategies.

## Features

- Theoretical justification for model merging compatibility
- Evaluation metrics for model merging without labeled test data
- Hyperparameter selection guidance for merging methods
- Dynamic parameter pruning schedules for optimal model consolidation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/mloss.git
cd mloss

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

#### Download Checkpoints

Before running experiments, you need to download the pre-trained and fine-tuned checkpoints:

1. Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw?usp=share_link)
2. Place the downloaded checkpoints in the `checkpoints` directory:
   ```
   mkdir -p checkpoints
   # Extract downloaded files to checkpoints/
   ```

The checkpoints include CLIP models (ViT-B/32, ViT-B/16, and ViT-L/14) fine-tuned on eight downstream tasks: Stanford Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, and SVHN.

#### Build Classification Heads

For new datasets, you need to generate classification heads:

```bash
# Generate classification heads for new datasets
python build_head.py 
```

### Running Experiments

The repository provides several scripts for different model architectures:

```bash
# Run M-loss evaluation with ViT-B-32
bash run_mties_B32.sh

# Run M-loss evaluation with ViT-L-14
bash run_mties_L14.sh

# Run Task Arithmetic experiments
bash run_arith_B32.sh
bash run_arith_L14.sh

# Run DARE experiments
bash run_dare_B32.sh
bash run_dare_L14.sh

# Run M-DARE experiments
bash run_mdare_B32.sh
bash run_mdare_L14.sh
```

### Configuration

Key parameters for M-loss experiments:
- `k`: Controls pruning ratio (recommended range: 0.2-0.8)
- `e`: Controls fine-tuning intensity (recommended range: 0.05-0.2)

## Datasets

The framework supports multiple vision datasets including MNIST, DTD, EuroSAT, GTSRB, SUN397, SVHN, Stanford Cars, and RESISC45.

## Acknowledgements

This codebase is built upon the [Task Vectors](https://github.com/mlfoundations/task_vectors) repository. We extend our gratitude to the original authors for their foundational work on task arithmetic and model merging.

## Citation

If you find this work useful for your research, please cite our paper:

```
@article{anonymous2025mloss,
  title={M-loss: Quantifying Model Merging Compatibility With Limited Unlabeled Data},
  author={Anonymous},
  journal={NeurIPS},
  year={2025}
}
```

Please also consider citing the original Task Vectors paper:

```
@inproceedings{ilharco2022patching,
  title={Patching open-vocabulary models by interpolating weights},
  author={Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and others},
  booktitle={ICML},
  year={2022}
}
```

## License

[MIT License](LICENSE)