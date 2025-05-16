#!/bin/bash

# Set model name
MODEL="ViT-L-14"

# Set evaluation dataset list, separated by commas (including Cars and RESISC45)
EVAL_DATASETS="RESISC45,Cars,MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

# Set data storage path
DATA_LOCATION="datasets"

# Set checkpoint save path
SAVE_PATH="checkpoints/${MODEL}"
SEED=42
K=0.7
LAYERWISE="False"
DEVICE="cuda:2"
# Set result save path (optional)
RESULTS_DB="logs_l14/dare_log/${MODEL}_k${K}_seed${SEED}_${LAYERWISE}.json"

# Create save directory if it does not exist
mkdir -p "${SAVE_PATH}"
mkdir -p "$(dirname "${RESULTS_DB}")"

# Run dare.py
python dare.py \
    --model "${MODEL}" \
    --eval-datasets "${EVAL_DATASETS}" \
    --save "${SAVE_PATH}" \
    --data-location "${DATA_LOCATION}" \
    --batch-size 128 \
    --lr 0.001 \
    --wd 0.1 \
    --epochs 10 \
    --results-db "${RESULTS_DB}" \
    --k "${K}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --layerwise "${LAYERWISE}"
