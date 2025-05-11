#!/bin/bash

# Set the model name
MODEL="ViT-B-32"

# Provide the dataset list as a single comma-separated string
EVAL_DATASETS="RESISC45,Cars,MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

# Set data storage path
DATA_LOCATION="datasets"

# Set the checkpoint save path
SAVE_PATH="checkpoints/${MODEL}"


K=0.8
E=0.1
SEED=42

SAMPLING=1
DEVICE="cuda:2"
# Set the results JSON path
RESULTS_DB="logs/mties_log/${MODEL}_seed${SEED}_k${K}_e${E}_sam${SAMPLING}.json"

# Optionally, set the number of workers (default is 4)
NUM_WORKERS=4


# Create save directories if they don't exist
mkdir -p "${SAVE_PATH}"
mkdir -p "$(dirname "${RESULTS_DB}")"

# Run m_ties.py with the new --num-workers argument
python m_ties.py \
    --model "${MODEL}" \
    --eval-datasets "${EVAL_DATASETS}" \
    --save "${SAVE_PATH}" \
    --data-location "${DATA_LOCATION}" \
    --k "${K}" \
    --e "${E}" \
    --seed "${SEED}" \
    --batch-size 128 \
    --lr 0.001 \
    --wd 0.1 \
    --epochs 10 \
    --results-db "${RESULTS_DB}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --sampling-size "${SAMPLING}"
