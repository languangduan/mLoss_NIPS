#!/bin/bash

# Set model name
MODEL="ViT-B-32"

# Set evaluation dataset list, separated by commas (including Cars and RESISC45)
EVAL_DATASETS="RESISC45,Cars,MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

# Set data storage path
DATA_LOCATION="datasets"

# Set checkpoint save path
SAVE_PATH="checkpoints/${MODEL}"
SCALING=0.5
SEED=42
# Set result save path (optional)
RESULTS_DB="logs/arith_log/${MODEL}_scaling${SCALING}_seed${SEED}.json"

# Create save directory if it does not exist
mkdir -p "${SAVE_PATH}"
mkdir -p "$(dirname "${RESULTS_DB}")"

# Run task_arith.py
python task_arith.py \
    --seed "${SEED}" \
    --scaling "${SCALING}" \
    --model "${MODEL}" \
    --eval-datasets "${EVAL_DATASETS}" \
    --save "${SAVE_PATH}" \
    --data-location "${DATA_LOCATION}" \
    --batch-size 128 \
    --lr 0.001 \
    --wd 0.1 \
    --epochs 10 \
    --results-db "${RESULTS_DB}"
