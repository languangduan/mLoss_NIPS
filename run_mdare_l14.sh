#!/bin/bash

# Set the model name
MODEL="ViT-L-14"

# Provide the dataset list as a single comma-separated string
EVAL_DATASETS="RESISC45,Cars,MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

# Set data storage path
DATA_LOCATION="datasets"

# Set the checkpoint save path
SAVE_PATH="checkpoints/${MODEL}"


K=0.7
E=0.1
SEED=42

# Set the results JSON path
RESULTS_DB="mdare_log/${MODEL}_seed${SEED}_k${K}_e${E}.json"

# Optionally, set the number of workers (default is 4)
NUM_WORKERS=4


# Create save directories if they don't exist
mkdir -p "${SAVE_PATH}"
mkdir -p "$(dirname "${RESULTS_DB}")"

# Run m_ties.py with the new --num-workers argument
python m_dare_l14.py \
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
    --num-workers "${NUM_WORKERS}"