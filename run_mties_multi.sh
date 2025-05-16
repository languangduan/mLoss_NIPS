#!/bin/bash

# Set model name
MODEL="ViT-B-32"

# Provide a comma-separated list of datasets
EVAL_DATASETS="RESISC45,Cars,MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

# Set data storage path
DATA_LOCATION="datasets"

# Set checkpoint save path
SAVE_PATH="checkpoints/${MODEL}"

# Fixed random seed
SEED=42
SAMPLING=1
DEVICE="cuda:2"
# Set number of worker threads (optional)
NUM_WORKERS=4

# Define the list of e values to iterate
E_VALUES=(0.05 0.075 0.1 0.125 0.15)

# Define the list of k values to iterate (extended)
K_VALUES=(0.2 0.4 0.6 0.8)

# Iterate over each e value
for E in "${E_VALUES[@]}"; do
    # Iterate over each k value
    for K in "${K_VALUES[@]}"; do
        # Set result save path
        RESULTS_DB="mties_log/${MODEL}_seed${SEED}_k${K}_e${E}.json"

        # Create save directory (if it does not exist)
        mkdir -p "${SAVE_PATH}"
        mkdir -p "$(dirname "${RESULTS_DB}")"

        # Run m_ties.py
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
    done
done
