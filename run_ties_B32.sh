#!/bin/bash

MODEL="ViT-B-32"

EVAL_DATASETS="RESISC45,Cars,MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

DATA_LOCATION="datasets"

SAVE_PATH="checkpoints/${MODEL}"
SEED=42
K=0.2
LAYERWISE="False"

RESULTS_DB="logs/ties_log/${MODEL}_k${K}_seed${SEED}_${LAYERWISE}.json"

mkdir -p "${SAVE_PATH}"
mkdir -p "$(dirname "${RESULTS_DB}")"

python ties.py \
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
    --layerwise "${LAYERWISE}"
