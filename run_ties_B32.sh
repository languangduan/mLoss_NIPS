#!/bin/bash

# 设置模型名称
MODEL="ViT-B-32"

# 设置数据集列表，逗号分隔（包括 Cars 和 RESISC45）
EVAL_DATASETS="MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

# 设置数据存放路径
DATA_LOCATION="datasets"

# 设置检查点保存路径
SAVE_PATH="checkpoints/${MODEL}"
SEED=42
K=0.2
LAYERWISE="False"
# 设置结果保存路径（可选）
RESULTS_DB="ties_log/${MODEL}_k${K}_seed${SEED}_${LAYERWISE}.json"

# 创建保存目录（如果不存在）
mkdir -p "${SAVE_PATH}"
mkdir -p "$(dirname "${RESULTS_DB}")"

# 运行 ties.py
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
