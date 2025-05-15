#!/bin/bash

# 设置模型名称
MODEL="ViT-B-32"

# 提供以逗号分隔的数据集列表
EVAL_DATASETS="RESISC45,Cars,MNIST,DTD,EuroSAT,GTSRB,SUN397,SVHN"

# 设置数据存储路径
DATA_LOCATION="datasets"

# 设置检查点保存路径
SAVE_PATH="checkpoints/${MODEL}"

# 固定的随机种子
SEED=42
SAMPLING=1
DEVICE="cuda:2"
# 设置工作线程数量（可选）
NUM_WORKERS=4

# 定义要遍历的 e 值列表
E_VALUES=(0.05 0.075 0.1 0.125 0.15)

# 定义要遍历的 k 值列表（扩展后）
K_VALUES=(0.2 0.4 0.6 0.8)

# 遍历每个 e 值
for E in "${E_VALUES[@]}"; do
    # 遍历每个 k 值
    for K in "${K_VALUES[@]}"; do
        # 设置结果保存路径
        RESULTS_DB="mties_log/${MODEL}_seed${SEED}_k${K}_e${E}.json"

        # 创建保存目录（如果不存在）
        mkdir -p "${SAVE_PATH}"
        mkdir -p "$(dirname "${RESULTS_DB}")"

        # 运行 m_ties.py
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
