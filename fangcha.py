import pandas as pd

# 原始数据（不含 Avg 列）
data = {
    "Method": [
        "M-TIES",
        "TIES",
        "Task Arithmetic",
        "Simple Averaging",
        "DARE",
        "Ensemble"
    ],
    "RESISC45": [72.60, 70.67, 71.27, 71.46, 69.97, 79.87],
    "Cars":     [61.07, 58.61, 60.70, 63.34, 57.98, 66.60],
    "MNIST":    [97.62, 98.30, 95.32, 87.46, 97.95, 95.80],
    "DTD":      [54.84, 54.20, 51.76, 50.11, 53.24, 58.30],
    "EuroSAT":  [82.02, 80.22, 79.74, 73.00, 78.89, 98.30],
    "GTSRB":    [72.44, 72.11, 67.32, 52.79, 72.00, 81.11],
    "SUN397":   [62.19, 59.01, 62.06, 64.91, 59.14, 66.35],
    "SVHN":     [83.06, 86.20, 76.68, 64.16, 83.96, 82.15]
}

# 构造 DataFrame 并计算行方差（人口方差，ddof=0）
df = pd.DataFrame(data).set_index("Method")
df["Variance"] = df.var(axis=1, ddof=0).round(2)

# 输出结果
print(df[["Variance"]])
