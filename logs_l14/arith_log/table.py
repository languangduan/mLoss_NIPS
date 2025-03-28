import json

# JSON 文件名称
json_filename = "ViT-B-32_k0.7_seed42_False.json"

# 读取 JSON 数据
with open(json_filename, "r") as f:
    data = json.load(f)

# 定义需要显示的 8 个数据集
datasets = ["RESISC45", "Cars", "MNIST", "DTD", "EuroSAT", "GTSRB", "SUN397", "SVHN"]

# 提取各数据集的 top1 准确率（乘以 100 显示百分比），如果不存在则设置为空
values = {}
for ds in datasets:
    if ds in data and "top1" in data[ds]:
        values[ds] = data[ds]["top1"] * 100
    else:
        values[ds] = None

# 如果 JSON 中有 "avg_accuracy"，则使用该值；否则计算这 8 个数据集的平均值
if "avg_accuracy" in data:
    avg_val = data["avg_accuracy"] * 100
else:
    valid_vals = [v for v in values.values() if v is not None]
    avg_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

def fmt(val):
    return f"{val:.2f}" if val is not None else ""

# 格式化各数据集的数值
row_entries = [fmt(values[ds]) for ds in datasets]
avg_entry = fmt(avg_val)

# 使用 tabular* 环境并设置 @{\extracolsep{\fill}} 均分表格宽度，再用 \resizebox 缩放到 \textwidth
latex_table = r"""\begin{table*}[hbt!]
\centering
\caption{Accuracy of merging ViT-B/32 fine-tuned models evaluated on their fine-tuning datasets.}
\label{vitb32}
\resizebox{\textwidth}{!}{
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr}
\toprule
Method & """ + " & ".join(datasets) + " & Avg \\\\ \n" + r"\midrule" + "\n"
latex_table += "M-TIES & " + " & ".join(row_entries) + " & " + avg_entry + r" \\" + "\n"
latex_table += r"""\bottomrule
\end{tabular*}
}
\end{table*}
"""

print(latex_table)
