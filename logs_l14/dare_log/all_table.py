import json
import os

# 获取当前目录中所有的 JSON 文件
json_files = [f for f in os.listdir() if f.endswith(".json")]

# 定义需要显示的 8 个数据集
datasets = ["RESISC45", "Cars", "MNIST", "DTD", "EuroSAT", "GTSRB", "SUN397", "SVHN"]

# 格式化函数
def fmt(val):
    return f"{val:.2f}" if val is not None else ""

# 生成 LaTeX 表格
latex_table = r"""\begin{table*}[hbt!]
\centering
\caption{Accuracy of merging ViT-B/32 fine-tuned models evaluated on their fine-tuning datasets.}
\label{vitb32}
\resizebox{\textwidth}{!}{
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr}
\toprule
Method & """ + " & ".join(datasets) + " & Avg \\\\ \n" + r"\midrule" + "\n"

# 读取每个 JSON 文件并生成对应的行
for json_filename in json_files:
    # 读取 JSON 数据
    with open(json_filename, "r") as f:
        data = json.load(f)

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

    # 格式化各数据集的数值
    row_entries = [fmt(values[ds]) for ds in datasets]
    avg_entry = fmt(avg_val)

    # 添加当前 JSON 文件对应的行
    method_name = json_filename.replace(".json", "")  # 使用文件名作为方法名
    latex_table += method_name + " & " + " & ".join(row_entries) + " & " + avg_entry + r" \\" + "\n"

latex_table += r"""\bottomrule
\end{tabular*}
}
\end{table*}
"""

# 输出 LaTeX 表格
print(latex_table)
