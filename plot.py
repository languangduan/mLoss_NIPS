import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置全局美观参数
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.family": "Arial",  # 设置字体
    "font.size": 10,          # 字体大小
    "axes.titlesize": 12,     # 标题大小
    "axes.labelsize": 10,     # 轴标签大小
    "xtick.labelsize": 9,     # x轴刻度字体大小
    "ytick.labelsize": 9,     # y轴刻度字体大小
    "legend.fontsize": 9,     # 图例字体大小
})

# 数据
groups = ['0.001', '0.2', '0.5', '0.7', '1', 'Avg', 'Ours']
vit_b32 = [50, 70, 55, 60, 30, 65, 80]
vit_b16 = [60, 75, 60, 50, 45, 75, 85]
vit_l14 = [70, 85, 70, 60, 55, 80, 90]

# 颜色选择（自然色系，色盲友好）
colors = ["#F28E2B", "#4E79A7", "#59A14F"]

# 创建图表
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
fig.subplots_adjust(wspace=0.3)  # 调整子图之间的间距

# ViT-B/32 子图
axes[0].bar(groups, vit_b32, color=colors[0], edgecolor='black', alpha=0.8)
axes[0].set_title("ViT-B/32")
axes[0].set_xlabel("Weight")
axes[0].set_ylabel("Accuracy")

# ViT-B/16 子图
axes[1].bar(groups, vit_b16, color=colors[1], edgecolor='black', alpha=0.8)
axes[1].set_title("ViT-B/16")
axes[1].set_xlabel("Weight")

# ViT-L/14 子图
axes[2].bar(groups, vit_l14, color=colors[2], edgecolor='black', alpha=0.8)
axes[2].set_title("ViT-L/14")
axes[2].set_xlabel("Weight")

# 调整整体布局
plt.tight_layout()
plt.savefig("optimized_vit_results.png", dpi=300, bbox_inches='tight')
plt.show()
