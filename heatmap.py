import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# 重构数据
data = {
    'Weight': [0.001, 0.2, 0.33, 0.5, 1],
    'MNIST': [57.73, 94.52, 98.23, 98.79, 85.8],
    'DTD': [46.06, 54.31, 53.94, 43.4, 12.87],
    'EuroSAT': [57.7, 74.24, 72.74, 56.3, 26.69],
    'GTSRB': [37.78, 67.72, 77.35, 71.79, 26.37],
    'SUN397': [63.4, 64.88, 58.77, 32.3, 0.91],
    'SVHN': [37.48, 77.81, 86.42, 87.37, 68.89],
    'Avg': [50.025, 72.24666667, 74.575, 64.99166667, 36.92166667]
}

# 转换为 DataFrame
df = pd.DataFrame(data).set_index('Weight')

# 绘制热力图
plt.figure(dpi=500)
sns.heatmap(df.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'},annot_kws={'size': 10})
# 修改横纵坐标字体
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# 修改标题和坐标轴标签
plt.title('Performance Impact Across Weights and Datasets', fontsize=10, pad=15)  # pad增加标题与图之间的距离
plt.xlabel('Weight', fontsize=8, labelpad=10)  # labelpad增加与图的距离
plt.ylabel('Dataset', fontsize=8, labelpad=10)

# 调整布局，防止标题和标签被截断
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 使布局适应图形

# 保存图像
output_path = "heatmap_weight_horizontal.pdf"
plt.savefig(output_path)
plt.show()
