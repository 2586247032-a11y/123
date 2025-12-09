import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
chinese_columns = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
df_renamed = df.rename(columns=dict(zip(iris.feature_names, chinese_columns)))

sepal_width = df_renamed['花萼宽度']
normalized_sepal_width = min_max_normalization(sepal_width)

print("原始数据统计:")
print(f"最小值: {np.min(sepal_width):.4f}")
print(f"最大值: {np.max(sepal_width):.4f}")
print(f"均值: {np.mean(sepal_width):.4f}")

print("\n规范化后数据统计:")
print(f"最小值: {np.min(normalized_sepal_width):.4f}")
print(f"最大值: {np.max(normalized_sepal_width):.4f}")
print(f"均值: {np.mean(normalized_sepal_width):.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(sepal_width, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('原始花萼宽度分布')
plt.xlabel('花萼宽度 (cm)')
plt.ylabel('频次')

plt.subplot(1, 2, 2)
plt.hist(normalized_sepal_width, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Min-Max规范化后分布')
plt.xlabel('规范化值')
plt.ylabel('频次')

plt.tight_layout()
plt.show()



