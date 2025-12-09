import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
chinese_columns = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
df_renamed = df.rename(columns=dict(zip(iris.feature_names, chinese_columns)))

sepal_width = df_renamed['花萼宽度'].values.reshape(-1, 1)

scaler = StandardScaler()
standardized_sepal_width = scaler.fit_transform(sepal_width).flatten()

print("原始数据统计:")
print(f"均值: {np.mean(sepal_width):.4f}")
print(f"标准差: {np.std(sepal_width):.4f}")

print("\nZ-Score规范化后数据统计:")
print(f"均值: {np.mean(standardized_sepal_width):.4f}")
print(f"标准差: {np.std(standardized_sepal_width):.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(sepal_width, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('原始花萼宽度分布')
plt.xlabel('花萼宽度 (cm)')
plt.ylabel('频次')

plt.subplot(1, 2, 2)
plt.hist(standardized_sepal_width, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('Z-Score规范化后分布')
plt.xlabel('标准化值')
plt.ylabel('频次')

plt.tight_layout()
plt.show()



