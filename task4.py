import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def detect_outliers_3sigma(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    lower_bound = mean_val - 3 * std_val
    upper_bound = mean_val + 3 * std_val
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    return data[outliers_mask], np.where(outliers_mask)[0]

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
chinese_columns = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
df_renamed = df.rename(columns=dict(zip(iris.feature_names, chinese_columns)))

sepal_width = df_renamed['花萼宽度']
outliers, outlier_indices = detect_outliers_3sigma(sepal_width)

print(f"异常值数量: {len(outliers)}")
if len(outliers) > 0:
    print(f"异常值: {outliers.values}")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(sepal_width, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax2 = ax1.twinx()
kde = gaussian_kde(sepal_width)
x_range = np.linspace(sepal_width.min(), sepal_width.max(), 200)
density = kde(x_range)
ax2.plot(x_range, density, color='red', linewidth=2)
ax1.set_xlabel('花萼宽度 (cm)')
ax1.set_ylabel('频率密度', color='blue')
ax2.set_ylabel('密度', color='red')
plt.title('花萼宽度的直方图与核密度图')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



