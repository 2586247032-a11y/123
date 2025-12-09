import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    return data[outliers_mask]

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
chinese_columns = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
df_renamed = df.rename(columns=dict(zip(iris.feature_names, chinese_columns)))

sepal_width = df_renamed['花萼宽度']
outliers = detect_outliers_iqr(sepal_width)

print(f"异常值数量: {len(outliers)}")
if len(outliers) > 0:
    print(f"异常值: {outliers.values}")

plt.figure(figsize=(8, 6))
plt.boxplot(sepal_width)
plt.title('花萼宽度箱型图')
plt.ylabel('花萼宽度 (cm)')
plt.grid(True, alpha=0.3)
plt.show()



