from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

chinese_columns = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
df_renamed = df.rename(columns=dict(zip(iris.feature_names, chinese_columns)))

print("\n重命名后的列名:")
print(df_renamed.columns.tolist())



