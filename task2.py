from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

print("前5行数据 data:")
print(iris.data[:5])
print("后5行target:")
print(iris.target[-5:])
print("feature_names:")
print(iris.feature_names)
print("DESCR:")
print(iris.DESCR)



