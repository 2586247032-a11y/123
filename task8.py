from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model_scaled = LogisticRegression(random_state=42, max_iter=200)
lr_model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = lr_model_scaled.predict(X_test_scaled)

accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
precision_scaled = precision_score(y_test, y_pred_scaled, average='weighted')
recall_scaled = recall_score(y_test, y_pred_scaled, average='weighted')
f1_scaled = f1_score(y_test, y_pred_scaled, average='weighted')

print("使用标准化的模型性能:")
print(f"准确率: {accuracy_scaled:.4f}")
print(f"精确率: {precision_scaled:.4f}")
print(f"召回率: {recall_scaled:.4f}")
print(f"F1 Score: {f1_scaled:.4f}")

cm_scaled = confusion_matrix(y_test, y_pred_scaled)
disp_scaled = ConfusionMatrixDisplay(confusion_matrix=cm_scaled, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp_scaled.plot(ax=ax, cmap='Blues')
plt.title('标准化模型混淆矩阵')
plt.show()

lr_model_original = LogisticRegression(random_state=42, max_iter=200)
lr_model_original.fit(X_train, y_train)
y_pred_original = lr_model_original.predict(X_test)
# sgvs
accuracy_original = accuracy_score(y_test, y_pred_original)
precision_original = precision_score(y_test, y_pred_original, average='weighted')
recall_original = recall_score(y_test, y_pred_original, average='weighted')
f1_original = f1_score(y_test, y_pred_original, average='weighted')

print("\n不使用标准化的模型性能:")
print(f"准确率: {accuracy_original:.4f}")
print(f"精确率: {precision_original:.4f}")
print(f"召回率: {recall_original:.4f}")
print(f"F1 Score: {f1_original:.4f}")

cm_original = confusion_matrix(y_test, y_pred_original)
disp_original = ConfusionMatrixDisplay(confusion_matrix=cm_original, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp_original.plot(ax=ax, cmap='Blues')
plt.title('原始数据模型混淆矩阵')
plt.show()

print("\n性能对比:")
print(f"标准化后准确率提升: {accuracy_scaled - accuracy_original:.4f}")
print(f"标准化后精确率提升: {precision_scaled - precision_original:.4f}")
print(f"标准化后召回率提升: {recall_scaled - recall_original:.4f}")
print(f"标准化后F1 Score提升: {f1_scaled - f1_original:.4f}")



