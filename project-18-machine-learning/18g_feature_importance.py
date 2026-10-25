
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train Random Forest with all 30 features
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Feature importance
importances = model.feature_importances_

for name, importance in zip(data.feature_names, importances):
    print(name, ":", round(importance, 4))

# 5. Feature importance chart
plt.figure(figsize=(12, 8))
plt.barh(data.feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Breast Cancer - Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# 6. Correlation heatmap
df = pd.DataFrame(X, columns=data.feature_names)
correlation = df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f"
)

plt.title("Breast Cancer - Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 7. Select the 10 most important features
selector = SelectFromModel(
    model,
    max_features=10,
    threshold=-float("inf"),
    prefit=True
)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print("\nSelected features:")
print(data.feature_names[selector.get_support()])

# 8. Train a new model using only 10 features
model_selected = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model_selected.fit(X_train_selected, y_train)

# 9. Make predictions with both models
predictions_all = model.predict(X_test)
predictions_selected = model_selected.predict(X_test_selected)

# 10. Compare results (malignant class = 0)
print("\n--- Model A: All 30 Features ---")
print("Accuracy:", accuracy_score(y_test, predictions_all))
print("Precision:", precision_score(y_test, predictions_all, pos_label=0))
print("Recall:", recall_score(y_test, predictions_all, pos_label=0))
print("F1-score:", f1_score(y_test, predictions_all, pos_label=0))

print("\n--- Model B: Top 10 Features ---")
print("Accuracy:", accuracy_score(y_test, predictions_selected))
print("Precision:", precision_score(y_test, predictions_selected, pos_label=0))
print("Recall:", recall_score(y_test, predictions_selected, pos_label=0))
print("F1-score:", f1_score(y_test, predictions_selected, pos_label=0))
