from pandas.core.common import random_state
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix
)

data = load_breast_cancer()

x = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, pos_label=0))
print("Recall:", recall_score(y_test, predictions, pos_label=0))
print("F1-score:", f1_score(y_test, predictions, pos_label=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))