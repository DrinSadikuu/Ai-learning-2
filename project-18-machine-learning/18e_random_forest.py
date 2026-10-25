
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    "temperature": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                    42, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98],
    "vibration": [1, 2, 1, 3, 2, 4, 3, 5, 6, 7, 8, 9,
                  2, 1, 3, 2, 4, 3, 5, 6, 7, 8, 9, 10],
    "problem": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})


X = data[["temperature", "vibration"]]
y = data["problem"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


print("Actual:", y_test.to_list())
print("Predicted:", predictions.tolist())
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


new_machine = pd.DataFrame({
    "temperature": [82],
    "vibration": [6]
})

result = model.predict(new_machine)

print("New machine:", "Problem" if result[0] == 1 else "Normal")
