
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer
)

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

# 3. Create Random Forest
model = RandomForestClassifier(random_state=42)

# 4. Define hyperparameters to test
parameters = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5]
}

# 5. Create GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,
    scoring=make_scorer(recall_score, pos_label=0),
    n_jobs=-1
)

# 6. Find the best hyperparameters
grid_search.fit(X_train, y_train)

# 7. Display results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation recall:", grid_search.best_score_)

# 8. Get the best trained model
best_model = grid_search.best_estimator_

# 9. Predict on test data
predictions = best_model.predict(X_test)

# 10. Evaluate malignant class (0)
print("\n--- Test Results ---")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, pos_label=0))
print("Recall:", recall_score(y_test, predictions, pos_label=0))
print("F1-score:", f1_score(y_test, predictions, pos_label=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
