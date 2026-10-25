
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Our dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([20000, 18500, 15000, 15500, 11000, 9500, 8500, 5500])

# Split the data: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train using ONLY training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices for cars the model hasn't seen
y_pred = model.predict(X_test)

# Compare predictions with actual prices
for age, actual, predicted in zip(X_test, y_test, y_pred):
    print("Car age:", age[0])
    print("Actual price:", actual)
    print("Predicted price:", round(predicted, 2))
    print()

# Evaluate the model on test data
mae = mean_absolute_error(y_test, y_pred)
print("Test MAE:", round(mae, 2))
