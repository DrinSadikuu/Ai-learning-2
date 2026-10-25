
import numpy as np
from sklearn.linear_model import LinearRegression

# Features: [car age, mileage]
X = np.array([
    [1, 10000],
    [2, 25000],
    [3, 40000],
    [4, 55000],
    [5, 70000],
    [6, 90000],
    [7, 110000],
    [8, 130000]
])

# Actual car prices
y = np.array([
    22000, 20000, 17500, 15500,
    13000, 10500, 8000, 5500
])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict a car that is 4 years old with 60000 km
prediction = model.predict([[4, 60000]])

print("Predicted price:", round(prediction[0], 2))
