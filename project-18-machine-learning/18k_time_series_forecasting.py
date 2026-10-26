
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Generate 8 weeks of fictional cafe customer data
np.random.seed(42)

days = np.arange(1, 57)
weekdays = (days - 1) % 7

# Monday=0, Tuesday=1, ..., Sunday=6
weekday_effect = np.array([0, 5, 8, 10, 20, 60, 45])

customers = (
    40
    + days * 0.5
    + weekday_effect[weekdays]
    + np.random.randint(-5, 6, size=56)
)

# 2. Create a DataFrame
df = pd.DataFrame({
    "day": days,
    "weekday": weekdays,
    "customers": customers
})

# 3. Convert weekdays into separate 0/1 features
weekday_features = pd.get_dummies(
    df["weekday"],
    prefix="weekday",
    dtype=int
)

# Drop Monday to use it as the reference weekday
weekday_features = weekday_features.drop(columns="weekday_0")

# 4. Prepare features
X_simple = df[["day"]]

X_weekly = pd.concat(
    [df[["day"]], weekday_features],
    axis=1
)

y = df["customers"]

# 5. Train on 6 weeks, test on the final 2 weeks
X_simple_train = X_simple.iloc[:42]
X_simple_test = X_simple.iloc[42:]

X_weekly_train = X_weekly.iloc[:42]
X_weekly_test = X_weekly.iloc[42:]

y_train = y.iloc[:42]
y_test = y.iloc[42:]

# 6. Original model: trend only
simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_train)

simple_predictions = simple_model.predict(X_simple_test)

# 7. Improved model: trend + weekday
weekly_model = LinearRegression()
weekly_model.fit(X_weekly_train, y_train)

weekly_predictions = weekly_model.predict(X_weekly_test)

# 8. Compare model errors
simple_mae = mean_absolute_error(y_test, simple_predictions)
weekly_mae = mean_absolute_error(y_test, weekly_predictions)

print(f"Original Model MAE: {simple_mae:.2f} customers")
print(f"Weekly Model MAE: {weekly_mae:.2f} customers")

# 9. Show predictions for the test days
print("\nPredictions for the final 14 days:")

for day, actual, simple, weekly in zip(
    days[42:],
    y_test,
    simple_predictions,
    weekly_predictions
):
    print(
        f"Day {day}: Actual={actual}, "
        f"Original={simple:.0f}, "
        f"Weekly={weekly:.0f}"
    )

# 10. Visualize the comparison
plt.figure(figsize=(11, 5))

plt.plot(
    days[42:],
    y_test,
    marker="o",
    label="Actual Customers"
)

plt.plot(
    days[42:],
    simple_predictions,
    linestyle="--",
    label="Original Model"
)

plt.plot(
    days[42:],
    weekly_predictions,
    linestyle="--",
    marker="o",
    label="Weekly Model"
)

plt.xlabel("Day")
plt.ylabel("Number of Customers")
plt.title("Cafe Forecasting: Trend vs Weekly Seasonality")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Forecast the next 7 days
future_days = np.arange(57, 64)
future_weekdays = (future_days - 1) % 7

future_df = pd.DataFrame({
    "day": future_days,
    "weekday": future_weekdays
})

future_weekday_features = pd.get_dummies(
    future_df["weekday"],
    prefix="weekday",
    dtype=int
)

future_weekday_features = future_weekday_features.reindex(
    columns=weekday_features.columns,
    fill_value=0
)

future_X = pd.concat(
    [
        future_df[["day"]].reset_index(drop=True),
        future_weekday_features.reset_index(drop=True)
    ],
    axis=1
)

future_predictions = weekly_model.predict(future_X)

weekday_names = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

print("\nNext 7 days forecast:")

for day, weekday, prediction in zip(
    future_days,
    future_weekdays,
    future_predictions
):
    print(
        f"Day {day} ({weekday_names[weekday]}): "
        f"{prediction:.0f} customers"
    )
