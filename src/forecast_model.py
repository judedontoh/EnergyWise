
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the forecasting dataset
data = pd.read_csv("data/energydata_complete.csv")

# Prepare dates
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date").reset_index(drop=True)

# Create the 10-minute-ahead target
data["target_next_10min"] = data["Appliances"].shift(-1)

# Keep current consumption as a feature
data["Appliances_current"] = data["Appliances"]

# Create historical consumption features
data["Appliances_lag_1"] = data["Appliances"].shift(1)
data["Appliances_lag_2"] = data["Appliances"].shift(2)
data["Appliances_lag_3"] = data["Appliances"].shift(3)

# Average consumption over the previous 30 minutes
data["Appliances_rolling_mean_3"] = (
    data["Appliances"].shift(1).rolling(window=3).mean()
)

# Create time features
forecast_date = data["date"] + pd.Timedelta(minutes=10)
data["forecast_hour"] = forecast_date.dt.hour
data["forecast_day"] = forecast_date.dt.day_of_week

# Remove final row with no future target
data = data.dropna(
    subset=[
        "target_next_10min",
        "Appliances_lag_1",
        "Appliances_lag_2",
        "Appliances_lag_3",
        "Appliances_rolling_mean_3"
    ]
)

# Select input features
features = [
    "Appliances_current",
    "Appliances_lag_1",
    "Appliances_lag_2",
    "Appliances_lag_3",
    "Appliances_rolling_mean_3",
    "lights",
    "T1", "T2", "T3", "T4", "T5",
    "T6", "T7", "T8", "T9",
    "RH_1", "RH_2", "RH_3", "RH_4", "RH_5",
    "RH_6", "RH_7", "RH_8", "RH_9",
    "T_out", "Press_mm_hg", "RH_out",
    "Windspeed", "Visibility", "Tdewpoint",
    "forecast_hour", "forecast_day"
]

X = data[features]
y = data["target_next_10min"]

# Chronological 80/20 split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Persistence baseline: predict current consumption
baseline_predictions = X_test["Appliances_current"]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

import joblib
joblib.dump(model, "models/forecast_random_forest.pkl")

# Generate predictions
predictions = model.predict(X_test)

# Evaluate both approaches
print("\n--- PERSISTENCE BASELINE ---")
print("MAE:", round(mean_absolute_error(y_test, baseline_predictions), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, baseline_predictions)), 2))
print("R2:", round(r2_score(y_test, baseline_predictions), 4))

print("\n--- RANDOM FOREST FORECAST ---")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
print("R2:", round(r2_score(y_test, predictions), 4))


# Analyse Random Forest prediction errors

actual_values = y_test.to_numpy()
errors = predictions - actual_values

print("\n--- RANDOM FOREST ERROR ANALYSIS ---")

# Average prediction error (positive means overprediction)
print("Mean prediction error (bias):", round(np.mean(errors), 2), "Wh")

# Largest overprediction
max_over = np.argmax(errors)

print("\nLargest overprediction:")
print("Actual:", actual_values[max_over], "Wh")
print("Predicted:", round(predictions[max_over], 2), "Wh")
print("Error:", round(errors[max_over], 2), "Wh")

# Largest underprediction
max_under = np.argmin(errors)

print("\nLargest underprediction:")
print("Actual:", actual_values[max_under], "Wh")
print("Predicted:", round(predictions[max_under], 2), "Wh")
print("Error:", round(errors[max_under], 2), "Wh")



# Compare model performance across consumption levels

high_consumption = actual_values > 300
normal_consumption = actual_values <= 300

print("\n--- CONSUMPTION LEVEL ANALYSIS ---")

# Normal-consumption events
print("\nNormal consumption (300 Wh or below):")
print("Number of events:", np.sum(normal_consumption))

if np.sum(normal_consumption) > 0:
    rf_normal_mae = mean_absolute_error(
        actual_values[normal_consumption],
        predictions[normal_consumption]
    )

    persistence_normal_mae = mean_absolute_error(
        actual_values[normal_consumption],
        baseline_predictions.to_numpy()[normal_consumption]
    )

    print("Random Forest MAE:", round(rf_normal_mae, 2), "Wh")
    print("Persistence MAE:", round(persistence_normal_mae, 2), "Wh")

# High-consumption events
print("\nHigh consumption (above 300 Wh):")
print("Number of events:", np.sum(high_consumption))

if np.sum(high_consumption) > 0:
    rf_high_mae = mean_absolute_error(
        actual_values[high_consumption],
        predictions[high_consumption]
    )

    persistence_high_mae = mean_absolute_error(
        actual_values[high_consumption],
        baseline_predictions.to_numpy()[high_consumption]
    )

    print("Random Forest MAE:", round(rf_high_mae, 2), "Wh")
    print("Persistence MAE:", round(persistence_high_mae, 2), "Wh")


# Compare actual and predicted consumption visually

# Display the first 200 test observations
number_of_points = 200

plt.figure(figsize=(15, 6))

plt.plot(
    y_test.iloc[:number_of_points].values,
    label="Actual Consumption",
    linewidth=2
)

plt.plot(
    baseline_predictions.iloc[:number_of_points].values,
    label="Persistence Forecast",
    alpha=0.8
)

plt.plot(
    predictions[:number_of_points],
    label="Random Forest Forecast",
    alpha=0.8
)

plt.title("10-Minute Appliance Energy Consumption Forecast")
plt.xlabel("Test Observation")
plt.ylabel("Appliance Consumption (Wh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
