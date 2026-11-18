
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and prepare dataset
data = pd.read_csv("data/energydata_complete.csv")

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date").reset_index(drop=True)

# Create forecasting target
data["target_next_10min"] = data["Appliances"].shift(-1)

# Current and historical consumption features
data["Appliances_current"] = data["Appliances"]
data["Appliances_lag_1"] = data["Appliances"].shift(1)
data["Appliances_lag_2"] = data["Appliances"].shift(2)
data["Appliances_lag_3"] = data["Appliances"].shift(3)

data["Appliances_rolling_mean_3"] = (
    data["Appliances"].shift(1).rolling(window=3).mean()
)

# Forecast time features
forecast_date = data["date"] + pd.Timedelta(minutes=10)
data["forecast_hour"] = forecast_date.dt.hour
data["forecast_day"] = forecast_date.dt.day_of_week

# Remove rows with missing values
data = data.dropna().reset_index(drop=True)

# Select features
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

# Five chronological validation folds
tscv = TimeSeriesSplit(n_splits=5)

baseline_mae_scores = []
rf_mae_scores = []

baseline_rmse_scores = []
rf_rmse_scores = []

baseline_r2_scores = []
rf_r2_scores = []

for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Persistence baseline
    baseline_predictions = X_test["Appliances_current"]

    # Random Forest
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
    predictions = model.predict(X_test)

    # Calculate metrics
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    rf_mae = mean_absolute_error(y_test, predictions)

    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
    rf_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    baseline_r2 = r2_score(y_test, baseline_predictions)
    rf_r2 = r2_score(y_test, predictions)

    baseline_mae_scores.append(baseline_mae)
    rf_mae_scores.append(rf_mae)

    baseline_rmse_scores.append(baseline_rmse)
    rf_rmse_scores.append(rf_rmse)

    baseline_r2_scores.append(baseline_r2)
    rf_r2_scores.append(rf_r2)

    print(f"\n--- FOLD {fold} ---")
    print(f"Persistence MAE: {baseline_mae:.2f}")
    print(f"Random Forest MAE: {rf_mae:.2f}")
    print(f"Persistence RMSE: {baseline_rmse:.2f}")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
    print(f"Persistence R2: {baseline_r2:.4f}")
    print(f"Random Forest R2: {rf_r2:.4f}")

# Average results
print("\n--- AVERAGE CROSS-VALIDATION RESULTS ---")

print("\nPERSISTENCE BASELINE")
print("Average MAE:", round(np.mean(baseline_mae_scores), 2))
print("Average RMSE:", round(np.mean(baseline_rmse_scores), 2))
print("Average R2:", round(np.mean(baseline_r2_scores), 4))

print("\nRANDOM FOREST")
print("Average MAE:", round(np.mean(rf_mae_scores), 2))
print("Average RMSE:", round(np.mean(rf_rmse_scores), 2))
print("Average R2:", round(np.mean(rf_r2_scores), 4))
