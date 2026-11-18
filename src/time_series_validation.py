
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

# Load dataset
data = pd.read_csv("data/energydata_complete.csv")

# Prepare date column
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date").reset_index(drop=True)

# Create time-based features
data["hour"] = data["date"].dt.hour
data["day_of_week"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month
data["is_weekend"] = (data["date"].dt.dayofweek >= 5).astype(int)

# Select features
features = [
    "lights",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6",
    "RH_7", "RH_8", "RH_9",
    "T_out", "Press_mm_hg", "RH_out",
    "Windspeed", "Visibility", "Tdewpoint",
    "hour", "day_of_week", "month", "is_weekend"
]

X = data[features]
y = data["Appliances"]

# Define models
models = {
    "Baseline (Mean)": DummyRegressor(strategy="mean"),

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# Create time-series validation
tscv = TimeSeriesSplit(n_splits=5)

# Evaluate each model
for model_name, model in models.items():

    results = []

    print(f"\n--- {model_name} ---")

    for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append([mae, rmse, r2])

        print(
            f"Fold {fold}: "
            f"MAE = {mae:.2f} Wh, "
            f"RMSE = {rmse:.2f} Wh, "
            f"R² = {r2:.4f}"
        )

    results = np.array(results)

    print("\nAverage Performance:")
    print("Mean MAE:", round(results[:, 0].mean(), 2), "Wh")
    print("Mean RMSE:", round(results[:, 1].mean(), 2), "Wh")
    print("Mean R²:", round(results[:, 2].mean(), 4))

