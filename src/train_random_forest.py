
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# Load dataset
data = pd.read_csv("data/energydata_complete.csv")

# Convert date column to datetime
data["date"] = pd.to_datetime(data["date"])

# Sort records chronologically
data = data.sort_values("date").reset_index(drop=True)

# Create time-based features
data["hour"] = data["date"].dt.hour
data["day_of_week"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month
data["is_weekend"] = (data["date"].dt.dayofweek >= 5).astype(int)

# Select input features
features = [
    "lights",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6",
    "RH_7", "RH_8", "RH_9",
    "T_out", "Press_mm_hg", "RH_out",
    "Windspeed", "Visibility", "Tdewpoint",
    "hour", "day_of_week", "month", "is_weekend"
]

# Define input and target variables
X = data[features]
y = data["Appliances"]

# Split data chronologically (80% training, 20% testing)
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Create Random Forest model

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)


# Train the model
print("Training Random Forest model...")
model.fit(X_train, y_train)

# Check performance on training data
train_predictions = model.predict(X_train)

train_r2 = r2_score(y_train, train_predictions)

print("\n--- TRAINING PERFORMANCE ---")
print("Training R-squared:", round(train_r2, 4))

# Generate predictions
predictions = model.predict(X_test)

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n--- ACTUAL VS PREDICTED VALUES ---")
print(comparison.head(10).round(2))

# Calculate prediction bias
prediction_bias = np.mean(predictions - y_test)

print("\n--- PREDICTION BIAS ---")
print("Average prediction bias:", round(prediction_bias, 2), "Wh")

# Evaluate performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Display results
print("\n--- RANDOM FOREST RESULTS ---")
print("Mean Absolute Error (MAE):", round(mae, 2), "Wh")
print("Root Mean Squared Error (RMSE):", round(rmse, 2), "Wh")
print("R-squared (R²):", round(r2, 4))

# Analyse feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- TOP 10 IMPORTANT FEATURES ---")
print(importance.head(10).round(4).to_string(index=False))

# Save trained model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nRandom Forest model trained and saved successfully!")
