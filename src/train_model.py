
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
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

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Display results
print("\n--- LINEAR REGRESSION RESULTS ---")
print("Mean Absolute Error (MAE):", round(mae, 2), "Wh")
print("Root Mean Squared Error (RMSE):", round(rmse, 2), "Wh")
print("R-squared (R²):", round(r2, 4))

print("\nLinear Regression model trained successfully!")
