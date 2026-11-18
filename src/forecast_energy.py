
import pandas as pd

# Load dataset
data = pd.read_csv("data/energydata_complete.csv")

# Convert date column to datetime
data["date"] = pd.to_datetime(data["date"])

# Sort records chronologically
data = data.sort_values("date").reset_index(drop=True)

# Check time intervals
data["time_difference"] = data["date"].diff()

print("--- TIME INTERVAL CHECK ---")
print(data["time_difference"].value_counts().head())

# Create the forecasting target:
# Predict appliance consumption 10 minutes ahead
data["target_next_10min"] = data["Appliances"].shift(-1)

# Keep current consumption as an input feature
data["Appliances_current"] = data["Appliances"]

# Create time features for the forecast timestamp
forecast_date = data["date"] + pd.Timedelta(minutes=10)

data["forecast_hour"] = forecast_date.dt.hour
data["forecast_day"] = forecast_date.dt.day_of_week

# Remove the final row because it has no next observation
data = data.dropna(subset=["target_next_10min"])

print("\n--- FORECASTING DATASET ---")
print("Number of records:", len(data))
print("Forecasting interval: 10 minutes")

print("\nExample forecasting records:")
print(
    data[[
        "date",
        "Appliances_current",
        "target_next_10min"
    ]].head(10).to_string(index=False)
)

