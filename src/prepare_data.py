
import pandas as pd

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

# Define input (X) and target (y)
X = data[features]
y = data["Appliances"]

# Split data chronologically: first 80% for training
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Display results
print("\n--- DATA SPLIT RESULTS ---")
print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining period:")
print(data["date"].iloc[0], "to", data["date"].iloc[split_index - 1])

print("\nTesting period:")
print(data["date"].iloc[split_index], "to", data["date"].iloc[-1])

print("\nNumber of input features:", len(features))
print("\nData splitting completed successfully!")


# Compare energy consumption across training and testing periods
split_index = int(len(data) * 0.8)

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

print("\n--- ENERGY CONSUMPTION COMPARISON ---")

print("Training average:",
      round(train_data["Appliances"].mean(), 2), "Wh")

print("Testing average:",
      round(test_data["Appliances"].mean(), 2), "Wh")

print("Training standard deviation:",
      round(train_data["Appliances"].std(), 2), "Wh")

print("Testing standard deviation:",
      round(test_data["Appliances"].std(), 2), "Wh")

# Compare energy consumption ranges
print("\n--- ENERGY CONSUMPTION RANGES ---")

print("Training minimum:",
      train_data["Appliances"].min(), "Wh")

print("Training maximum:",
      train_data["Appliances"].max(), "Wh")

print("Testing minimum:",
      test_data["Appliances"].min(), "Wh")

print("Testing maximum:",
      test_data["Appliances"].max(), "Wh")

# Analyse energy consumption distribution
print("\n--- ENERGY CONSUMPTION DISTRIBUTION ---")

print("Percentage below 100 Wh:",
      round((data["Appliances"] < 100).mean() * 100, 2), "%")

print("Percentage between 100 and 300 Wh:",
      round(((data["Appliances"] >= 100) &
             (data["Appliances"] <= 300)).mean() * 100, 2), "%")

print("Percentage above 300 Wh:",
      round((data["Appliances"] > 300).mean() * 100, 2), "%")

# Compare high-consumption events across periods
print("\n--- HIGH-CONSUMPTION COMPARISON ---")

train_high = (train_data["Appliances"] > 300).mean() * 100
test_high = (test_data["Appliances"] > 300).mean() * 100

print("Training observations above 300 Wh:",
      round(train_high, 2), "%")

print("Testing observations above 300 Wh:",
      round(test_high, 2), "%")

# Compare average consumption by hour
print("\n--- HOURLY ENERGY CONSUMPTION ---")

hourly_consumption = data.groupby("hour")["Appliances"].mean()

print(hourly_consumption.round(2).to_string())

# Compare hourly consumption in training and testing periods
print("\n--- TRAINING VS TESTING HOURLY PATTERNS ---")

train_hourly = train_data.groupby("hour")["Appliances"].mean()
test_hourly = test_data.groupby("hour")["Appliances"].mean()

hourly_comparison = pd.DataFrame({
    "Training": train_hourly,
    "Testing": test_hourly
})

print(hourly_comparison.round(2).to_string())

# Visualise training vs testing hourly consumption
import matplotlib.pyplot as plt

hourly_comparison.plot(
    kind="line",
    figsize=(12, 6),
    marker="o"
)

plt.title("Hourly Energy Consumption: Training vs Testing")
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption (Wh)")
plt.xticks(range(24))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

