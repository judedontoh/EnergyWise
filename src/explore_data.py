
import pandas as pd

# Load the energy consumption dataset
data = pd.read_csv("data/energydata_complete.csv")

# Display the first five rows
print("\n--- FIRST FIVE ROWS ---")
print(data.head())

# Display the dataset dimensions
print("\n--- DATASET SHAPE ---")
print(data.shape)

# Display column names
print("\n--- COLUMN NAMES ---")
print(data.columns.tolist())

# Display dataset information
print("\n--- DATASET INFORMATION ---")
data.info()

# Check for missing values
print("\n--- MISSING VALUES ---")
print(data.isnull().sum())

# Statistical summary of numerical variables
print("\n--- STATISTICAL SUMMARY ---")
print(data.describe())

# Average appliance energy consumption
print("\n--- AVERAGE APPLIANCE ENERGY ---")
print(data["Appliances"].mean(), "Wh")

# Highest appliance energy consumption
print("\n--- MAXIMUM APPLIANCE ENERGY ---")
print(data["Appliances"].max(), "Wh")

# Lowest appliance energy consumption
print("\n--- MINIMUM APPLIANCE ENERGY ---")
print(data["Appliances"].min(), "Wh")
