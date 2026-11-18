
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data/energydata_complete.csv")

# Convert date column into datetime format
data["date"] = pd.to_datetime(data["date"])

# Set chart style
sns.set_theme(style="whitegrid")

# 1. Distribution of appliance energy consumption
plt.figure(figsize=(10, 5))
sns.histplot(data["Appliances"], bins=30, kde=True)
plt.title("Distribution of Appliance Energy Consumption")
plt.xlabel("Energy Consumption (Wh)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Boxplot of appliance energy consumption
plt.figure(figsize=(10, 5))
sns.boxplot(x=data["Appliances"])
plt.title("Appliance Energy Consumption - Boxplot")
plt.xlabel("Energy Consumption (Wh)")
plt.tight_layout()
plt.show()

# 3. Energy consumption over time
plt.figure(figsize=(12, 5))
plt.plot(data["date"], data["Appliances"], linewidth=0.7)
plt.title("Appliance Energy Consumption Over Time")
plt.xlabel("Date")
plt.ylabel("Energy Consumption (Wh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
