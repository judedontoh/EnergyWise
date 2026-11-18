
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/energydata_complete.csv")

# Select numerical variables
correlation = data.corr(numeric_only=True)

# Display correlation with appliance energy consumption
print("\n--- CORRELATION WITH APPLIANCE ENERGY ---")
print(
    correlation["Appliances"]
    .sort_values(ascending=False)
)

# Create correlation heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Energy Consumption Correlation Heatmap")
plt.tight_layout()
plt.show()
