
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/energydata_complete.csv")

# Set chart style
sns.set_theme(style="whitegrid")

# Variables to examine
variables = ["lights", "T2", "T6"]

# Create scatterplots
for variable in variables:
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=data,
        x=variable,
        y="Appliances",
        alpha=0.4
    )

    plt.title(f"{variable} vs Appliance Energy Consumption")
    plt.xlabel(variable)
    plt.ylabel("Appliance Energy Consumption (Wh)")
    plt.tight_layout()
    plt.show()
