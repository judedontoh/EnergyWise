
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EnergyWise | Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DASHBOARD STYLING
# --------------------------------------------------


# --------------------------------------------------
# MODERN ENERGYWISE DASHBOARD STYLING
# --------------------------------------------------

st.markdown("""
<style>

/* ===== GLOBAL THEME ===== */

.stApp {
    background: #F4F7F5;
    color: #24372D;
    font-family: "Inter", "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* ===== TYPOGRAPHY ===== */

h1 {
    color: #12372A !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px;
}

h2 {
    color: #164B36 !important;
    font-size: 1.8rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.5px;
    margin-top: 1.8rem !important;
}

h3 {
    color: #285943 !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}

p {
    color: #64756B;
    line-height: 1.7;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {
    background: #102F24;
    border-right: none;
}

[data-testid="stSidebar"] * {
    color: #EAF3ED !important;
}

[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-size: 1.65rem !important;
    letter-spacing: -0.5px;
}

[data-testid="stSidebar"] a {
    display: block;
    padding: 10px 14px;
    margin: 5px 0;
    border-radius: 9px;
    color: #EAF3ED !important;
    text-decoration: none !important;
    transition: background 0.2s ease;
}

[data-testid="stSidebar"] a:hover {
    background: #24543E;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #315444;
}

/* ===== SECTION SPACING ===== */

[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

/* ===== METRIC CARDS ===== */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E3EBE5;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 4px 14px rgba(20, 55, 35, 0.045);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(20, 55, 35, 0.09);
}

[data-testid="stMetricLabel"] {
    color: #718278 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #16834A !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}

/* ===== DATA TABLES ===== */

[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #E3EBE5;
    border-radius: 14px;
    overflow: hidden;
}

/* ===== CHART CONTAINERS ===== */

[data-testid="stImage"] {
    background: #FFFFFF;
    border: 1px solid #E3EBE5;
    border-radius: 16px;
    padding: 12px;
}

/* ===== SLIDERS ===== */

.stSlider [data-baseweb="slider"] {
    color: #16834A;
}

.stSlider [role="slider"] {
    background: #16834A;
    border-color: #16834A;
}

/* ===== BUTTONS ===== */

.stButton > button {
    background: #16834A;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.3rem;
    font-weight: 650;
    transition: background 0.2s ease, transform 0.2s ease;
}

.stButton > button:hover {
    background: #116B3C;
    color: #FFFFFF;
    transform: translateY(-2px);
}

/* ===== ALERTS AND INFORMATION BOXES ===== */

[data-testid="stAlert"] {
    border-radius: 13px;
    border: 1px solid #DCE9DF;
}

/* ===== DIVIDERS ===== */

hr {
    border: none;
    border-top: 1px solid #DFE8E1;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* ===== LINKS ===== */

a {
    color: #16834A;
}

/* ===== RESPONSIVE LAYOUT ===== */

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 2rem !important;
    }

    h2 {
        font-size: 1.5rem !important;
    }
}

</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("⚡ EnergyWise")
st.sidebar.caption("Energy Intelligence Dashboard")

st.sidebar.markdown("---")
st.sidebar.markdown("### NAVIGATION")

st.sidebar.markdown("[🏠 Overview](#overview)")
st.sidebar.markdown("[📊 Dataset Analysis](#dataset-analysis)")
st.sidebar.markdown("[🤖 Model Performance](#model-performance)")
st.sidebar.markdown("[🔮 Energy Forecast](#energy-forecast)")
st.sidebar.markdown("[ℹ️ About Project](#about-project)")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python & Machine Learning")


# --------------------------------------------------
# LOAD DATASET AND TRAINED MODEL
# --------------------------------------------------

data = pd.read_csv("data/energydata_complete.csv")

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date").reset_index(drop=True)

model = joblib.load("models/forecast_random_forest.pkl")


# --------------------------------------------------
# 1. OVERVIEW
# --------------------------------------------------

st.header("🏠 Overview", anchor="overview")

st.title("⚡ EnergyWise")
st.subheader("Appliance Energy Consumption Prediction & Analytics")


# Dashboard KPI summary

st.markdown("### Your Energy at a Glance")

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="Average Consumption",
        value=f"{data['Appliances'].mean():.1f} Wh",
        help="Average appliance energy consumption across all observations."
    )

with kpi2:
    st.metric(
        label="Peak Consumption",
        value=f"{data['Appliances'].max():.0f} Wh",
        help="Highest recorded appliance energy consumption."
    )

with kpi3:
    st.metric(
        label="Total Observations",
        value=f"{len(data):,}",
        help="Total number of recorded observations in the dataset."
    )

st.markdown("---")

st.markdown("""
Predict household appliance energy consumption **10 minutes ahead**
using historical energy readings, environmental sensor data,
and machine learning.
""")

st.markdown("---")

st.markdown("""
Welcome to **EnergyWise**!

This dashboard explores household appliance energy consumption
and demonstrates how machine learning can forecast energy usage
10 minutes ahead.

Use the sections below to explore the dataset, understand
consumption patterns, and evaluate forecasting performance.
""")


# --------------------------------------------------
# 2. DATASET ANALYSIS
# --------------------------------------------------

st.header("📊 Dataset Analysis", anchor="dataset-analysis")

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(data):,}")

with col2:
    st.metric("Number of Features", len(data.columns))

with col3:
    st.metric(
        "Average Appliance Consumption",
        f"{data['Appliances'].mean():.2f} Wh"
    )


# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(data.head(10), width="stretch")



# Energy consumption distribution
st.subheader("Distribution of Appliance Energy Consumption")

fig, ax = plt.subplots(figsize=(11, 5))

fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

sns.histplot(
    data=data,
    x="Appliances",
    bins=35,
    kde=True,
    color="#16834A",
    alpha=0.75,
    edgecolor="white",
    linewidth=0.5,
    ax=ax
)

# Chart styling
ax.set_title(
    "Household Appliance Energy Consumption",
    fontsize=16,
    fontweight="bold",
    color="#12372A",
    pad=20
)

ax.set_xlabel(
    "Energy Consumption (Wh)",
    fontsize=11,
    color="#53665B",
    labelpad=12
)

ax.set_ylabel(
    "Number of Observations",
    fontsize=11,
    color="#53665B",
    labelpad=12
)

ax.grid(axis="y", linestyle="--", alpha=0.2)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#E3EBE5")
ax.spines["bottom"].set_color("#E3EBE5")

ax.tick_params(colors="#64756B", labelsize=10)

fig.tight_layout()

st.pyplot(fig, width="stretch")
plt.close(fig)




# Hourly energy consumption
st.subheader("Average Energy Consumption by Hour")

data["hour"] = data["date"].dt.hour

hourly_consumption = data.groupby("hour")["Appliances"].mean()

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    hourly_consumption.index,
    hourly_consumption.values,
    color="green",
    marker="o",
    linewidth=2.5,
    markersize=5
)

ax.set_title("Average Appliance Energy Consumption by Hour")
ax.set_xlabel("Hour of the Day")
ax.set_ylabel("Average Consumption (Wh)")

ax.set_xticks(range(24))
ax.grid(axis="y", alpha=0.3)

ax.set_facecolor("#FFFFFF")
fig.patch.set_facecolor("#FFFFFF")

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 3. MODEL PERFORMANCE
# --------------------------------------------------

st.header("🤖 Model Performance", anchor="model-performance")

st.subheader("Forecasting Model Performance")

st.markdown("""
The models predict appliance energy consumption 10 minutes ahead.

We compare the Random Forest model against a simple persistence
baseline, which assumes the next reading will equal the current reading.
""")

performance_data = pd.DataFrame({
    "Model": ["Persistence Baseline", "Random Forest"],
    "MAE (Wh)": [26.65, 40.75],
    "RMSE (Wh)": [66.44, 72.45],
    "R² Score": [0.4674, 0.3667]
})

st.dataframe(performance_data, width="stretch")


# Model performance comparison chart

st.subheader("Model Performance Comparison")

fig, ax = plt.subplots(figsize=(10, 5))

performance_plot = performance_data.set_index("Model")[
    ["MAE (Wh)", "RMSE (Wh)"]
]

performance_plot.plot(
    kind="bar",
    ax=ax,
    color=["#16834A", "#A8C9B5"],
    width=0.65
)

ax.set_title(
    "Forecasting Error: Persistence vs Random Forest",
    fontsize=15,
    fontweight="bold"
)

ax.set_ylabel("Prediction Error (Wh)")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

ax.legend(title="Metric")
fig.tight_layout()

st.pyplot(fig, width="stretch")
plt.close(fig)


# Interpretation of model performance

st.markdown("### What Do These Results Mean?")

st.info("""
**The Persistence Baseline performed better than the Random Forest model
on the final test dataset.**

- **Persistence Baseline:** MAE of 26.65 Wh. It predicts that consumption
  in the next 10 minutes will be similar to current consumption.
- **Random Forest:** MAE of 40.75 Wh. It uses historical readings and
  environmental sensor data to make its forecast.

**Key takeaway:** A simple baseline can outperform a more complex
machine learning model. This shows why comparing models against a
baseline is important.
""")

st.caption(
    "Results from the chronological 80/20 test split. "
    "Lower MAE and RMSE are better; higher R² is better."
)

st.info(
    "The persistence baseline achieved lower MAE and RMSE on this test split. "
    "The Random Forest model was not better on every evaluation measure."
)


# --------------------------------------------------
# 4. ENERGY FORECAST
# --------------------------------------------------

st.header("🔮 Energy Forecast", anchor="energy-forecast")

st.subheader("Try a 10-Minute Forecast")

st.markdown("""
Select a historical observation from the dataset.

EnergyWise will use its sensor readings and recent consumption
history to forecast appliance consumption 10 minutes ahead.
""")


# Prepare forecasting features
forecast_data = data.copy()

forecast_data["target_next_10min"] = (
    forecast_data["Appliances"].shift(-1)
)

forecast_data["Appliances_current"] = forecast_data["Appliances"]

forecast_data["Appliances_lag_1"] = (
    forecast_data["Appliances"].shift(1)
)

forecast_data["Appliances_lag_2"] = (
    forecast_data["Appliances"].shift(2)
)

forecast_data["Appliances_lag_3"] = (
    forecast_data["Appliances"].shift(3)
)

forecast_data["Appliances_rolling_mean_3"] = (
    forecast_data["Appliances"].shift(1).rolling(window=3).mean()
)

forecast_data["forecast_hour"] = (
    (forecast_data["date"] + pd.Timedelta(minutes=10)).dt.hour
)

forecast_data["forecast_day"] = (
    (forecast_data["date"] + pd.Timedelta(minutes=10)).dt.day_of_week
)


# Remove rows without required historical features or target
forecast_data = forecast_data.dropna().reset_index(drop=True)


# Model input features
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


# Choose a historical observation
selected_index = st.slider(
    "Select historical observation",
    min_value=0,
    max_value=len(forecast_data) - 1,
    value=min(100, len(forecast_data) - 1)
)

selected_row = forecast_data.iloc[selected_index]

forecast_time = selected_row["date"] + pd.Timedelta(minutes=10)

st.write(
    "Selected observation:",
    selected_row["date"].strftime("%Y-%m-%d %H:%M")
)

st.write(
    "Forecast target time:",
    forecast_time.strftime("%Y-%m-%d %H:%M")
)


# Generate forecast
input_data = selected_row[features].to_frame().T

prediction = model.predict(input_data)[0]


# Compare forecast with actual consumption
comparison = pd.DataFrame({
    "Reading": [
        "Current Consumption",
        "Random Forest Prediction",
        "Actual Next Reading"
    ],
    "Energy (Wh)": [
        selected_row["Appliances_current"],
        prediction,
        selected_row["target_next_10min"]
    ]
})



st.subheader("Forecast vs Actual Consumption")

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(
    comparison["Reading"],
    comparison["Energy (Wh)"],
    color=["#A8C9B5", "#16834A", "#E49B35"],
    width=0.6
)

ax.set_title(
    "10-Minute Energy Forecast Comparison",
    fontsize=15
)

ax.set_ylabel("Energy Consumption (Wh)")
ax.set_xlabel("")

ax.tick_params(axis="x", rotation=10)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

# Display values above each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# Download forecast report

st.subheader("Download Forecast Report")

csv_report = comparison.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Forecast Report",
    data=csv_report,
    file_name="energywise_forecast_report.csv",
    mime="text/csv"
)


# Display forecast results
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Consumption",
        f"{selected_row['Appliances_current']:.2f} Wh"
    )

with col2:
    st.metric(
        "Predicted Consumption",
        f"{prediction:.2f} Wh"
    )

with col3:
    st.metric(
        "Actual Next Reading",
        f"{selected_row['target_next_10min']:.2f} Wh"
    )

st.caption(
    "This is a historical forecasting demonstration using "
    "real dataset observations, not a live household prediction."
)


# --------------------------------------------------
# 5. ABOUT THE PROJECT
# --------------------------------------------------

st.header("ℹ️ About Project", anchor="about-project")

st.subheader("Project Limitations")

st.markdown("""
### Dataset limitations

- The dataset contains approximately 4.5 months of observations.
- It represents one experimental household, not all households.
- Results may differ in homes with different appliances and usage habits.

### Model limitations

- The forecasting model predicts consumption only 10 minutes ahead.
- Random Forest does not always outperform the persistence baseline.
- High-consumption events are more difficult to predict accurately.

### Future improvements

- Test additional machine learning algorithms.
- Improve prediction of sudden consumption spikes.
- Evaluate performance on data from other households.
- Explore longer-term energy forecasting.
""")


st.subheader("💡 Key Project Insights")

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.markdown("### 🎯 Short-term forecasting")
    st.write(
        "Recent appliance consumption readings provide "
        "useful information for predicting the next 10 minutes."
    )

with insight_col2:
    st.markdown("### 📈 Model comparison")
    st.write(
        "The persistence baseline achieves lower average "
        "absolute error than Random Forest on the test set."
    )

with insight_col3:
    st.markdown("### ⚡ High consumption")
    st.write(
        "Random Forest performs better than persistence "
        "on the high-consumption test events, although "
        "these events are relatively uncommon."
    )


# --------------------------------------------------
# DASHBOARD FOOTER
# --------------------------------------------------

st.markdown("---")

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.markdown("**⚡ EnergyWise**")
    st.caption("Appliance Energy Consumption Prediction & Analytics")

with footer_col2:
    st.markdown("**Built with**")
    st.caption("Python · Pandas · Scikit-learn · Streamlit")

st.caption(
    "Dataset: Candanedo (2017), UCI Appliances Energy Prediction. "
    "DOI: 10.24432/C5VC8G"
)
