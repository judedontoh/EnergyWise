# ⚡ EnergyWise – Appliance Energy Consumption Forecasting

## 📌 Project Overview

EnergyWise is a beginner-friendly data science and machine learning project that analyses household appliance energy consumption and forecasts energy usage **10 minutes ahead**.

The project uses historical energy readings and environmental sensor measurements to explore consumption patterns, evaluate forecasting models, and present insights through an interactive Streamlit dashboard.

## 🎯 Project Objectives

- Explore and analyse household appliance energy consumption.
- Identify patterns and relationships in energy usage.
- Prepare time-series data for forecasting.
- Develop and evaluate machine learning models.
- Compare machine learning predictions against a persistence baseline.
- Build an interactive dashboard for visualising results.

## 📊 Dataset

The project uses the **Appliances Energy Prediction Dataset** from the UCI Machine Learning Repository.

- Records: 19,735 observations.
- Measurement interval: 10 minutes.
- Target variable: `Appliances` (Wh).
- Additional variables: indoor temperature, humidity, outdoor weather conditions, lighting and time-related features.

**Dataset citation:**

Candanedo, L. M. (2017). Appliances Energy Prediction. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 🤖 Forecasting Approach

EnergyWise forecasts appliance energy consumption 10 minutes into the future.

Two approaches are evaluated:

1. **Persistence Baseline:** Assumes the next reading will be similar to the current reading.
2. **Random Forest Regressor:** Uses historical energy readings, lag features and environmental measurements.

### Model Performance

Results on the chronological holdout test set:

| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Persistence Baseline | 26.65 | 66.44 | 0.4674 |
| Random Forest | 40.75 | 72.45 | 0.3667 |

The persistence baseline performed better on both MAE and RMSE in this test. This demonstrates the importance of comparing machine learning models against simple baseline methods.

## 📁 Project Structure

```text
EnergyWise/
│
├── data/
│   └── energydata_complete.csv
│
├── models/
│   └── forecast_random_forest.pkl
│
├── notebooks/
│
├── src/
│   ├── explore_data.py
│   ├── visualise_data.py
│   ├── correlation_analysis.py
│   ├── relationship_analysis.py
│   ├── prepare_data.py
│   ├── forecast_energy.py
│   ├── forecast_model.py
│   └── forecast_validation.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd EnergyWise
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at `http://localhost:8501`.

## 📈 Dashboard Features

- Energy consumption overview and KPI cards.
- Dataset preview and statistical analysis.
- Appliance consumption distribution.
- Average hourly energy consumption.
- Forecasting model performance comparison.
- Interactive 10-minute-ahead forecast demonstration.
- Downloadable CSV forecast report.

## ⚠️ Limitations

- The dataset represents a single experimental household.
- Measurements cover approximately 4.5 months.
- Results may not generalise to other homes or appliances.
- The forecasting models have limitations when predicting sudden consumption spikes.
- The dashboard demonstrates historical forecasting rather than live smart-meter monitoring.

## 🔮 Future Improvements

- Improve forecasting accuracy.
- Explore additional time-series forecasting models.
- Develop a more robust forecasting evaluation process.
- Integrate live energy monitoring.
- Expand the analysis to datasets from multiple households.

## 👩‍💻 Author


**Jude**

Creator and developer of EnergyWise – an appliance energy consumption forecasting and analytics project built using Python, data science and machine learning.

---

*EnergyWise – Making household energy consumption easier to understand through data and machine learning.*
