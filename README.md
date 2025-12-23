# France Grid Stress Prediction

This project develops a reproducible data science pipeline for day-ahead (J+1) forecasting of electricity consumption in France, with the long-term objective of integrating renewable generation forecasts (wind and solar) to analyze stress on the power grid.

The project is positioned at the intersection of energy systems, time-series forecasting, and applied machine learning. It is designed as an academic and exploratory framework, but follows professional engineering standards in terms of structure, reproducibility, and evaluation methodology.

---

## Motivation and context

The French electricity system is undergoing a structural transformation. While it has historically relied on a large and stable nuclear base, the increasing penetration of intermittent renewable energies and the electrification of uses (heating, mobility) introduce new sources of variability on both the demand and supply sides.

France also exhibits a strong thermosensitivity of electricity consumption, mainly due to the widespread use of electric heating. As a result, short-term forecasting errors can quickly translate into several gigawatts of imbalance, with direct implications for system security, market prices, and carbon intensity.

Accurate day-ahead forecasting is therefore critical for:
- balancing supply and demand on the grid,
- participating efficiently in day-ahead electricity markets,
- anticipating periods of grid stress and reduced safety margins.

---

## Project objectives

The main objectives of the project are:

- Build an automated and reproducible pipeline to collect, clean, and align electricity and weather data.
- Construct a high-quality hourly dataset for mainland France combining load, weather, and calendar effects.
- Develop and evaluate day-ahead electricity load forecasting models using time-series aware validation.
- Extend the framework to forecast renewable electricity generation (wind and solar).
- Analyze grid stress through residual load and extreme demand situations.

---

## Data sources

### Electricity data
- Source: RTE (éCO2mix open data platform)
- Frequency: hourly or sub-hourly (aggregated to hourly)
- Coverage: mainland France
- Variables:
  - total electricity consumption,
  - production by generation source (nuclear, wind, solar, hydro, thermal),
  - imports and exports.

### Weather data
- Source: Open-Meteo (aggregating public meteorological models and observations)
- Variables:
  - temperature,
  - wind speed,
  - cloud cover,
  - solar radiation.
- Methodology:
  - weather is collected for a set of representative French agglomerations,
  - variables are aggregated using population-based weights to approximate a national weather signal.

### Calendar data
- Day of week and hour of day
- Public holidays and seasonal effects
- Cyclical encodings for time variables

---

## Project structure

The repository follows a professional data science structure inspired by industry standards.

```text
france-grid-stress-prediction/
├─ configs/        Configuration files (data sources, features, model parameters)
├─ data/
│  ├─ external/    Reference files (agglomerations, weights)
│  ├─ raw/         Raw data (not versioned)
│  ├─ interim/     Intermediate datasets
│  └─ processed/   Final datasets used for modeling
├─ notebooks/      Exploratory analysis and prototyping
├─ src/fgsp/       Python package containing core logic
├─ scripts/        Reproducible command-line scripts
├─ models/         Trained model artifacts
├─ reports/        Figures and tables
└─ tests/          Basic tests (time alignment, leakage checks)
