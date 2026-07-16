# Vehicle Speed Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Predicting vehicle actions (speed categories) based on multiple car parameters using machine learning.

---

## Overview

This project classifies vehicle actions into five categories — **very slow, slow, normal, fast, really fast** — based on features like speed, engine temperature, fuel level, tire pressure, car age, weather conditions, driver experience, and traffic conditions.

The project is organized into two versions:

- **V1.0** — Single model (Decision Tree) with full EDA and visualization
- **V1.1** — Multi-model comparison with cross-validation and hyperparameter tuning

---

## Dataset

| Feature | Description | Type |
|---------|-------------|------|
| speed | Current vehicle speed (km/h) | Numeric |
| engine_temp | Engine temperature | Numeric |
| fuel_level | Remaining fuel percentage | Numeric |
| tire_pressure | Tire pressure (PSI) | Numeric |
| car_age | Age of the vehicle (years) | Numeric |
| weather_condition | Weather (0=sunny, 1=cloudy, 2=rainy) | Categorical |
| driver_experience | Experience level (0=beginner, 1=intermediate, 2=expert) | Categorical |
| traffic_condition | Traffic (0=light, 1=moderate, 2=heavy) | Categorical |
| action | Target: very slow, slow, normal, fast, really fast | Categorical |

**Size:** 500 samples with 8 features

---

## Project Structure

```
VehicleSpeedAnalysis/
├── car_speed_data.csv                    # Dataset
├── car_speed_predictions_V1.0.ipynb      # Single model (Decision Tree) + EDA
├── car_speed_predictions_V1.1.ipynb      # Multi-model comparison + tuning
└── README.md
```

---

## V1.0 — Decision Tree Classifier

A beginner-friendly notebook that covers:
- Data loading and inspection
- Exploratory Data Analysis (distributions, correlations, scatter plots)
- Decision Tree training and evaluation
- Confusion matrix visualization
- Feature importance analysis
- 5-fold cross validation

**Key Result:** Decision Tree achieves ~99% accuracy with speed as the dominant feature.

---

## V1.1 — Multi-Model Comparison

An advanced notebook that includes:
- Comprehensive EDA with violin plots, pair plots, and scatter matrices
- Six models trained and compared:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Stratified 5-fold cross-validation
- GridSearchCV hyperparameter tuning (RF, GB, SVM)
- Confusion matrices for all models
- Feature importance from best model
- Multi-metric comparison (Accuracy, Precision, Recall, F1)

---

## Results Summary

| Model | CV Accuracy | Test Accuracy |
|-------|------------|---------------|
| Decision Tree | ~99.5% | ~100% |
| Gradient Boosting | ~99.5% | ~100% |
| Random Forest | ~92-98% | ~91-100% |
| Logistic Regression | ~81% | ~80% |
| KNN | ~85-90% | ~85-90% |
| SVM | ~68-76% | ~76% |

> Tree-based models dominate because the action classes are defined by axis-aligned speed thresholds.

---

## Installation

```bash
git clone https://github.com/naveed1011/VehicleSpeedAnalysis.git
cd VehicleSpeedAnalysis
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

---

## How to Run

1. Open either notebook in Jupyter
2. Run all cells from top to bottom
3. All visualizations and results will be generated automatically

---

## Key Visualizations

- Feature distribution histograms by action class
- Correlation heatmap
- Speed violin and box plots
- Scatter plots (speed vs engine temp, fuel level, etc.)
- Pair plot for all feature relationships
- Confusion matrices for all models
- Cross-validation score bar and box plots
- Feature importance bar charts
- Model performance comparison chart

---

## Author

**Naveed Ahmad**

---

## License

This project is open source and available under the MIT License.
