# 🏠 California House Price Prediction — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete, production-style Machine Learning project that predicts **median housing prices** across California districts using the California Census dataset. This project follows the full ML workflow — from framing the business problem to fine-tuning the final model and evaluating it on a held-out test set.

---

## 📌 Project Objective

> Build a regression model to predict the **median house value** of a California district given features like location, income, housing age, and proximity to the ocean — accurate enough to inform real estate investment decisions.

The model feeds into a larger decision system that determines whether a given area is **worth investing in or not**. The existing manual approach was often 10%+ off from actual prices — this ML model aims to significantly reduce that error.

---

## 📁 Repository Structure

```
california-house-price-prediction/
│
├── notebook/
│   └── house_price_prediction.ipynb     # Full Jupyter notebook (main project)
│
├── data/
│   └── housing.csv                      # California census housing dataset
│
├── models/
│   └── lin_reg.pkl                      # Saved Linear Regression model (joblib)
│
├── screenshots/
│   ├── data_histogram.png               # Feature distribution histograms
│   ├── geo_scatter.png                  # Geographic housing price scatter plot
│   ├── correlation_matrix.png           # Feature correlation heatmap
│   └── scatter_matrix.png              # Scatter matrix of key attributes
│
├── docs/
│   └── ml_checklist.md                 # ML project checklist reference
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | California Census Data |
| Records | 20,640 districts |
| Features | 10 (9 numerical + 1 categorical) |
| Target | `median_house_value` |
| Missing Values | `total_bedrooms` — 207 nulls (handled via median imputation) |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `longitude` | float | Geographic longitude |
| `latitude` | float | Geographic latitude |
| `housing_median_age` | float | Median age of houses in district |
| `total_rooms` | float | Total rooms in district |
| `total_bedrooms` | float | Total bedrooms (has nulls) |
| `population` | float | District population |
| `households` | float | Number of households |
| `median_income` | float | Median income (scaled, in tens of thousands) |
| `ocean_proximity` | object | Categorical — NEAR BAY, INLAND, etc. |
| `median_house_value` | float | **Target variable** |

---

## 🔄 ML Pipeline — Step by Step

### 1. Frame the Problem
- **Type:** Supervised Learning → Regression (predicting a continuous value)
- **Performance Measure:** RMSE (Root Mean Squared Error) — penalises large errors more heavily
- **Business Context:** Part of a downstream investment decision system

### 2. Data Exploration & Visualisation
- Plotted histograms for all 9 numerical features to understand distributions
- Created geographic scatter plots (longitude vs latitude) coloured by `median_house_value` — confirmed coastal areas are more expensive
- Built correlation matrix: **`median_income` (0.687)** is by far the strongest predictor of house value
- Created scatter matrix for top correlated features

### 3. Feature Engineering
Three new ratio features were derived to improve signal:

| New Feature | Formula | Correlation with Target |
|-------------|---------|------------------------|
| `rooms_per_hh` | `total_rooms / households` | +0.146 |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | **−0.260** (strong negative) |
| `population_per_hh` | `population / households` | −0.022 |

> `bedrooms_per_room` had a **stronger correlation than total_bedrooms or total_rooms alone** — a key insight from feature engineering.

### 4. Stratified Train/Test Split
Used `StratifiedShuffleSplit` on an `Income_cat` column (median income bucketed into 5 categories) to ensure income distribution is preserved in both train and test sets — avoiding sampling bias.

| Split | Size |
|-------|------|
| Training set | 16,512 rows (80%) |
| Test set | 4,128 rows (20%) |

### 5. Data Preparation Pipeline

Built a full `sklearn` `Pipeline` + `ColumnTransformer`:

```
Numerical Features:
  → SimpleImputer (strategy='median')   # Fill 207 missing values
  → FunctionTransformer                 # Add engineered features
  → StandardScaler                      # Normalise scale

Categorical Feature (ocean_proximity):
  → OneHotEncoder                       # 5 categories → 5 binary columns
```

### 6. Model Selection & Evaluation

Three models were trained and compared using **10-fold cross-validation RMSE**:

| Model | Train RMSE | CV RMSE (mean) | CV Std | Notes |
|-------|-----------|----------------|--------|-------|
| Linear Regression | 68,627 | 69,104 | ±2,880 | Underfitting |
| Decision Tree | **0.0** | 71,629 | ±2,914 | Severe overfitting |
| Random Forest (n=10) | 22,413 | 52,792 | ±2,262 | Best so far |

> Decision Tree's 0.0 training RMSE was a red flag for overfitting — cross-validation confirmed this immediately.

### 7. Hyperparameter Tuning — GridSearchCV

Tuned `RandomForestRegressor` over 18 parameter combinations using 5-fold CV:

```python
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
```

**Best Parameters:** `max_features=8, n_estimators=30`
**Best CV RMSE:** ~49,899

### 8. Feature Importance (from best Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `median_income` | **37.90%** |
| 2 | `<1H OCEAN` (ocean proximity) | 16.57% |
| 3 | `pop_per_hhold` | 10.70% |
| 4 | `longitude` | 6.97% |
| 5 | `latitude` | 6.04% |
| 6 | `rooms_per_hhold` | 5.48% |

> Median income alone accounts for nearly **38% of the model's predictive power**.

### 9. Final Test Set Evaluation

| Metric | Value |
|--------|-------|
| Final RMSE on test set | **47,873** |
| 95% Confidence Interval | [46,874 — 48,852] |

The confidence interval gives a statistically robust range for real-world performance expectations.

---

## 📐 Key Metrics Summary

| Metric | Formula | Best Model Result |
|--------|---------|------------------|
| RMSE | √(mean(y_pred − y_true)²) | 47,873 |
| MAE | mean(|y_pred − y_true|) | 49,438 (Linear) |
| CV Score | 10-fold mean RMSE | 49,899 (Random Forest tuned) |

---

## 🔍 Key Learnings & Insights

- **Median income is king.** With 37.9% feature importance, it's the single strongest signal. Location features (lat/lon) matter, but income dominates.
- **0.0 error = red flag.** Decision Tree's perfect training score was immediately suspicious — cross-validation revealed it was the *worst* of the three models.
- **Engineered features outperformed raw ones.** `bedrooms_per_room` (−0.26 correlation) was more informative than `total_bedrooms` (0.048) alone.
- **Stratified splitting matters.** Without income-based stratification, the test set could have over/underrepresented key income groups, making evaluation misleading.
- **Confidence intervals > single numbers.** Reporting the 95% CI [46,874–48,852] is more honest and actionable than just stating RMSE = 47,873.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisations |
| `seaborn` | Statistical plots |
| `scikit-learn` | ML pipeline, models, evaluation |
| `scipy` | Confidence interval calculation |
| `joblib` | Model serialisation |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/california-house-price-prediction.git
cd california-house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebook/house_price_prediction.ipynb
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
joblib
jupyter
```

---

## 📋 ML Project Checklist Followed

- [x] Frame the problem and business objective
- [x] Get and explore the data
- [x] Visualise the data to gain insights
- [x] Prepare data (imputation, encoding, scaling, feature engineering)
- [x] Train and compare multiple models
- [x] Fine-tune the best model using GridSearchCV
- [x] Evaluate on test set with confidence interval
- [x] Save model with joblib

---

## 📌 Notes

- The notebook file (`housing.csv`) path references a local directory — update the `read_csv` path when running locally
- The saved model (`lin_reg.pkl`) uses the Linear Regression model; replace with the tuned Random Forest for production use
- This project is based on the California Housing dataset from Aurélien Géron's *Hands-On Machine Learning* (Chapter 2)
