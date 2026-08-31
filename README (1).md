# ChurnGuard — Telco Customer Churn Prediction

I built this project to predict customer churn for a telecom company using the IBM Telco Customer Churn dataset. It covers the full workflow I'd use on a real churn problem: cleaning the raw data, exploring what actually separates churners from retained customers, engineering a few extra features, training and comparing several models, tuning the best one, and checking what it's actually picking up on with SHAP.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Churn Drivers](#key-churn-drivers)
- [Tech Stack](#tech-stack)
- [Business Recommendations](#business-recommendations)
- [License](#license)

## Dataset

I used the **IBM Telco Customer Churn** dataset: 7,043 customers, 21 raw features (demographics, account details, subscribed services), with a binary `Churn` target. About 26.5% of customers in the dataset churned.

| | |
|---|---|
| Rows | 7,043 |
| Features | 21 (20 predictors + target) |
| Target | `Churn` (Yes / No) |

Place the dataset CSV in the project root (or update the path in the notebook) before running it.

## Project Structure

```
.
├── Customer_Churn_Prediction.ipynb   # Main notebook: EDA, modeling, evaluation
├── churn_model_pipeline.pkl          # Saved model pipeline (created after running the notebook)
├── requirements.txt
└── README.md
```

## Pipeline

1. **Load the data** directly from the source URL
2. **Check data quality** — duplicate IDs, the missing `TotalCharges` values, fixing column types
3. **Explore the data** — class balance, how numeric features split by churn, churn rate per category, correlations
4. **Engineer a few features** — tenure buckets, a count of subscribed services, average monthly spend
5. **Preprocess** — scale numeric columns, one-hot encode categoricals, all inside a `ColumnTransformer`
6. **Handle the class imbalance** with SMOTE, applied only within the training folds
7. **Compare models** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, KNN
8. **Tune the best model** with `GridSearchCV` and 5-fold stratified cross-validation, optimizing ROC-AUC
9. **Evaluate** — confusion matrix, ROC curve, precision-recall curve, full classification report
10. **Interpret** — feature importances from XGBoost plus a SHAP summary plot
11. **Save the pipeline** with `joblib` so it can be reused for scoring without retraining

## Installation

```bash
git clone https://github.com/<your-username>/churnguard.git
cd churnguard
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Open the notebook and run all cells:

```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```

That reproduces the whole analysis, trains and tunes the models, and saves `churn_model_pipeline.pkl` at the end.

To score new customers with the saved pipeline:

```python
import joblib

pipeline = joblib.load("churn_model_pipeline.pkl")
proba = pipeline.predict_proba(new_customers_df)[:, 1]
prediction = (proba >= 0.5).astype(int)
```

`new_customers_df` just needs the same raw columns as the training data — the pipeline takes care of scaling and encoding on its own.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (tuned) | ~0.80 | ~0.65 | ~0.72 | ~0.68 | ~0.85 |
| LightGBM | ~0.79 | ~0.63 | ~0.70 | ~0.66 | ~0.84 |
| Gradient Boosting | ~0.79 | ~0.63 | ~0.69 | ~0.66 | ~0.84 |
| Random Forest | ~0.78 | ~0.61 | ~0.67 | ~0.64 | ~0.82 |
| Logistic Regression | ~0.77 | ~0.59 | ~0.68 | ~0.63 | ~0.81 |

These numbers come from running the notebook — expect small differences depending on your environment and random seed.

## Key Churn Drivers

From the EDA and feature importance results, a few things stood out:

- **Contract type** — month-to-month customers churn a lot more than customers on one- or two-year contracts
- **Tenure** — churn is concentrated in the first 12 months
- **Internet service** — fiber optic customers churn more than DSL customers
- **Add-on services** — customers without online security or tech support churn more
- **Payment method** — electronic check payers churn more than customers on automatic payment methods

## Tech Stack

- **Data processing:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, XGBoost, LightGBM
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **Interpretability:** SHAP
- **Persistence:** joblib

## Business Recommendations

1. Target month-to-month customers in their first year with retention offers or discounted contract upgrades
2. Bundle tech support and online security into fiber optic plans to make the service stickier
3. Encourage electronic-check customers to switch to automatic payments
4. Run the saved pipeline as a monthly batch job to flag high-risk customers for the retention team

## License

MIT License. See [LICENSE](LICENSE) for details.
