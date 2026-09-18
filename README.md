# High Resource Utilizer Prediction

A machine learning project that predicts, at the point of hospital admission, which patients are likely to become "high resource utilizers" — hospitalizations in the top 25% by length of stay. Built as a capstone project using the MIMIC-IV clinical dataset.

## What's in this repository

- **`High_Risk_Patient_Classification_FINAL.ipynb`** — the full Colab notebook: data cleaning, feature engineering, training and comparing seven classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, K-Nearest Neighbors, Naive Bayes), hyperparameter tuning, calibration checks, SHAP interpretability, bootstrap significance testing, and a fairness audit.
- **`dashboard_app.py`** — an interactive Streamlit dashboard with five views: Executive Summary, Care Team Priority Outreach, Operations & Capacity Planning, Full Patient Risk List, and Model Transparency & Fairness.

## Deployed model

**Gradient Boosting** — AUROC 0.852, Recall 81.4%, selected after applying consistent class-weighting across all seven models and tuning each one under matching settings.

## Data

This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/), a large, freely accessible, deidentified electronic health record dataset from Beth Israel Deaconess Medical Center, available via PhysioNet (credentialed access required). Raw patient data is **not included in this repository**.

## Running the dashboard

```bash
pip install streamlit plotly pandas
streamlit run dashboard_app.py
```

The dashboard reads its input from a local `outputs_classification` folder (the same folder the notebook saves its results into — CSVs and figures). Point the sidebar "Outputs folder" field at that folder's path after launching.

## Author

Vineesh Velluvayala — REVA University, PGDBA
