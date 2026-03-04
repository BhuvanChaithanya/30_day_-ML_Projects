# Day 1 — Salary Predictor (Linear Regression)

> Part of the [30-Day ML Challenge](https://github.com/BhuvanChaithanya/30-day-ml-challenge)

A Streamlit app that predicts salary from months of experience using a simple Linear Regression model trained on 1,000 data points.

## Features

- Interactive slider & number input for experience (months)
- Live salary prediction displayed in a gradient card
- Scatter plot with regression line, highlighting your prediction
- Actual vs Predicted chart on the held-out test set
- Model metrics: R², RMSE, regression equation
- Dataset explorer

## Tech Stack

`Python 3.12` · `scikit-learn` · `Streamlit` · `Matplotlib` · `Pandas` · `NumPy`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

| Metric | Value |
|--------|-------|
| Algorithm | Linear Regression |
| R² Score | ~0.62 |
| RMSE | ~5.26 $K |
| Data | 1,000 rows, 80/20 train-test split |
