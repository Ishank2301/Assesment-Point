# Model Assessment Report

## Overview

This report documents the end-to-end model pipeline, validation strategy, and December prediction chart generation for the rate prediction assessment.

## Data and Split Strategy

- Training data: `data/train-test.csv` with 48,000 rows covering 2025-01-01 through 2025-10-31.
- Validation data: `data/validation.csv` contains 12,000 unlabeled load requests.
- The model uses a time-based split for development: training through 2025-08-31, validation for 2025-09-01 through 2025-10-31.
- This preserves temporal order and ensures December predictions are out-of-sample.

## Feature Engineering

- Derived features:
  - `haversine_dist`: geographic great-circle distance from pickup to delivery.
  - `distance_ratio`: ratio of reported distance to haversine distance.
  - `month_sin` / `month_cos`: cyclical encoding for date seasonality.
  - `day_of_week`: day-level seasonality.
  - `lane`: pickup and delivery concatenated.
- Data-quality handling:
  - Negative weights converted to absolute values and flagged.
  - Missing weights imputed with training median.
  - Missing `market_index` imputed by month median, then global median.
  - `quote_signal` is imputed with training median for December inference.

## Model

- Algorithm: LightGBM regression (`LGBMRegressor`).
- Hyperparameters: 2000 estimators, learning rate 0.03, 31 leaves, 0.8 subsample, 0.8 colsample_bytree.
- Categorical features: `equipment`, `lane`.

## Results

- Training MAE: 127.23
- Training RMSE: 514.70
- Training MAPE: 7.01%
- Validation MAE: 157.43
- Validation RMSE: 648.44
- Validation MAPE: 7.57%

## December Prediction Workflow

- December inputs are in `data/december-chart-inputs.csv`.
- The model predicts candidate rates for the fixed route: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb.
- Predictions are written to `data/december_predictions.csv` and validated with `score.py`.
- The final December chart is `scorer_results/candidate_december.png`.

## Validation and Output

- Final validation predictions: `data/validation_predictions.csv` (12,000 rows).
- December candidate chart: `scorer_results/candidate_december.png`.

## Notes

- December is outside the original training window, so cyclical month encoding is essential.
- The model is more reliable when it uses distance, route lane, and seasonality features rather than direct calendar labels.

