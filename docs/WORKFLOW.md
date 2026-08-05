Implementation workflow (ordered)

Phase 0 — Setup
- Confirm files in `data/`:
  - `train-test.csv` (development labeled data)
  - `validation.csv` (12,000 rows requiring predictions)
  - `validation-predictions-template.csv`
- Create virtualenv and install packages from `requirements.txt`.

Phase 1 — Exploratory Data Analysis (EDA)
- Load data and assess:
  - Column types, missingness, unique values
  - Target distribution and leaks
  - Correlations and simple visual checks
- Produce an `01_eda.ipynb` notebook that documents findings.

Phase 2 — Data cleaning and quality
- Handle missing values, duplicates, and impossible values.
- Normalize or standardize fields where appropriate.
- Save cleaned datasets under `data/processed/`.

Phase 3 — Feature engineering
- Create candidate features (date/time decompositions, aggregations, encodings).
- Keep feature transforms deterministic and serializable (functions in `src/features.py`).

Phase 4 — Validation strategy
- Decide split (time-based, grouped, or stratified) and justify choice in the report.
- Implement reproducible folds and logging of fold metrics.

Phase 5 — Modeling
- Start with simple baselines (mean, linear model) to set expectations.
- Progress to tree-based models (XGBoost/LightGBM) and tune hyperparameters.

Phase 6 — Evaluation
- Evaluate on held-out folds and select final model using logged metrics.
- Perform error analysis and sanity checks (extreme predictions, distribution drift).

Phase 7 — Produce final predictions
- Create `scripts/predict_validation.py` that loads the final model and writes `validation_predictions.csv` with exactly two columns: `load_id,predicted_rate`.

Phase 8 — Reporting and delivery
- Create a concise PDF/DOCX covering validation approach, final chart, and key findings.
- Record a 2–3 minute video walk-through covering requested points in the assignment.
