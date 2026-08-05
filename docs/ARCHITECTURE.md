Project architecture and folder rationale

Goal
- Provide a clean, testable layout so experiments and production artifacts are separated and reproducible.

Folders
- data/: Raw and working datasets. Keep originals read-only and create derived files under `data/processed/`.
- notebooks/: Exploratory analysis notebooks. Use clear names like `01_eda.ipynb`, `02_feature_engineering.ipynb`.
- src/: Production-style code organized as packages:
  - src/data.py — data loading and basic validation
  - src/features.py — feature engineering functions
  - src/models.py — model training and evaluation wrappers
  - src/predict.py — load a model and run predictions on validation.csv
- models/: Store trained model files and metadata (version, params, metrics).
- reports/: Final PDF/DOCX and diagnostic plots used in the report.
- scripts/: CLI helpers for training, evaluation, and creating `validation_predictions.csv`.

Design notes
- Keep notebooks for discovery; implement stable pipelines in `src/` for reproducibility.
- Use small utilities in `scripts/` to glue steps (e.g., `scripts/run_train.sh` or `run_train.ps1`).
- Store experiment results (metrics, validation folds) alongside models for traceability.

Validation guidance
- Choose a validation strategy appropriate for the data (time-based split, grouped split by entity, or stratified k-fold). Document the choice in `reports/`.
