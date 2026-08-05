import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from src.data_loader import ensure_dir, load_validation, load_validation_template


def main():
    ensure_dir("data")

    print("Loading fitted pipeline and model...")
    pipeline = joblib.load("models/pipeline.pkl")
    model = joblib.load("models/model.pkl")

    print("Loading validation.csv...")
    val_df = load_validation("data/validation.csv")

    print("Transforming (no fitting)...")
    val_feat = pipeline.transform(val_df)
    X_val = val_feat[pipeline.feature_columns()]

    print("Predicting...")
    val_feat = val_feat.copy()
    val_feat["predicted_rate"] = model.predict(X_val)

    print("Merging onto template by load_id...")
    template = load_validation_template("data/validation-predictions-template.csv")

    out = template[["load_id"]].merge(
        val_feat[["load_id", "predicted_rate"]], on="load_id", how="left"
    )

    missing = out["predicted_rate"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} load_id(s) in the template had no matching prediction. "
            "Check that validation.csv contains every load_id in the template."
        )

    out.to_csv("data/validation_predictions.csv", index=False)
    print(f"Wrote data/validation_predictions.csv ({len(out)} rows)")
    print(out["predicted_rate"].describe())


if __name__ == "__main__":
    main()
