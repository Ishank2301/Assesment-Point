import json

import joblib
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_train_test, ensure_dir
from src.split import time_based_split
from src.features import FeaturePipeline
from src.model import build_model, train_model, evaluate, feature_importance


def main():
    ensure_dir(ROOT / "models")
    ensure_dir(ROOT / "reports" / "figures")

    print("Loading data...")
    df = load_train_test(ROOT / "data" / "train-test.csv")

    print("Splitting (time-based)...")
    train_df, val_df = time_based_split(df, train_end="2025-08-31")
    print(
        f"  train: {len(train_df)} rows ({train_df['date'].min().date()} - {train_df['date'].max().date()})"
    )
    print(
        f"  val:   {len(val_df)} rows ({val_df['date'].min().date()} - {val_df['date'].max().date()})"
    )

    print("Fitting feature pipeline on train only...")
    pipeline = FeaturePipeline()
    train_feat = pipeline.fit_transform(train_df)
    val_feat = pipeline.transform(val_df)  # transform only -- never fit on val

    feature_cols = pipeline.feature_columns()
    cat_cols = pipeline.CATEGORICAL_FEATURES

    X_train, y_train = train_feat[feature_cols], train_feat["posted_rate"]
    X_val, y_val = val_feat[feature_cols], val_feat["posted_rate"]

    print("Training model...")
    model = build_model()
    model = train_model(
        model, X_train, y_train, X_val, y_val, categorical_features=cat_cols
    )

    print("Evaluating...")
    train_metrics = evaluate(model, X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)
    print(f"  train: {train_metrics}")
    print(f"  val:   {val_metrics}")

    # feature importance chart
    fi = feature_importance(model, feature_cols)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#2980b9")
    ax.set_title("Feature importance")
    plt.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "feature_importance.png", dpi=150)
    plt.close(fig)

    # persist artifacts
    joblib.dump(pipeline, ROOT / "models" / "pipeline.pkl")
    joblib.dump(model, ROOT / "models" / "model.pkl")

    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump({"train": train_metrics, "val": val_metrics}, f, indent=2)

    print("Saved models/pipeline.pkl, models/model.pkl, reports/metrics.json")


if __name__ == "__main__":
    main()
