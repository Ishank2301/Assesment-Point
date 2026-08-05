import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from src.data_loader import ensure_dir

CITY_COORDS = {
    "Lexington": (36.99152, -84.99876),
    "Fort Wayne": (41.31561, -85.36206),
}


def enrich_december_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_lat"] = (
        df["pickup"]
        .map(lambda city: CITY_COORDS.get(city, None))
        .map(lambda value: value[0] if value is not None else None)
    )
    df["pickup_lon"] = (
        df["pickup"]
        .map(lambda city: CITY_COORDS.get(city, None))
        .map(lambda value: value[1] if value is not None else None)
    )
    df["delivery_lat"] = (
        df["delivery"]
        .map(lambda city: CITY_COORDS.get(city, None))
        .map(lambda value: value[0] if value is not None else None)
    )
    df["delivery_lon"] = (
        df["delivery"]
        .map(lambda city: CITY_COORDS.get(city, None))
        .map(lambda value: value[1] if value is not None else None)
    )
    df["market_index"] = pd.NA
    df["quote_signal"] = pd.NA
    return df


def main():
    ensure_dir("data")
    pipeline = joblib.load("models/pipeline.pkl")
    model = joblib.load("models/model.pkl")

    dec_path = Path("data/december-chart-inputs.csv")
    df = pd.read_csv(dec_path)
    df["date"] = pd.to_datetime(df["date"])
    df = enrich_december_inputs(df)

    transformed = pipeline.transform(df)
    feature_cols = pipeline.feature_columns()
    X = transformed[feature_cols]
    df["predicted_rate"] = model.predict(X)

    out_path = Path("data/december_predictions.csv")
    df[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ].to_csv(
        out_path,
        index=False,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
