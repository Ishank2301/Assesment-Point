from pathlib import Path
import pandas as pd

TRAIN_TEST_COLUMN = {
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate",
}

VALIDATION_COLUMNS = TRAIN_TEST_COLUMN - {"posted_rate"}


def load_train_test(path: str = "data/train-test.csv") -> pd.DataFrame:
    # Load the labelled development data:
    df = pd.read_csv(path)
    _check_columns(df, TRAIN_TEST_COLUMN, path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_validation(path: str = "data/validation.csv") -> pd.DataFrame:
    # Load the unlabeled validation data used for prediction.
    df = pd.read_csv(path)
    _check_columns(df, VALIDATION_COLUMNS, path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_validation_template(
    path: str = "data/validation-predictions-template.csv",
) -> pd.DataFrame:
    # Load the template in that final prediction must be written:
    df = pd.read_csv(path)

    if "load_id" not in df.columns:
        raise ValueError(f"{path} is missing froom the required 'load_id' column.")
    return df


def _check_columns(df: pd.DataFrame, expected: set, path: str) -> None:
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
