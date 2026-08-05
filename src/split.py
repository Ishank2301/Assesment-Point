import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = "2025-08-31",
):

    # Split `df` into (train, val) using a date cutoff.

    # Rows with date <= train_end go to train; everything after goes to val.
    # Default cutoff leaves ~8 months (Jan-Aug) for training and ~2 months
    # (Sep-Oct) as a held-out, forward-looking validation window.

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = pd.Timestamp(train_end)

    train = df[df[date_col] <= cutoff].reset_index(drop=True)
    val = df[df[date_col] > cutoff].reset_index(drop=True)

    if len(train) == 0 or len(val) == 0:
        raise ValueError(
            f"Split produced an empty set (train={len(train)}, val={len(val)}). "
            f"Check that train_end='{train_end}' falls inside the data's date range."
        )

    return train, val
