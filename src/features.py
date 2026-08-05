"""We will create a FeaturePipeline in this case. Katen Kyotsu KaramasuShinjikui Bankaaiiiiii
FeaturePipeline: All Cleaning and feature engineering must be performed in this.
Transformation of data is done according to the 1_eda.ipynb and its result."""

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    # Great Circle distance between two loacations in miles:

    r = 3959.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


class FeaturePipeline:
    NUMERIC_FEATURES = [
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "haversine_dist",
        "distance_ratio",
        "month_sin",
        "month_cos",
        "day_of_week",
    ]

    CATEGORICAL_FEATURES = ["equipment", "lane"]
    FLAG_FEATURES = [
        "weight_was_negative",
        "weight_was_missing",
        "market_index_was_missing",
    ]

    def __init__(self):
        self.weight_median_: float | None = None
        self.market_index_by_month_: dict | None = None
        self.market_index_global_median_: float | None = None
        self.quote_signal_global_median_: float | None = None
        self.is_fitted_: bool = False

    # Public API: A way to use the Pipeline externally for other data of the same type
    # Increases reusability and versatility:

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        pre = self._pre_clean(df)

        self.weight_median_ = float(pre["weight"].median())
        self.market_index_by_month_ = (
            pre.groupby("month")["market_index"].median().to_dict()
        )
        self.market_index_global_median_ = float(pre["market_index"].median())
        self.quote_signal_global_median_ = float(pre["quote_signal"].median())
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError(
                "FeatuePipeline() is called before fit()."
                "Fit on training data first, or load a pre fitted pipeline."
            )
        out = self._pre_clean(df)

        # Impute using train-derived statistics only:
        out["weight"] = out["weight"].fillna(self.weight_median_)

        out["market_index"] = out.apply(
            lambda r: (
                self.market_index_by_month_.get(
                    r["month"], self.market_index_global_median_
                )
                if pd.isna(r["market_index"])
                else r["market_index"]
            ),
            axis=1,
        )

        out["quote_signal"] = out["quote_signal"].fillna(
            self.quote_signal_global_median_
        )

        # seasonality:
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

        # lane acc to delivery
        out["lane"] = out["pickup"].astype(str) + "_" + out["delivery"].astype(str)

        # geo sanity features
        out["haversine_dist"] = haversine_miles(
            out["pickup_lat"],
            out["pickup_lon"],
            out["delivery_lat"],
            out["delivery_lon"],
        )
        out["distance_ratio"] = out["distance"] / out["haversine_dist"].replace(
            0, np.nan
        )
        out["distance_ratio"] = out["distance_ratio"].fillna(1.0)

        for c in self.CATEGORICAL_FEATURES:
            out[c] = out[c].astype("category")

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def feature_columns(self) -> list[str]:
        return self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES + self.FLAG_FEATURES

    # Internal:
    @staticmethod
    def _pre_clean(df: pd.DataFrame) -> pd.DataFrame:
        "Function not depending on any fitted statistics"
        out = df.copy()

        # Fix negative, missing, and dense same weights:
        out["weight_was_negative"] = out["weight"] < 0
        out["weight"] = out["weight"].abs()
        out["weight_was_missing"] = out["weight"].isna()

        out["market_index_was_missing"] = out["market_index"].isna()

        out["date"] = pd.to_datetime(out["date"])
        out["month"] = out["date"].dt.month
        out["day_of_week"] = out["date"].dt.dayofweek
        return out
