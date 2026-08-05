from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 42

DEFAULT_PARAMS = dict(
    objective="regression",
    random_state=RANDOM_SEED,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
)


def build_model(**overrides) -> lgb.LGBMRegressor:
    params = dict(DEFAULT_PARAMS)
    params.update(overrides)
    return lgb.LGBMRegressor(**params)


def train_model(
    model: lgb.LGBMRegressor,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    categorical_features: list[str] | None = None,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMRegressor:
    fit_kwargs = {}
    if categorical_features:
        fit_kwargs["categorical_feature"] = categorical_features

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
            **fit_kwargs,
        )
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    return model


def evaluate(model: lgb.LGBMRegressor, X, y) -> dict:
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mape = float(np.mean(np.abs((y - preds) / y)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def feature_importance(model: lgb.LGBMRegressor, feature_names: list[str]):
    import pandas as pd

    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
