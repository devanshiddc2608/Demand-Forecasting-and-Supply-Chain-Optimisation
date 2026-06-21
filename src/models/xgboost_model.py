# ============================================================
# src/models/xgboost_model.py
# ============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.models.prophet_model import calculate_metrics
from src.config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE, FIGURES_DIR
)
ACCENT  = "#00d4ff"   # cyan
ACCENT2 = "#ff6b6b"   # coral
ACCENT3 = "#51cf66"   # green
ACCENT4 = "#ffd43b"   # amber

# ---- Define the feature set ----
# These are the columns that go into the model as inputs (X).
# Notably absent: Date, Sales (the target), Sales_log (derived target),
# and any columns that would constitute data leakage.

FEATURE_COLS = [
    # Calendar features
    "DayOfWeek", "Month", "WeekOfYear", "Quarter",
    "DayOfYear", "Year", "IsWeekend", "IsMonthStart", "IsMonthEnd",

    # Holiday features
    "IsGermanHoliday", "IsPreHoliday", "SchoolHoliday",

    # Store identity and characteristics
    # StoreType and Assortment must be label-encoded (strings → numbers)
    "Store", "StoreType_enc", "Assortment_enc",

    # Promotional features
    "Promo", "Promo2Active",

    # Competition features
    "CompetitionDistance", "HasCompetitor", "MonthsSinceCompOpen",

    # Lag features (the 'memory' of the time series)
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14",
    "Sales_lag_21", "Sales_lag_28",

    # Rolling window features (the 'context' of current demand level)
    "Rolling_mean_7", "Rolling_std_7",
    "Rolling_mean_14", "Rolling_std_14",
    "Rolling_mean_28", "Rolling_std_28",
]

TARGET_COL = "Sales_log"    # We predict log(1+Sales) and inverse-transform


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost requires numeric inputs.
    StoreType (a/b/c/d) and Assortment (a/b/c) are strings — encode them.
    Label encoding is appropriate here (not one-hot) because XGBoost
    handles arbitrary numeric categories through its tree structure.
    """
    df = df.copy()
    type_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    asm_map  = {"a": 0, "b": 1, "c": 2}
    df["StoreType_enc"]  = df["StoreType"].map(type_map).fillna(-1)
    df["Assortment_enc"] = df["Assortment"].map(asm_map).fillna(-1)
    return df


def train_xgboost_model(train: pd.DataFrame,
                        val: pd.DataFrame) -> xgb.XGBRegressor:
    """
    Train a GLOBAL XGBoost model — one model for ALL stores simultaneously.

    Why global over local (one model per store)?
    A local model for a low-volume store (C-class) might only have 400 rows
    of history. XGBoost with 500 trees trained on 400 rows will overfit badly.
    A global model trained on all 1,115 stores × ~700 days = ~780,000 rows
    has far more signal. The model learns patterns that generalise across
    stores — if promotions always uplift Store A by 20%, the model can infer
    a similar effect for Store B even if Store B has had fewer promotions.

    The store_id is a feature (encoded as an integer), so the model can
    also learn store-specific level effects where the data supports it.

    Hyperparameter explanations:
    ----------------------------
    n_estimators    : number of trees in the ensemble (500 is a good start)
    max_depth       : how deep each tree grows (6 = moderate complexity)
    learning_rate   : how much each new tree corrects the previous (lower = more trees needed but better generalisation)
    subsample       : fraction of rows used for each tree (0.8 = 80% random sample, prevents overfitting)
    colsample_bytree: fraction of FEATURES used for each tree (0.8 = 80%, adds diversity)
    early_stopping_rounds: stop training if validation score does not improve after N rounds
    """
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_val   = val[FEATURE_COLS]
    y_val   = val[TARGET_COL]

    model = xgb.XGBRegressor(
        n_estimators         = XGB_N_ESTIMATORS,
        max_depth            = XGB_MAX_DEPTH,
        learning_rate        = XGB_LEARNING_RATE,
        subsample            = XGB_SUBSAMPLE,
        colsample_bytree     = XGB_COLSAMPLE_BYTREE,
        objective            = "reg:squarederror",
        eval_metric          = "rmse",
        early_stopping_rounds= 50,
        random_state         = 42,
        n_jobs               = -1,     # use all CPU cores
        verbosity            = 1,
    )

    model.fit(
        X_train, y_train,
        eval_set            = [(X_val, y_val)],
        verbose             = 100,    # print every 100 rounds
    )

    print(f"Best iteration: {model.best_iteration}")
    return model


def predict_and_evaluate(model: xgb.XGBRegressor,
                          test: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions and evaluate.

    We predict log(1+Sales), then inverse-transform with expm1()
    (which is exp(x)-1, the inverse of log1p) to get back to euros.
    """
    X_test = test[FEATURE_COLS]
    preds_log = model.predict(X_test)

    # Inverse log transform — get predictions back in original euro scale
    preds_sales = np.expm1(preds_log).clip(min=0)

    results = test[["Store","Date","Sales","ABC_Class","XYZ_Class"]].copy()
    results["Predicted_Sales"] = preds_sales

    # Calculate metrics on original scale
    metrics = calculate_metrics(results["Sales"], results["Predicted_Sales"])
    return results, metrics


def plot_feature_importance(model: xgb.XGBRegressor,
                             top_n: int = 20):
    """
    Plot the most important features in the XGBoost model.

    Feature importance (by 'gain') tells us how much each feature
    REDUCES PREDICTION ERROR when used in a split.

    For a supply chain manager, this answers: 'what information matters
    most for predicting demand?' If Promo has the highest importance,
    it confirms that promotional planning is the most powerful lever
    for demand management — and that the commercial team must share
    the promotional calendar with supply chain in advance.
    """
    importance = pd.DataFrame({
        "Feature"   : FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance["Feature"], importance["Importance"],
                   color=ACCENT, alpha=0.85, edgecolor="none")
    ax.set_title(f"XGBoost Feature Importance (Top {top_n} by Gain)",
                 fontweight="bold")
    ax.set_xlabel("Relative Importance (Gain)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/forecasts/xgb_feature_importance.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1923")
    plt.show()


def plot_xgb_vs_actual(results: pd.DataFrame, store_id: int,
                        split_date: str):
    """Actual vs predicted for a single store — for portfolio visuals."""
    store_results = results[results["Store"] == store_id].sort_values("Date")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(store_results["Date"], store_results["Sales"],
            color=ACCENT2, linewidth=1.5, label="Actual Sales", alpha=0.8)
    ax.plot(store_results["Date"], store_results["Predicted_Sales"],
            color=ACCENT3, linewidth=1.5, label="XGBoost Forecast",
            linestyle="--", alpha=0.9)

    ax.set_title(f"XGBoost Forecast vs Actual — Store {store_id}",
                 fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sales (€)")
    ax.legend()
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/forecasts/xgb_store_{store_id}.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1923")
    plt.show()