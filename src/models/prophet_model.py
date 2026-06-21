# ============================================================
# src/models/prophet_model.py
# ============================================================

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import holidays as hol
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

ACCENT  = "#00d4ff"   # cyan
ACCENT2 = "#ff6b6b"   # coral
ACCENT3 = "#51cf66"   # green
ACCENT4 = "#ffd43b"   # amber

from src.config import (
    PROPHET_CHANGEPOINT_PRIOR, PROPHET_SEASONALITY_PRIOR,
    PROPHET_HOLIDAYS_PRIOR, FORECAST_HORIZON_DAYS, FIGURES_DIR
)
from src.config import ACCENT, ACCENT2, ACCENT3, ACCENT4

def build_german_holidays_df(years: list) -> pd.DataFrame:
    """
    Convert Python holidays library output into the DataFrame format
    Prophet expects: columns 'ds' (date) and 'holiday' (name string).

    We also add window columns (lower_window, upper_window) to tell
    Prophet that the effect of a holiday extends into surrounding days.
    lower_window=-2 means 'start capturing the effect 2 days BEFORE the holiday'
    upper_window=1  means 'continue capturing the effect 1 day AFTER'
    This handles the pre-holiday spike we saw in EDA.
    """
    de_holidays = hol.Germany(years=years)
    holiday_df = pd.DataFrame([
        {"ds": pd.Timestamp(date), "holiday": name}
        for date, name in de_holidays.items()
    ])
    # Pre-holiday window: effect starts 2 days before
    holiday_df["lower_window"] = -2
    # Post-holiday window: effect lingers 1 day after
    holiday_df["upper_window"] = 1
    return holiday_df


def prepare_prophet_data(df: pd.DataFrame, store_id: int) -> pd.DataFrame:
    """
    Filter to a single store and rename columns to Prophet's required format.
    Prophet is strict: it ONLY accepts columns named 'ds' and 'y'.
    Any other columns are treated as additional regressors.
    """
    store_df = df[df["Store"] == store_id].copy()
    store_df = store_df[["Date", "Sales", "Promo", "Promo2Active",
                          "IsGermanHoliday", "StateHoliday"]].copy()
    store_df = store_df.rename(columns={"Date": "ds", "Sales": "y"})
    store_df = store_df.sort_values("ds").reset_index(drop=True)
    return store_df


def train_prophet_model(train_data: pd.DataFrame,
                        holiday_df: pd.DataFrame) -> Prophet:
    """
    Instantiate and train a Prophet model.

    Parameter explanations:
    ----------------------
    changepoint_prior_scale (default 0.05):
        How flexible the trend is allowed to be. Higher values (0.5)
        allow the trend to change direction frequently — risk of overfitting.
        Lower values (0.001) make the trend rigid — risk of missing real shifts.
        0.05 is a reasonable starting point for retail demand.

    seasonality_prior_scale (default 10.0):
        How strong the seasonal components are allowed to be. Higher means
        the model can fit more pronounced seasonal swings. 10.0 is generous —
        retail usually has strong seasonality so this is appropriate.

    holidays_prior_scale (default 10.0):
        How large the holiday effects are allowed to be. Same reasoning.

    seasonality_mode ('multiplicative' vs 'additive'):
        Additive: seasonal_effect = fixed number of units above/below baseline.
        Multiplicative: seasonal_effect = fixed PERCENTAGE above/below baseline.
        For retail, multiplicative is usually better — the December uplift in
        a high-volume store is larger in absolute terms than in a low-volume
        store, but the percentage uplift is similar. Multiplicative captures this.
    """
    model = Prophet(
        changepoint_prior_scale   = PROPHET_CHANGEPOINT_PRIOR,
        seasonality_prior_scale   = PROPHET_SEASONALITY_PRIOR,
        holidays_prior_scale      = PROPHET_HOLIDAYS_PRIOR,
        seasonality_mode          = "multiplicative",
        holidays                  = holiday_df,
        weekly_seasonality        = True,
        yearly_seasonality        = True,
        daily_seasonality         = False,   # We have daily data but no sub-daily
    )

    # Add promotion as an additional regressor.
    # Prophet treats these as linear effects on top of trend+seasonality+holidays.
    model.add_regressor("Promo",        standardize=False)
    model.add_regressor("Promo2Active", standardize=False)

    model.fit(train_data)
    return model


def generate_forecast(model: Prophet,
                      future_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate forecast including uncertainty intervals.

    The output columns we care about:
    - yhat        : the point forecast (single best estimate)
    - yhat_lower  : lower bound of the 80% uncertainty interval
    - yhat_upper  : upper bound of the 80% uncertainty interval

    What uncertainty intervals mean for supply chain:
    A supply chain planner should ORDER based on yhat_upper, not yhat.
    Ordering to the point forecast means you will run out in 50% of
    scenarios (the interval is symmetric). Ordering to the upper bound
    provides safety cover. The choice of which bound to use is
    equivalent to choosing a service level — this is the connection
    between the statistical forecast and inventory policy.
    """
    forecast = model.predict(future_df)
    # Ensure forecasts are non-negative (sales cannot be below zero)
    forecast["yhat"]       = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)
    return forecast


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Calculate all four forecast accuracy metrics.

    MAPE (Mean Absolute Percentage Error):
        Average of |actual - forecast| / actual, expressed as %.
        Most intuitive for business communication — '15% MAPE' means
        'on average our forecast is 15% off from actual demand'.
        Limitation: undefined when actual = 0 (division by zero).

    MAE (Mean Absolute Error):
        Average of |actual - forecast| in original units (euros/units).
        Useful for understanding the absolute scale of errors.
        'Our average daily forecast error is €2,400' is meaningful.

    RMSE (Root Mean Squared Error):
        Square root of the average of squared errors.
        Penalises large errors more than MAE — a single day with
        €10,000 error counts more than ten days with €1,000 error.
        Use this when large errors are disproportionately costly
        (e.g., stockouts at peak demand are more damaging than small
        daily misses).

    WAPE (Weighted Absolute Percentage Error):
        Sum of |actual - forecast| / Sum of actual.
        More robust than MAPE when some actual values are near zero.
        The standard metric used by SAP IBP and most enterprise tools.
    """
    # Filter out zero actuals to avoid MAPE division by zero
    mask = actual > 0
    actual_f    = actual[mask]
    predicted_f = predicted[mask]

    mape  = np.mean(np.abs((actual_f - predicted_f) / actual_f)) * 100
    mae   = mean_absolute_error(actual, predicted)
    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    wape  = np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

    return {"MAPE": round(mape, 2), "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "WAPE": round(wape, 2)}


def plot_forecast(forecast: pd.DataFrame, 
                  actuals: pd.DataFrame,
                  store_id: int,
                  split_date: str):
    """
    Plot actual vs forecasted sales with confidence interval.
    This is the primary visual for Page 2 of the Power BI dashboard.
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    # Confidence interval band
    ax.fill_between(forecast["ds"], 
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    alpha=0.2, color=ACCENT, label="80% Confidence Interval")

    # Forecast line
    ax.plot(forecast["ds"], forecast["yhat"],
            color=ACCENT, linewidth=1.8, label="Prophet Forecast")

    # Actual values
    ax.scatter(actuals["ds"], actuals["y"],
               s=4, color=ACCENT2, alpha=0.6, label="Actual Sales", zorder=5)

    # Train/test boundary
    ax.axvline(pd.Timestamp(split_date), color=ACCENT4,
               linewidth=2, linestyle="--", label="Train | Test split")

    ax.set_title(f"Prophet Forecast vs Actual — Store {store_id}",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sales (€)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/forecasts/prophet_store_{store_id}.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1923")
    plt.show()