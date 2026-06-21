# ============================================================
# src/feature_engineering.py
# All feature creation functions — called from notebooks.
# ============================================================

import pandas as pd
import numpy as np
import holidays as hol
from src.config import LAG_DAYS, ROLLING_WINDOWS, TRAIN_END_DATE


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar features from the Date column.

    Why each feature matters for demand forecasting:
    - DayOfWeek    : captures weekly seasonality (Sat peak)
    - Month        : captures annual seasonality (Dec peak)
    - WeekOfYear   : finer-grained annual cycle (52 data points vs 12)
    - Quarter      : useful for quarterly business review patterns
    - IsWeekend    : binary flag — weekend shopping behaviour differs fundamentally
    - IsMonthStart : salary payment date in Germany — affects consumer spending
    - IsMonthEnd   : similar effect at month end
    - DayOfYear    : 1–365 continuous signal for annual cycle
    - Year         : allows the model to learn the growth trend
    """
    df = df.copy()
    df["DayOfWeek"]   = df["Date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["Month"]        = df["Date"].dt.month
    df["WeekOfYear"]   = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]      = df["Date"].dt.quarter
    df["DayOfYear"]    = df["Date"].dt.dayofyear
    df["Year"]         = df["Date"].dt.year
    df["IsWeekend"]    = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"]   = df["Date"].dt.is_month_end.astype(int)
    return df


def add_german_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create precise German public holiday features.

    The `holidays` library gives us exact German holiday dates by state.
    We use national (DE) holidays as a baseline — state-level would be
    more precise but requires knowing each store's state, which the
    dataset does not provide directly.

    We also create a pre-holiday flag (1–3 days before) because the EDA
    showed a demand spike in the pre-holiday window.
    """
    de_holidays = hol.Germany(years=df["Date"].dt.year.unique().tolist())
    holiday_dates = set(de_holidays.keys())

    df = df.copy()
    df["IsGermanHoliday"] = df["Date"].dt.date.isin(holiday_dates).astype(int)

    # Pre-holiday flag: 1 if the date is 1, 2, or 3 days before a holiday
    df["IsPreHoliday"] = df["Date"].apply(
        lambda d: int(any((d + pd.Timedelta(days=i)).date() in holiday_dates
                         for i in range(1, 4)))
    )

    return df


def add_promo2_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary flag indicating whether Promo2 is active for a store
    on a given date.

    PromoInterval contains strings like 'Jan,Apr,Jul,Oct' — these are the
    months in which Promo2 restarts. We check whether the current month
    falls in that interval AND whether the date is after Promo2Since.
    """
    month_map = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,
    "Sep":9, "Sept":9,   # ✅ BOTH handled
    "Oct":10,"Nov":11,"Dec":12
}

    def is_promo2_active(row):
        if row["PromoInterval"] == "None" or row["Promo2"] == 0:
            return 0
        active_months = [month_map[m] for m in row["PromoInterval"].split(",")]
        if row["Month"] not in active_months:
            return 0
        # Check if Promo2 has already started
        if row["Year"] < row["Promo2SinceYear"]:
            return 0
        if (row["Year"] == row["Promo2SinceYear"] and
                row["WeekOfYear"] < row["Promo2SinceWeek"]):
            return 0
        return 1

    df = df.copy()
    df["Promo2Active"] = df.apply(is_promo2_active, axis=1)
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform competitor data into model-ready features.

    CompetitionDistance is already clean (NaN filled with 99999).
    We add a binary 'HasCompetitor' flag because the distance value
    and the binary presence signal may have different predictive power.
    We also calculate months since competitor opened — the effect of
    a competitor may be strongest immediately after opening, then plateau.
    """
    df = df.copy()
    df["HasCompetitor"] = (df["CompetitionDistance"] < 99999).astype(int)

    # Months since competitor opened (0 if no competitor)
    df["CompOpenYear"]  = df["CompetitionOpenSinceYear"].astype(int)
    df["CompOpenMonth"] = df["CompetitionOpenSinceMonth"].astype(int)

    def months_since_comp(row):
        if row["CompOpenYear"] == 0:
            return 0
        months = ((row["Year"] - row["CompOpenYear"]) * 12 
                  + (row["Month"] - row["CompOpenMonth"]))
        return max(months, 0)   # negative means competitor not yet open

    df["MonthsSinceCompOpen"] = df.apply(months_since_comp, axis=1)
    return df


def add_lag_features(df: pd.DataFrame, 
                     lag_days: list = LAG_DAYS) -> pd.DataFrame:
    """
    Add lagged sales values as features for XGBoost.

    What a lag feature is: Sales_lag_7 on day T is the actual sales
    value from day T-7. This gives XGBoost explicit knowledge of what
    happened one week ago — which is highly predictive of what will
    happen today, given weekly seasonality.

    CRITICAL: lags must be created PER STORE, not across the whole
    dataframe. Sorting by Store then Date before shifting ensures that
    lag_7 for Store 2 on Monday does not accidentally use Store 1's
    Sunday sales.

    Why these specific lags:
    - Lag 1  : yesterday's sales (short-term momentum)
    - Lag 7  : same day last week (weekly seasonality signal)
    - Lag 14 : two weeks ago (smooths out one-off weekly anomalies)
    - Lag 21 : three weeks ago
    - Lag 28 : four weeks ago (monthly comparison point)
    """
    df = df.copy().sort_values(["Store", "Date"])
    for lag in lag_days:
        col = f"Sales_lag_{lag}"
        df[col] = df.groupby("Store")["Sales"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame,
                         windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add rolling window statistics as features.

    Rolling mean: the average sales over the last N days.
    Smooths out day-to-day noise and gives the model a sense of the
    current demand LEVEL (is demand currently high or low relative to
    its history?).

    Rolling std: the standard deviation of sales over the last N days.
    Tells the model how VOLATILE demand has been recently. High recent
    volatility means the model should be less confident — this is an
    implicit uncertainty signal.

    min_periods=1 prevents NaN at the start of each store's history.
    shift(1) ensures we only use data available BEFORE the prediction
    date — we cannot use today's sales to predict today.
    """
    df = df.copy().sort_values(["Store", "Date"])
    for w in windows:
        df[f"Rolling_mean_{w}"] = (df.groupby("Store")["Sales"]
                                     .transform(lambda x: 
                                         x.shift(1).rolling(w, min_periods=1).mean()))
        df[f"Rolling_std_{w}"]  = (df.groupby("Store")["Sales"]
                                     .transform(lambda x:
                                         x.shift(1).rolling(w, min_periods=1).std()))
    return df


def build_full_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Calls every feature function in the correct order.
    """
    print("Adding date features ...")
    df = add_date_features(df)

    print("Adding German holiday features ...")
    df = add_german_holidays(df)

    print("Adding Promo2 active flag ...")
    df = add_promo2_active(df)

    print("Adding competition features ...")
    df = add_competition_features(df)

    print("Adding lag features ...")
    df = add_lag_features(df)

    print("Adding rolling window features ...")
    df = add_rolling_features(df)

    # Drop rows where lags are NaN (first 28 days of each store's history)
    before = len(df)
    df = df.dropna(subset=[f"Sales_lag_{max(LAG_DAYS)}"])
    print(f"Dropped {before - len(df):,} rows with NaN lags "
          f"(first {max(LAG_DAYS)} days of each store's history)")

    return df


def chronological_split(df: pd.DataFrame, 
                         split_date: str = TRAIN_END_DATE):
    """
    Split the dataset chronologically.

    WHY NOT RANDOM SPLIT: If you randomly assign rows to train/test,
    your training set will contain rows from the future (e.g., Dec 2014)
    and your test set will contain rows from the past (e.g., Jan 2013).
    The model learns future patterns and appears to predict the past very
    accurately — but this is data leakage, not real predictive power.

    The correct split: everything up to split_date is training data;
    everything after is the validation/test set. This mimics real-world
    deployment where the model is always predicting the FUTURE.
    """
    train = df[df["Date"] <= split_date].copy()
    test  = df[df["Date"] >  split_date].copy()
    print(f"Train: {train['Date'].min().date()} to {train['Date'].max().date()} "
          f"— {len(train):,} rows")
    print(f"Test : {test['Date'].min().date()} to {test['Date'].max().date()} "
          f"— {len(test):,} rows")
    return train, test