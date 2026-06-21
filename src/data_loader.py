# src/data_loader.py
# ============================================================
# DATA LOADING MODULE
# Handles all raw data ingestion and initial preparation.
# Called from notebooks — never run directly.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(train_path: str, store_path: str) -> tuple:
    """
    Load the two raw Rossmann files from disk.

    Why parse_dates on Date: pandas reads CSV columns as strings by default.
    Passing parse_dates converts the Date column to a proper datetime object,
    which unlocks .dt.month, .dt.dayofweek etc. without extra conversion steps.

    Why low_memory=False on train: the StateHoliday column has mixed types
    (the string '0' and the integers 1/2/3 appear in different rows depending
    on the dataset source). low_memory=False forces pandas to read the full
    column before deciding the dtype, preventing a DType warning.
    """
    print("Loading train.csv ...")
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)

    print("Loading store.csv ...")
    store = pd.read_csv(store_path)

    print(f"  train shape : {train.shape}")
    print(f"  store shape : {store.shape}")
    return train, store


def merge_datasets(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join store master data onto each transaction row.

    Why left join: every row in train must be kept. A store with no matching
    record in store.csv would be dropped by an inner join — we want to see
    those as NaN so we can investigate rather than silently lose them.
    """
    df = train.merge(store, on="Store", how="left")
    print(f"Merged dataset shape: {df.shape}")
    return df


def filter_open_stores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the store was closed (Open == 0).

    Business reason: on closed days Sales = 0 by definition, not because
    customer demand was zero. Including these zeros would teach the forecast
    model that sales drop to zero on certain days — corrupting every seasonal
    and trend estimate. This is the single most important data filter in the
    entire project.
    """
    before = len(df)
    df_open = df[df["Open"] == 1].copy()
    after = len(df_open)

    print(f"Rows before open filter : {before:,}")
    print(f"Rows after open filter  : {after:,}")
    print(f"Closed-day rows removed : {before - after:,} "
          f"({(before - after) / before * 100:.1f}%)")
    return df_open


def fix_state_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the StateHoliday column.

    Known quirk: depending on which Kaggle source was used, StateHoliday
    may contain the string '0' rather than the integer 0 for non-holidays.
    We convert everything to string and unify so downstream code can use
    df['StateHoliday'] != '0' as a clean boolean mask.
    """
    df["StateHoliday"] = df["StateHoliday"].astype(str).str.strip()
    # Replace integer zero that survived as '0.0' in some sources
    df["StateHoliday"] = df["StateHoliday"].replace("0.0", "0")
    return df


def load_and_prepare_data(train_path: str, store_path: str) -> pd.DataFrame:
    """
    Master pipeline function: load → merge → filter → fix types.
    Call this single function from every notebook to get a clean dataset.
    """
    train, store = load_raw_data(train_path, store_path)
    df = merge_datasets(train, store)
    df = filter_open_stores(df)
    df = fix_state_holiday(df)
    return df