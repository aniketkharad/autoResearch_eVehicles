"""step1_large_data.py - Data loading and high-impact feature engineering for the 668k dataset.

Loads train.csv (668,665 rows) and test.csv (286,571 rows).
Loads splits.npy (deterministic Stratified 5-Fold CV split, seed 101).
Engineers domain-specific EV features while strictly preserving out-of-fold integrity.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step1_large_data")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_LARGE_DIR = BASE_DIR / "cache_large"
RESULTS_LARGE_DIR = BASE_DIR / "results_large"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SPLITS_FILE = DATA_DIR / "splits.npy"

TARGET_COL = "Will_Buy_EV"
ID_COL = "id"

CATEGORICAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

NUMERIC_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]


def load_raw_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Loads raw train and test datasets, target variable, and 5-fold split assignments."""
    logger.info(f"Loading train dataset from {TRAIN_CSV}...")
    df_train = pd.read_csv(TRAIN_CSV)
    logger.info(f"Loading test dataset from {TEST_CSV}...")
    df_test = pd.read_csv(TEST_CSV)

    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"CV splits file not found at {SPLITS_FILE}")
    splits = np.load(SPLITS_FILE)
    assert len(splits) == len(df_train), f"Splits length ({len(splits)}) does not match train length ({len(df_train)})"

    # Map target
    target_mapping = {"No": 0, "Yes": 1, 0: 0, 1: 1}
    y = df_train[TARGET_COL].map(target_mapping).values.astype(np.int64)

    test_ids = df_test[ID_COL].values
    train_ids = df_train[ID_COL].values

    logger.info(
        f"Data loaded: Train shape {df_train.shape} | Test shape {df_test.shape} | "
        f"Target mean: {np.mean(y):.4f} (Positives: {np.sum(y):,})"
    )
    return df_train, y, splits, df_test, test_ids


def engineer_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Engineers high-impact domain features across train and test sets."""
    logger.info("Executing EV domain feature engineering...")

    df_tr = df_train.copy()
    df_te = df_test.copy()

    # Drop id and target from working frames
    for drop_c in [ID_COL, TARGET_COL]:
        if drop_c in df_tr.columns:
            df_tr.drop(columns=[drop_c], inplace=True)
        if drop_c in df_te.columns:
            df_te.drop(columns=[drop_c], inplace=True)

    for df in [df_tr, df_te]:
        # 1. Total Charging & Ratios
        total_charging = df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
        df["charging_stations_total"] = total_charging.astype(np.int32)
        df["charging_home_work_ratio"] = (
            df["Charging_Stations_Near_Home"] / (df["Charging_Stations_Near_Work"] + 1.0)
        ).astype(np.float32)
        df["commute_per_station"] = (
            df["Daily_Commute_km"] / (total_charging + 1.0)
        ).astype(np.float32)
        df["income_per_car"] = (
            df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"] + 1.0)
        ).astype(np.float32)

        # 2. Ordinal & Binary Domain Synergies
        anxiety_map = {"Low": 1, "Medium": 2, "High": 3}
        home_charge_map = {"No": 0, "Yes": 1}
        subsidy_map = {"No": 0, "Yes": 1}

        anxiety_num = df["Range_Anxiety_Level"].map(anxiety_map).fillna(2).astype(np.int32)
        home_charge_num = df["Home_Charging_Possible"].map(home_charge_map).fillna(0).astype(np.int32)
        subsidy_num = df["Subsidy_Available"].map(subsidy_map).fillna(0).astype(np.int32)

        df["unmitigated_range_anxiety"] = (anxiety_num * (1 - home_charge_num)).astype(np.int32)
        df["green_subsidy_synergy"] = (
            df["Environmental_Concern_Level"].astype(np.float32) * subsidy_num
        ).astype(np.float32)

        # 3. High Correlation Interaction Features (+0.636 and +0.423 target correlation)
        env_f = df["Environmental_Concern_Level"].astype(np.float32)
        anx_f = anxiety_num.astype(np.float32)
        sub_f = subsidy_num.astype(np.float32)
        inc_f = df["Annual_Income_USD"].astype(np.float32)

        df["net_green_subsidy"] = (sub_f * (env_f - anx_f)).astype(np.float32)
        df["sub_x_inc"] = (sub_f * (inc_f / 10000.0)).astype(np.float32)

    # 4. Group Aggregation Cohort Differences (computed using train stats to prevent test leakage)
    for grp in ["City_Type", "Current_Car_Type"]:
        grp_inc_mean = df_tr.groupby(grp, observed=False)["Annual_Income_USD"].mean()
        df_tr[f"income_diff_{grp}"] = (df_tr["Annual_Income_USD"] - df_tr[grp].map(grp_inc_mean)).astype(np.float32)
        df_te[f"income_diff_{grp}"] = (df_te["Annual_Income_USD"] - df_te[grp].map(grp_inc_mean)).astype(np.float32)

        grp_com_mean = df_tr.groupby(grp, observed=False)["Daily_Commute_km"].mean()
        df_tr[f"commute_diff_{grp}"] = (df_tr["Daily_Commute_km"] - df_tr[grp].map(grp_com_mean)).astype(np.float32)
        df_te[f"commute_diff_{grp}"] = (df_te["Daily_Commute_km"] - df_te[grp].map(grp_com_mean)).astype(np.float32)

    # Identify final categorical and numeric column sets
    cat_cols = [c for c in CATEGORICAL_COLS if c in df_tr.columns]
    num_cols = [c for c in df_tr.columns if c not in cat_cols]

    # Downcast and type cast
    for col in cat_cols:
        df_tr[col] = df_tr[col].astype("category")
        df_te[col] = df_te[col].astype("category")

    for col in num_cols:
        df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce").astype(np.float32)
        df_te[col] = pd.to_numeric(df_te[col], errors="coerce").astype(np.float32)

    logger.info(
        f"Feature engineering complete. Total features: {df_tr.shape[1]} "
        f"({len(cat_cols)} categorical, {len(num_cols)} numeric)."
    )
    return df_tr, df_te, cat_cols, num_cols


def prepare_large_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, List[str], List[str]]:
    """Master data preparation function for the 668k dataset."""
    CACHE_LARGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_LARGE_DIR.mkdir(parents=True, exist_ok=True)

    df_train_raw, y, splits, df_test_raw, test_ids = load_raw_data()
    X_train, X_test, cat_cols, num_cols = engineer_features(df_train_raw, df_test_raw)

    del df_train_raw, df_test_raw
    gc.collect()

    return X_train, y, splits, X_test, test_ids, cat_cols, num_cols


if __name__ == "__main__":
    X_tr, y, splits, X_te, test_ids, cat_cols, num_cols = prepare_large_data()
    print("Train features shape:", X_tr.shape)
    print("Test features shape:", X_te.shape)
    print("Target shape:", y.shape)
    print("Splits shape:", splits.shape)
    print("Categorical columns:", cat_cols)
    print("Numeric columns count:", len(num_cols))
